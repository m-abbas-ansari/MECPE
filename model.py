import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, einsum
from einops import rearrange, repeat, pack
from enum import Enum
import numpy as np

class TokenTypes(Enum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    FUSION = 3

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        pad_mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_x = x

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim = -1))

        q, k, v = map(lambda t:
                      rearrange(t, 'b n (h d) -> b h n d', h = self.heads),
                      (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        if pad_mask is not None:
          # print(pad_mask.size())
          sim = sim.masked_fill(pad_mask, -torch.finfo(sim.dtype).max)

        if attn_mask is not None:
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Zorro_AVT(nn.Module):
    """
      modified from: https://github.com/lucidrains/zorro-pytorch/blob/main/zorro_pytorch/zorro_pytorch.py
    """
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_fusion_tokens = 8,
        out_dim = 768,
        DEVICE = torch.device("cuda")
      ):
        super().__init__()

        # fusion tokens
        self.num_fusion_tokens = num_fusion_tokens
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim).to(DEVICE))

        self.fusion_mask = (torch.ones(num_fusion_tokens) == 1).to(DEVICE)
    
        # transformer
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)
        self.out_layer = nn.Linear(dim, out_dim, bias = False)

    def forward(
        self,
        video_tokens,
        audio_tokens,
        text_tokens,
        video_mask,
        audio_mask,
        text_mask
    ):
        # per utterance [Nc, Nv, dim]
        batch, device = video_tokens.shape[0], video_tokens.device

        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b = batch)
        fusion_mask = repeat(self.fusion_mask, 'n -> b n', b = batch)

        # construct all tokens
        video_tokens, audio_tokens, text_tokens, fusion_tokens = \
        map(lambda t:
            rearrange(t, 'b ... d -> b (...) d'),
            (video_tokens, audio_tokens, text_tokens, fusion_tokens))


        tokens, ps = pack((
            video_tokens,
            audio_tokens,
            text_tokens,
            fusion_tokens
        ), 'b * d') # [Nc, Nv + Na + Nt + Nf, d]

        # construct mask (thus zorro)
        token_types = torch.tensor(list((
            *((TokenTypes.VIDEO.value,) * video_tokens.shape[-2]),
            *((TokenTypes.AUDIO.value,) * audio_tokens.shape[-2]),
            *((TokenTypes.TEXT.value,) * text_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
        )), device = device, dtype = torch.long)


        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # the logic goes
        # every modality, including fusion can attend to self
        zorro_mask = token_types_attend_from == token_types_attend_to

        # fusion can attend to everything
        zorro_mask = zorro_mask | (token_types_attend_from == TokenTypes.FUSION.value)

        # construct padding mask
        pad_mask = torch.cat((video_mask, audio_mask, text_mask, fusion_mask), -1)[:, None, None, :]

        # attend and feedforward
        for attn, ff in self.layers:
            tokens = attn(tokens, pad_mask = pad_mask, attn_mask = zorro_mask) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.norm(tokens)
        fusion_tokens = tokens[:, -self.num_fusion_tokens:, :]
        pooled_tokens = torch.mean(fusion_tokens, 1)

        return self.out_layer(pooled_tokens)

class GraphAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        device = feat_in.device
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(F.tanh(h), self.w_src)
        attn_dst = torch.matmul(F.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        adj = torch.FloatTensor(adj).to(device)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask.bool(), -999)

        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = F.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'

class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)

        return doc_sents_h


class RankNN(nn.Module):
    def __init__(self, configs):
        super(RankNN, self).__init__()
        self.K = configs.K
        self.pos_emb_dim = configs.pos_emb_dim
        self.pos_layer = nn.Embedding(2*self.K + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.rank_feat_dim = 2*self.feat_dim + self.pos_emb_dim
        self.rank_layer1 = nn.Linear(self.rank_feat_dim, self.rank_feat_dim)
        self.rank_layer2 = nn.Linear(self.rank_feat_dim, 1)

    def forward(self, doc_sents_h):
        batch, _, _ = doc_sents_h.size()
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        couples = torch.cat([couples, rel_pos_emb], dim=2)

        couples = F.relu(self.rank_layer1(couples))
        couples_pred = self.rank_layer2(couples)
        return couples_pred.squeeze(2), emo_cau_pos

    def couple_generator(self, H, k):
        batch, seq_len, feat_dim = H.size()
        device = H.device
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(device)
        emo_pos = torch.LongTensor(emo_pos).to(device)
        cau_pos = torch.LongTensor(cau_pos).to(device)

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(device)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)

            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)
        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        device = rel_pos.device
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(device)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))


class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
