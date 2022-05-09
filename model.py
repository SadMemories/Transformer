import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return [copy.deepcopy(module) for _ in range(n)]


# word embedding
class WordEmbedding(nn.Module):

    def __init__(self, num_vocab, dim=512):
        super(WordEmbedding, self).__init__()
        self.model_dim = dim
        self.embedding = nn.Embedding(num_vocab, dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dim)


class PositionEmbedding(nn.Module):

    def __init__(self, max_len, dim=512):
        super(PositionEmbedding, self).__init__()

        pos_mat = torch.zeros((max_len, dim))
        pos = torch.arange(max_len).unsqueeze(1)
        # div_item = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000) / dim))
        div_item = 1 / torch.pow(10000, torch.arange(0, dim, 2) / dim)
        pos_mat[:, 0::2] = torch.sin(pos * div_item)
        pos_mat[:, 1::2] = torch.cos(pos * div_item)

        pos_mat = pos_mat.unsqueeze(0)
        self.register_buffer('pos_mat', pos_mat)

    def forward(self, x):
        return x + self.pos_mat


def attention(Q, K, V, mask):

    dim = Q.shape[-1]
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
    mask_score = score.masked_fill(mask, -1e9)
    soft_score = F.softmax(mask_score, -1)
    return torch.matmul(soft_score, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, H, dim=512):
        super(MultiHeadAttention, self).__init__()

        assert dim % H == 0, "Dimension is not divisible"

        self.each_head_dim = dim // H
        self.H = H
        self.Linears = clones(nn.Linear(dim, dim), 4)

    def forward(self, Q, K, V, mask):  # mask: bt x seq_n x seq_n

        mask = mask.unsqueeze(1)
        batch = Q.shape[0]
        query, key, value = [linear(x).view(batch, -1, self.H, self.each_head_dim).transpose(1, 2)
                             for linear, x in zip(self.Linears, (Q, K, V))]
        x = attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch, -1, self.H * self.each_head_dim)

        x = self.Linears[-1](x)

        return x


class FeedFoward(nn.Module):

    def __init__(self, dim=512):
        super(FeedFoward, self).__init__()

        self.Linear1 = nn.Linear(dim, 4 * dim)
        self.Linear2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.Linear2(F.relu(self.Linear1(x)))


class LayerNorm(nn.Module):

    def __init__(self, dim=512, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)

        return self.g * (x - x_mean) / (x_std + self.eps) + self.b


class EncoderLayer(nn.Module):

    def __init__(self, multi_atten, feed_foward, LayerNorm, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attn = multi_atten
        self.LayerNorm = LayerNorm
        self.feed_forward = feed_foward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        out = self.multi_head_attn(x, x, x, mask)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        output = self.LayerNorm(out + self.dropout(self.feed_forward(out)))
        return output


class Encoder(nn.Module):

    def __init__(self, encoder_layer, N=6):
        super(Encoder, self).__init__()

        self.encoder_layers = clones(encoder_layer, N)

    def forward(self, x, mask):

        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


def build_transformer(src_max_len, tgt_max_len, N=6, h=8, d_model=512, dropout=0.1):
    model = []

    # 输入数据准备工作
    batch_input = torch.tensor([2, 3, 4])
    seq_max_len = max(batch_input)
    ind_max_vacab = 6  # 单词的总数量为6
    # inp : bt x seq_max_len
    inp = torch.cat([F.pad(torch.randint(1, 6, (seq_len,)).reshape(1, -1), (0, seq_max_len - seq_len, 0, 0))
                     for seq_len in batch_input])
    # mask = inp > 0
    mask = torch.bmm(inp.unsqueeze(2), inp.unsqueeze(2).transpose(1, 2)) > 0

    # 模型搭建
    multi_attn = MultiHeadAttention(H=8)
    encoder = Encoder(EncoderLayer(multi_attn, FeedFoward(), LayerNorm()), 6)
    word_embed = WordEmbedding(ind_max_vacab, 512)
    pos_embed = PositionEmbedding(seq_max_len, 512)

    # 前向传播工作
    embed_pos_inp = pos_embed(word_embed(inp))
    encoder_out = encoder(embed_pos_inp, mask)

    print(encoder_out.shape)
    return model


def main():
    src_max_len = 10
    tgt_max_len = 10
    model = build_transformer(src_max_len, tgt_max_len)


if __name__ == '__main__':
    main()