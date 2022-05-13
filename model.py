import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# word embedding
class WordEmbedding(nn.Module):

    def __init__(self, num_vocab, dim=512):
        super(WordEmbedding, self).__init__()
        self.model_dim = dim
        self.embedding = nn.Embedding(num_vocab, dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dim)


class PositionEmbedding(nn.Module):

    def __init__(self, max_len=40000, dim=512, dropout=0.1):
        super(PositionEmbedding, self).__init__()

        pos_mat = torch.zeros((max_len, dim))
        pos = torch.arange(max_len).unsqueeze(1)
        # div_item = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000) / dim))
        div_item = 1 / torch.pow(10000, torch.arange(0, dim, 2) / dim)
        pos_mat[:, 0::2] = torch.sin(pos * div_item)
        pos_mat[:, 1::2] = torch.cos(pos * div_item)
        pos_mat = pos_mat.unsqueeze(0)
        self.register_buffer('pos_mat', pos_mat)

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(dim=dim)

    def forward(self, x):
        return self.LayerNorm(self.dropout(x + self.pos_mat[:, :x.size(1)]))


def attention(Q, K, V, mask, dropout):

    dim = Q.shape[-1]
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
    mask_score = score.masked_fill(mask == 0, -1e9)
    soft_score = dropout(F.softmax(mask_score, -1))
    return torch.matmul(soft_score, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, H, dim=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert dim % H == 0, "Dimension is not divisible"

        self.each_head_dim = dim // H
        self.dropout = dropout
        self.H = H
        self.Linears = clones(nn.Linear(dim, dim), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):  # mask: bt x seq_n x seq_n

        mask = mask.unsqueeze(1)
        batch = Q.shape[0]
        query, key, value = [linear(x).view(batch, -1, self.H, self.each_head_dim).transpose(1, 2)
                             for linear, x in zip(self.Linears, (Q, K, V))]
        x = attention(query, key, value, mask, self.dropout)

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
        self.LayerNorms = clones(LayerNorm, 2)
        self.dropouts = clones(nn.Dropout(dropout), 2)
        self.feed_forward = feed_foward

    def forward(self, x, mask):

        out = self.multi_head_attn(x, x, x, mask)
        out = self.dropouts[0](out)
        out = self.LayerNorms[0](out + x)
        output = self.LayerNorms[1](out + self.dropouts[1](self.feed_forward(out)))
        return output


class Encoder(nn.Module):

    def __init__(self, encoder_layer, N=6):
        super(Encoder, self).__init__()

        self.encoder_layers = clones(encoder_layer, N)

    def forward(self, x, mask):

        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, multi_attn, feed_foward, LayerNorm, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.multi_attns = clones(multi_attn, 2)
        self.feed_foward = feed_foward
        self.LayerNorms = clones(LayerNorm, 3)
        self.dropouts = clones(nn.Dropout(dropout), 3)

    def forward(self, x, memory, mask, tgt_mask):
        out = self.multi_attns[0](x, x, x, tgt_mask)
        out = self.LayerNorms[0](self.dropouts[0](out) + x)

        intra_out = self.multi_attns[1](out, memory, memory, mask)
        intra_out = self.LayerNorms[1](self.dropouts[1](intra_out) + out)

        feed_out = self.LayerNorms[2](self.dropouts[2](self.feed_foward(intra_out)) + intra_out)
        return feed_out


class Decoder(nn.Module):

    def __init__(self, decoder_layer, N=6):
        super(Decoder, self).__init__()

        self.decoder_layer = clones(decoder_layer, N)

    def forward(self, x, memory, mask, tgt_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, mask, tgt_mask)
        return x


class Generator(nn.Module):

    def __init__(self, vocab, dim=512):
        super(Generator, self).__init__()

        self.Linear = nn.Linear(dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.Linear(x), dim=-1)


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_embed, generator, dim=512):
        super(EncoderDecoder, self).__init__()

        self.d_model = dim
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.pos_embeds = clones(pos_embed, 2)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_out = self.encoder(self.pos_embeds[0](self.src_embed(src)), src_mask)
        decoder_out = self.decoder(self.pos_embeds[1](self.tgt_embed(tgt)), encoder_out, src_mask, tgt_mask)

        output = self.generator(decoder_out)
        return output


def build_transformer(src_vocab, tgt_vocab, N=6, h=8, d_model=512, dropout=0.1):
    # model = []

    model = EncoderDecoder(
        Encoder(EncoderLayer(MultiHeadAttention(H=h, dim=d_model, dropout=dropout),
                             FeedFoward(dim=d_model), LayerNorm(dim=d_model), dropout=dropout), N=N),
        Decoder(DecoderLayer(MultiHeadAttention(H=h, dim=d_model, dropout=dropout),
                             FeedFoward(dim=d_model), LayerNorm(dim=d_model), dropout=dropout), N=N),
        src_embed=WordEmbedding(src_vocab, dim=d_model), tgt_embed=WordEmbedding(tgt_vocab, dim=d_model),
        pos_embed=PositionEmbedding(dim=d_model, dropout=dropout),
        generator=Generator(tgt_vocab, dim=d_model), dim=d_model
    )

    # 模型初始化工作
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    '''
    # Encoder和Decoder的nn.Embedding在初始化时共享同一个权重 在实际中使用下面的权重共享会令训练变慢
    model.src_embed.embedding.weight = model.tgt_embed.embedding.weight
    model.generator.Linear.weight = model.tgt_embed.embedding.weight
    '''

    """
    # 输入数据准备工作
    batch_input = torch.tensor([2, 3, 4])
    seq_max_len = max(batch_input)
    ind_max_vacab = 6  # 单词的总数量为6
    # inp : bt x seq_max_len
    inp = torch.cat([F.pad(torch.randint(1, 6, (seq_len,)).reshape(1, -1), (0, seq_max_len - seq_len, 0, 0))
                     for seq_len in batch_input])
    mask = torch.bmm(inp.unsqueeze(2), inp.unsqueeze(2).transpose(1, 2)) > 0

    # 模型搭建
    multi_attn = MultiHeadAttention(H=8)
    encoder = Encoder(EncoderLayer(multi_attn, FeedFoward(), LayerNorm()), 6)
    word_embed = WordEmbedding(ind_max_vacab, 512)
    pos_embed = PositionEmbedding(seq_max_len, 512)

    # 前向传播工作
    embed_pos_inp = pos_embed(word_embed(inp))
    encoder_out = encoder(embed_pos_inp, mask)

    print(encoder_out)
    print(encoder_out.shape)
    """
    return model


def main():
    src_vocab = 10
    tgt_vocab = 10
    model = build_transformer(src_vocab, tgt_vocab, 2)
    print(model)


if __name__ == '__main__':
    main()