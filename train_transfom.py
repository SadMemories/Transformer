import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import build_transformer
# from transformer import make_model
# from bei_model import build_transformer

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()

        self.smoothing = smoothing
        self.loss_function = nn.KLDivLoss(reduction='sum')

    # pred: (batch x seq_len, vocab_num)  target: (batch x seq_len, )
    def forward(self, pred, target):
        vocab_num = pred.shape[-1]

        true_list = pred.data.clone()
        true_list = true_list.fill_(self.smoothing / (vocab_num - 2))
        true_list = true_list.scatter_(1, target.data.unsqueeze(1), 1-self.smoothing)
        true_list[:, 0] = 0
        mask = torch.nonzero(target.data == 0)

        if mask.dim() > 0:
            true_list.index_fill_(0, mask.squeeze(), 0.0)

        return self.loss_function(pred, true_list)


def get_inp_mask(bat_seq, pad):
    return (bat_seq != pad).unsqueeze(-2)

'''
def subsquent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
'''


def subsquent_mask(inp_tgt):
    batch_size = inp_tgt.shape[0]
    tgt_mask_box = torch.ones((batch_size, inp_tgt.shape[1], inp_tgt.shape[1]))

    return torch.tril(tgt_mask_box, diagonal=0).to(bool)


def prepare_data(vocab_num, max_seq_len, batch_size, iter_num, pad=0):

    for i in range(iter_num):
        inp = torch.randint(2, vocab_num, size=(batch_size, max_seq_len))
        inp_mask = get_inp_mask(inp, pad)  # bt x 1 x max_seq

        tgt = inp.data.clone()
        tgt[:, 0] = 1  # tgt的首位为初始字符
        inp_tgt = tgt[:, :-1].data.clone()
        out_tgt = tgt[:, 1:].data.clone()
        n_tokens = (out_tgt != pad).data.sum()

        tgt_mask = get_inp_mask(inp_tgt, pad) & subsquent_mask(inp_tgt)
        # tgt_mask_box = torch.ones((batch_size, inp_tgt.shape[1], inp_tgt.shape[1]))
        # tgt_mask = get_inp_mask(inp_tgt, pad) & torch.tril(tgt_mask_box, diagonal=0).to(bool)
        # tgt_mask = torch.tril(tgt_mask_box, diagonal=0).to(bool)

        yield {"inp": inp, "inp_tgt": inp_tgt, "out_tgt": out_tgt,
               "inp_mask": inp_mask, "tgt_mask": tgt_mask, "n_tokens": n_tokens}


def train_epoch(model, criterion, data_iter, device, vocab_num, optimizer, iter_num, pad=0):

    total_loss = 0.0
    total_tokens = 0
    for ind, batch_data in enumerate(data_iter):
        '''
        tgt_pred = model.forward(batch_data['inp'].to(device), batch_data['inp_tgt'].to(device),
                         batch_data['inp_mask'].to(device), batch_data['tgt_mask'].to(device))

        tgt_pred = model.generator(tgt_pred)
        '''
        tgt_pred = model(batch_data['inp'].to(device), batch_data['inp_tgt'].to(device),
                                 batch_data['inp_mask'].to(device), batch_data['tgt_mask'].to(device))
        tgt_gt = batch_data['out_tgt'].to(device)  # batch x (max_seq_len - 1)
        n_tokens = batch_data['n_tokens']

        tgt_pred = tgt_pred.contiguous().view(-1, vocab_num)
        tgt_gt = tgt_gt.contiguous().view(-1)

        loss = criterion(tgt_pred, tgt_gt)
        total_tokens += n_tokens
        loss /= n_tokens
        loss.backward()
        total_loss += loss.item() * n_tokens

        optimizer.step()
        optimizer.zero_grad()

        if (ind+1) % 50 == 0:
            print(f"epoch step: {ind+1} loss: {loss.item()}")
    return total_loss / total_tokens


def valid_epoch(model, criterion, data_iter, device, vocab_num, iter_num, pad):

    valid_loss = 0.0
    total_tokens = 0
    for ind, batch_data in enumerate(data_iter):
        '''
        tgt_pred = model.forward(batch_data['inp'].to(device), batch_data['inp_tgt'].to(device),
                         batch_data['inp_mask'].to(device), batch_data['tgt_mask'].to(device))
        tgt_pred = model.generator(tgt_pred)
        '''
        tgt_pred = model(batch_data['inp'].to(device), batch_data['inp_tgt'].to(device),
                                 batch_data['inp_mask'].to(device), batch_data['tgt_mask'].to(device))
        tgt_gt = batch_data['out_tgt'].to(device)  # batch x (max_seq_len - 1)
        n_tokens = batch_data['n_tokens']

        tgt_pred = tgt_pred.contiguous().view(-1, vocab_num)
        tgt_gt = tgt_gt.contiguous().view(-1)

        loss = criterion(tgt_pred, tgt_gt)

        total_tokens += n_tokens
        valid_loss += loss.item()

    return valid_loss / total_tokens


def train():
    vocab_num = 11
    max_seq_len = 15
    batch_size = 32
    n_epoch = 40
    train_iter_num_per_epoch = 30
    valid_iter_num_per_epoch = 10
    pad = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_transformer(vocab_num, vocab_num, N=2)
    # model = make_model(vocab_num, vocab_num, N=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    criterion = LabelSmoothing(0.0)
    model.to(device)

    for epoch in range(n_epoch):
        model.train()
        train_iter = prepare_data(vocab_num, max_seq_len, batch_size, train_iter_num_per_epoch, pad)
        train_loss = train_epoch(model, criterion, train_iter, device,
                                 vocab_num, optimizer, train_iter_num_per_epoch, pad)
        print(f"Epoch Loss: {train_loss}")

        model.eval()
        valid_iter = prepare_data(vocab_num, max_seq_len, batch_size, valid_iter_num_per_epoch, pad)
        val_loss = valid_epoch(model, criterion, valid_iter, device, vocab_num, valid_iter_num_per_epoch, pad)

        print(f"valid loss: {val_loss}")

    model.eval()
    inp_data = torch.LongTensor([[6,2,3,4,5,6,7,8,9,10]]).cuda()
    src_mask = torch.ones((1, 1, 10)).to(bool).cuda()

    memory = model.encoder(model.pos_embeds[0](model.src_embed(inp_data)), src_mask)
    # memory = model.encode(inp_data, src_mask)
    ys = torch.ones(1, 1).fill_(1).type_as(inp_data)

    for i in range(9):
        pred = model.decoder(model.pos_embeds[1](model.tgt_embed(ys)), memory, src_mask, subsquent_mask(ys).cuda())
        # pred= model.decode(memory, src_mask, ys, subsquent_mask(ys).cuda())
        prob = model.generator(pred[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(inp_data.data).fill_(next_word)], dim=1)
    print(ys[:, 1:])

    '''
    for ind, batch_data in enumerate(data_iter):
        tgt_pred = model(batch_data['inp'].to(device), batch_data['inp_tgt'].to(device),
                         batch_data['inp_mask'].to(device), batch_data['tgt_mask'].to(device))
        print(tgt_pred.shape)

        if ind == 0:
            break
    '''


if __name__ == '__main__':
    train()