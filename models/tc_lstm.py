from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class TC_LSTM(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(TC_LSTM, self).__init__()
    self.opt = opt
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.lstm_l = DynamicLSTM(
        opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
    self.lstm_r = DynamicLSTM(
        opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
    self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

  def forward(self, inputs):
    x_l, x_r, aspect = inputs[0], inputs[1], inputs[2]
    x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
    aspect_len = torch.tensor(
        torch.sum(aspect != 0, dim=-1), dtype=torch.float).to(self.opt.device)

    x_l, x_r, aspect = self.embed(x_l), self.embed(x_r), self.embed(aspect)

    aspect_pool = torch.div(
        torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
    aspect_l = torch.unsqueeze(
        aspect_pool, dim=1).expand(-1, self.opt.max_seq_len, -1)
    aspect_r = torch.unsqueeze(
        aspect_pool, dim=1).expand(-1, self.opt.max_seq_len, -1)

    x_l = torch.cat((x_l, aspect_l), dim=-1)
    x_r = torch.cat((x_r, aspect_r), dim=-1)
    _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
    _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
    h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
    out = self.dense(h_n)
    return out
