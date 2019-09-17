import torch
import torch.nn as nn
from layers.glu import GLU_Block
from layers.attention import NoQueryAttention
from layers.squeeze_embedding import SqueezeEmbedding


class Gated_CNN(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(Gated_CNN, self).__init__()
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))

    self.in_layer = GLU_Block(opt.kernel_size, opt.embed_dim, opt.in_channels,
                              opt.downbot)
    self.squeeze_embedding = SqueezeEmbedding()
    self.glu_layers = self.make_glu_layers(opt)
    self.attention = NoQueryAttention(
        opt.in_channels, score_function='bi_linear')
    self.dense = nn.Linear(opt.in_channels, opt.polarities_dim)

  def make_glu_layers(self, opt):
    layers = [
        GLU_Block(opt.kernel_size, opt.in_channels, opt.in_channels,
                  opt.downbot) for i in range(opt.num_layers)
    ]
    return nn.Sequential(*layers)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    x_len = torch.sum(text_raw_indices != 0, dim=-1)
    # [batch, max_seq_len] -> [batch, max_seq_len, embed_dim]
    x = self.embed(text_raw_indices)
    x = self.squeeze_embedding(x, x_len)
    # [batch, max_seq_len, embed_dim] -> [batch, embed_dim, max_seq_len]
    x = torch.transpose(x, 1, 2)
    x = x.unsqueeze(3)
    x = self.in_layer(x)  #[batch, in_channel, max_seq_len, 1]
    x = self.glu_layers(x)  #[batch, in_channel, max_seq_len, 1]
    x = torch.squeeze(x, 3)
    x = torch.transpose(x, 1, 2)  #[batch, max_seq_len, out_channel]

    _, score = self.attention(x)
    mat = torch.bmm(score, x)
    out = torch.squeeze(mat, dim=1)
    out = self.dense(out)
    return out
