import torch
import torch.nn as nn


class GLU_Block(nn.Module):

  def __init__(self, kernel_size, in_channels, out_channels, downbot):
    super().__init__()

    if in_channels == out_channels:
      self.use_projection = False
    else:
      self.use_projection = True

    self.convresid = nn.utils.weight_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)

    self.leftpad = nn.ConstantPad2d(
        (0, 0, kernel_size - 1, 0),
        0)  #(paddingLeft, paddingRight, paddingTop, paddingBottom)

    self.convx1a = nn.utils.weight_norm(
        nn.Conv2d(in_channels, int(in_channels / downbot), kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convx2a = nn.utils.weight_norm(
        nn.Conv2d(in_channels, int(in_channels / downbot), kernel_size=(1, 1)),
        name='weight',
        dim=0)

    self.convx1b = nn.utils.weight_norm(
        nn.Conv2d(
            int(in_channels / downbot),
            int(in_channels / downbot),
            kernel_size=(kernel_size, 1)),
        name='weight',
        dim=0)
    self.convx2b = nn.utils.weight_norm(
        nn.Conv2d(
            int(in_channels / downbot),
            int(in_channels / downbot),
            kernel_size=(kernel_size, 1)),
        name='weight',
        dim=0)

    self.convx1c = nn.utils.weight_norm(
        nn.Conv2d(int(in_channels / downbot), out_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convx2c = nn.utils.weight_norm(
        nn.Conv2d(int(in_channels / downbot), out_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)

  def forward(self, x):
    residual = x
    if self.use_projection:
      residual = self.convresid(residual)
    x = self.leftpad(x)  # [bs, in_channels, max_seq_len+(k-1), 1]
    x1 = self.convx1c(self.convx1b(self.convx1a(x)))
    x2 = self.convx2c(self.convx2b(self.convx2a(x)))
    x2 = torch.sigmoid(x2)
    x = torch.mul(x1, x2)
    return x + residual
