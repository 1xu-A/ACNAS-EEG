import torch
import torch.nn as nn

# OPS is a set of layers with same input/output channel.

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),

    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'avg_pool_3x1': lambda C, stride, affine: nn.AvgPool2d((3, 1), stride=stride, padding=(1, 0), affine=affine),
    'max_pool_3x1': lambda C, stride, affine: nn.MaxPool2d((3, 1), stride=stride, padding=(1, 0), affine=affine),
    'sep_conv_3x1': lambda C, stride, affine: SepConv(C, C, (3, 1), stride, (1, 0), affine=affine),
    'sep_conv_5x1': lambda C, stride, affine: SepConv(C, C, (5, 1), stride, (2, 0), affine=affine),
    'sep_conv_7x1': lambda C, stride, affine: SepConv(C, C, (7, 1), stride, (3, 0), affine=affine),
    'sep_conv_15x1': lambda C, stride, affine: SepConv(C, C, (15, 1), stride, (7, 0), affine=affine),
    'sep_conv_17x1': lambda C, stride, affine: SepConv(C, C, (17, 1), stride, (8, 0), affine=affine),
    'sep_conv_33x3': lambda C, stride, affine: SepConv(C, C, (33, 3), stride, (16, 1), affine=affine),
    'dil_conv_3x1': lambda C, stride, affine: DilConv(C, C, (3, 1), stride, (2, 0), 2, affine=affine),  # 5x5
    'dil_conv_5x1': lambda C, stride, affine: DilConv(C, C, (5, 1), stride, (4, 0), 2, affine=affine),  # 9x9
    'dil_conv_7x1': lambda C, stride, affine: DilConv(C, C, (7, 1), stride, (6, 0), 2, affine=affine),  # 5x5
    'dil_conv_9x1': lambda C, stride, affine: DilConv(C, C, (9, 1), stride, (8, 0), 2, affine=affine),  # 9x9
    'dil_conv_11x1': lambda C, stride, affine: DilConv(C, C, (11, 1), stride, (10, 0), 2, affine=affine),  # 9x9
    'max_pool_1x3': lambda C, stride, affine: nn.MaxPool2d((1, 3), stride, (0, 1)),
    'sep_conv_1x3': lambda C, stride, affine: SepConv(C, C, (1, 3), stride, (0, 1), affine=affine),
    'sep_conv_1x5': lambda C, stride, affine: SepConv(C, C, (1, 5), stride, (0, 2), affine=affine),
    'sep_conv_1x7': lambda C, stride, affine: SepConv(C, C, (1, 7), stride, (0, 3), affine=affine),
    'sep_conv_1x15': lambda C, stride, affine: SepConv(C, C, (1, 15), stride, (0, 7), affine=affine),
    'sep_conv_1x17': lambda C, stride, affine: SepConv(C, C, (1, 17), stride, (0, 8), affine=affine),
    'dil_conv_1x3': lambda C, stride, affine: DilConv(C, C, (1, 3), stride, (0, 2), 2, affine=affine),  # 5x5
    'dil_conv_1x5': lambda C, stride, affine: DilConv(C, C, (1, 5), stride, (0, 4), 2, affine=affine),  # 9x9
    'dil_conv_1x7': lambda C, stride, affine: DilConv(C, C, (1, 7), stride, (0, 6), 2, affine=affine),  # 5x5
    'dil_conv_1x9': lambda C, stride, affine: DilConv(C, C, (1, 9), stride, (0, 8), 2, affine=affine),  # 9x9
    'dil_conv_1x11': lambda C, stride, affine: DilConv(C, C, (1, 11), stride, (0, 10), 2, affine=affine),  # 9x9
}
'''
OPS = {
  'none': lambda C: Zero(),
  'skip_connect': lambda C : Identity(),
  # 'cnn': lambda C: CNN(C, C, (1, 3), 1),
  # 'lstm': lambda C: LSTM(C, C),
  'scnn' : lambda C : SCNN(C, C, (3, 3), 1),
  'dgcn': lambda C, cheb, nodevec1, nodevec2, alpha: Dgcn(2, cheb, C, C, nodevec1, nodevec2, alpha),
}
'''
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=C_in, bias=False),
            # depth-wise conv
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0,
                      bias=False),  # point-wise conv
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
'''
class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return x.mul(0.)
'''

class Zero(nn.Module):
    """
    zero by stride
    """

    def __init__(self, stride):
        super(Zero, self).__init__()

        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

'''
class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    n, c, h, w = x.size()
    h //= self.stride
    w //= self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h, w).fill_(0)
    return padding 
'''

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()

        assert C_out % 2 == 0

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, :, :])], dim=1)                    ###########################################[:, :, 1:, 1:]
        out = self.bn(out)
        return out


class SCNN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, padding, affine=False):
      super(SCNN, self).__init__()
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, 64, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        nn.Conv2d(256, C_in, kernel_size=kernel_size, stride=1, padding=0, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=1, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
      )
    def forward(self, x):
      return self.op(x)


class CNN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):

        x = self.relu(x)
        output = self.filter_conv(x)
        output = self.bn(output)
        print("CNN OUTPUT", output.shape)
        return output

class CausalConv2d(nn.Conv2d):
    """
    单向padding
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           padding=(0, self._padding),
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result

class Dgcn(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        super(Dgcn, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(4, 4).to(self.DEVICE)) for _ in range(K)])
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)
    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)

        batch_size, num_nodes, c_in, timesteps = x.shape
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]
            output = torch.zeros(
                batch_size, num_nodes, 4).to(self.DEVICE)

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k] , dim=0)
                T_k = alpha * self.cheb_polynomials[k] + beta * adp
                self.theta_k = self.Theta[k]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)

            outputs.append(output.unsqueeze(-1))
        outputs = F.relu(torch.cat(outputs, dim=-1)) # Concatenate the output of each time step
        outputs = self.bn(outputs)
        print("DGCNN OUTPUT", output.shape)

        return outputs