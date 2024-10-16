import torch

from gpytorch.means import Mean


class ZeroMean(Mean):
    def __init__(self, batch_shape=torch.Size(), **kwargs):
        super(ZeroMean, self).__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        input = torch.unsqueeze(input[:,0], 1)
        if len(input.shape)>2:
            input = torch.squeeze(input)
        mean = torch.zeros(*self.batch_shape, 1, dtype=input.dtype, device=input.device)
        if input.shape[:-2] == self.batch_shape:
            return mean.expand(input.shape[:-1])
        else:
            return mean.expand(torch.broadcast_shapes(input.shape[:-1], mean.shape))