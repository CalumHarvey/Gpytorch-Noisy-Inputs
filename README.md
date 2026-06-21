# GPyTorch Noisy Inputs

A GPyTorch extension for Gaussian Process regression when the training and test inputs are themselves uncertain, not just the outputs.

Standard GPs assume inputs are known exactly and only the targets carry noise. This repo implements the kernel from Dallaire, Besse and Chaib draa, "Learning Gaussian Process Models from Uncertain Data" (2009), which lets each input carry its own mean and variance, and folds that variance into the covariance computation directly.

## How it works

Each input point is encoded as a pair instead of a single value:

```
x = (mean, variance)
```

A training set of N points is therefore an N x 2 tensor, not an N x 1 tensor. The custom kernel and mean modules in this repo know how to split that tensor back into its mean and variance components.

```
(mean, variance) pairs
        |
        v
+-------------------+       +---------------------------+
| Mean module       |       | Covariance module         |
| ZeroMean /         |       | ScaleKernel(UncertainKernel)|
| ConstantMean       |       | uses mean AND variance    |
| uses mean only     |       +---------------------------+
+-------------------+                    |
        |                                |
        v                                v
              gpytorch.models.ExactGP
                       |
                       v
         GaussianLikelihood + Adam optimizer
         (fits lengthscale, outputscale, noise)
```

The kernel itself is a noise aware version of the RBF kernel. Wider input variance on either point spreads the effective covariance, so points with uncertain locations are treated as less informative neighbors than points known precisely.

## Files

| File | Purpose |
|---|---|
| `UncertainKernel.py` | The kernel itself, accepts `(mean, variance)` pairs and outputs a covariance matrix |
| `UncertainMeanZero.py` | A zero mean function adapted to read the mean column of the input pair |
| `UncertainMeanConstant.py` | A learned constant mean function adapted the same way |
| `kernelTesting.ipynb` | End to end example: fits a noisy sine curve and plots predictions |

## Usage

```python
import torch
import gpytorch
import UncertainKernel
import UncertainMeanZero

# inputs as (mean, variance) pairs
train_x = torch.stack((train_x_mean, train_x_var), dim=1)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = UncertainMeanZero.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(UncertainKernel.UncertainKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

See `kernelTesting.ipynb` for a full worked example, including training and plotting predictions with uncertainty bars.

Note: at prediction time, test inputs must be padded with a variance column (zero if the test input is known exactly). Training and test batches can be different sizes.

## Requirements

- torch
- gpytorch
- matplotlib (for the example notebook)

## Status

This is a research style proof of concept, not a packaged library. There is no `setup.py` or `pip` install path. Drop the `.py` files into your project and import them directly.

## Reference

Dallaire, P., Besse, C., and Chaib draa, B. "Learning Gaussian Process Models from Uncertain Data." International Conference on Neural Information Processing (ICONIP), 2009.
