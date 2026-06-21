import torch
import gpytorch


class UncertainKernel(gpytorch.kernels.Kernel):
    r"""
    RBF style kernel for one dimensional inputs whose locations are only
    known up to a Gaussian uncertainty.

    Implements the covariance function from Dallaire, Besse and Chaib draa,
    "Learning Gaussian Process Models from Uncertain Data" (ICONIP 2009),
    which marginalizes the input noise out of a standard RBF kernel in
    closed form.

    Each input point is encoded as a 2 column tensor, not a single value:

        x[..., 0] = mean of the input location
        x[..., 1] = variance of the input location (0 for an exactly known point)

    For two points x_i = (mu_i, s_i^2) and x_j = (mu_j, s_j^2), with
    lengthscale l, the kernel is:

        norm_ij = l^2 + s_i^2 + s_j^2
        k(x_i, x_j) = sqrt(l^2 / norm_ij) * exp(-0.5 * (mu_i - mu_j)^2 / norm_ij)

    When s_i = s_j = 0 this reduces exactly to a standard RBF kernel. Larger
    input variance on either point widens norm_ij, which both flattens the
    exponential term and shrinks the sqrt prefactor, so uncertain points
    contribute weaker covariance than precisely known ones.

    Wrap this kernel in gpytorch.kernels.ScaleKernel to add a learned output
    scale (signal variance), matching the example in kernelTesting.ipynb.
    """

    has_lengthscale = True

    @staticmethod
    def _split(x):
        """Split a (..., N, 2) tensor of (mean, variance) pairs into its two columns."""
        mean = x[..., 0:1]
        var = x[..., 1:2]
        return mean, var

    def forward(self, x1, x2, diag=False, **params):
        x1_mean, x1_var = self._split(x1)
        x2_mean, x2_var = self._split(x2)

        l2 = self.lengthscale.pow(2)

        # Squared distance between means, shape (..., N, M), or (..., N) if diag.
        sq_dist = self.covar_dist(x1_mean, x2_mean, diag=diag, square_dist=True)

        if diag:
            # x1_var and x2_var are already aligned point for point.
            norm = (l2 + x1_var + x2_var).squeeze(-1)
        else:
            # Pairwise sum: norm[..., i, j] = l^2 + var1[i] + var2[j]
            norm = l2 + x1_var + x2_var.transpose(-1, -2)

        return (l2 / norm).sqrt() * torch.exp(-0.5 * sq_dist / norm)
