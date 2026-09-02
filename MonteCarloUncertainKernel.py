import torch
import gpytorch


class MonteCarloUncertainKernel(gpytorch.kernels.Kernel):
    r"""
    Monte Carlo marginalization of input noise through a stock RBFKernel.

    This is the "vanilla gpytorch" baseline for noisy-input GP regression: rather than deriving a
    closed-form kernel that marginalizes input uncertainty analytically (see UncertainKernel.py), this
    kernel estimates a marginal covariance by drawing `n_samples` joint hypotheses of the training
    points' true locations and averaging a plain `gpytorch.kernels.RBFKernel` Gram matrix evaluated on
    each hypothesis. No new covariance formula is derived; every covariance number that comes out of this
    kernel is produced by the standard library's own RBFKernel.

    For off-diagonal entries (i != j) this estimates the same quantity UncertainKernel computes in closed
    form: E_{x_i, x_j}[k_RBF(x_i, x_j)] for independent x_i ~ N(mu_i, s_i^2), x_j ~ N(mu_j, s_j^2).

    On the diagonal (i == j) it does NOT match UncertainKernel numerically, and this is a real modeling
    difference, not a bug: each Monte Carlo draw uses one consistent sampled location per physical point
    (needed so every per-draw Gram matrix is a literal, valid RBF kernel matrix -- averaging over draws
    then stays PSD, which exact GP training requires). That makes k(x_i, x_i) trivially 1 in every draw.
    UncertainKernel's closed form instead applies its pairwise-independent-expectation formula uniformly,
    including at i == j, giving a self-covariance < 1 there. That is itself Dallaire et al.'s own
    approximation -- an exact expectation over the *joint* distribution of all true locations has no closed
    form in general, so their paper's closed-form kernel evaluates each pairwise expectation independently
    rather than jointly. This Monte Carlo kernel's shared-per-draw sampling is, if anything, the more
    literal joint marginalization; it just isn't the same quantity UncertainKernel's formula targets.

    Same 2-column (mean, variance) input convention as UncertainKernel:

        x[..., 0] = mean of the input location
        x[..., 1] = variance of the input location (0 for an exactly known point)
    """

    def __init__(self, n_samples=32, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.n_samples = n_samples

    @property
    def lengthscale(self):
        return self.base_kernel.lengthscale

    @staticmethod
    def _split(x):
        mean = x[..., 0:1]
        var = x[..., 1:2]
        return mean, var

    def forward(self, x1, x2, diag=False, **params):
        x1_mean, x1_var = self._split(x1)
        x2_mean, x2_var = self._split(x2)

        x1_std = x1_var.clamp_min(0).sqrt()
        x2_std = x2_var.clamp_min(0).sqrt()

        # When x1 and x2 are the same underlying points (e.g. the K(train, train) or K(test, test)
        # block), each Monte Carlo draw must perturb both sides identically. Otherwise the same
        # physical point gets two different noisy realizations depending on whether it's read from the
        # "row" or "column" argument, the per-draw matrix stops being a literal RBFKernel Gram matrix,
        # and averaging matrices that aren't individually PSD does not produce a PSD result -- Cholesky
        # then fails ("Matrix not positive definite") during training.
        same_inputs = x1.shape == x2.shape and torch.equal(x1, x2)

        eps1 = torch.randn(self.n_samples, *x1_mean.shape, dtype=x1_mean.dtype, device=x1_mean.device)
        x1_samples = x1_mean.unsqueeze(0) + eps1 * x1_std.unsqueeze(0)

        if same_inputs:
            x2_samples = x1_samples
        else:
            eps2 = torch.randn(self.n_samples, *x2_mean.shape, dtype=x2_mean.dtype, device=x2_mean.device)
            x2_samples = x2_mean.unsqueeze(0) + eps2 * x2_std.unsqueeze(0)

        # Bypass Kernel.__call__ (which would wrap the result in a LazyEvaluatedKernelTensor) and call
        # RBFKernel's forward directly so we get a plain dense tensor with a leading sample dimension
        # to average over.
        samples = self.base_kernel.forward(x1_samples, x2_samples, diag=diag, **params)
        return samples.mean(dim=0)
