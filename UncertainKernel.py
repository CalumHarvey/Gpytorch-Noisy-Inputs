import torch
import gpytorch

class UncertaintyCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, var1, var2, lengthscale, dist_func):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("LaplacianCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("LaplacianCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)

        var_dist = dist_func(var1, -var2)
        mean_dist = dist_func(x1, x2)

        l2 = lengthscale * lengthscale

        temp = mean_dist / (l2+var_dist)
        temp2 = 1.0+var_dist/l2

        covar_mat = torch.pow(temp2, -0.5)*torch.exp(-0.5*temp)

        if needs_grad:
            d_output_d_input = mean_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(d_output_d_input)

        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, None, None, lengthscale_grad, None


class UncertainKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag, **params):

        # preprocessing
        x1_mean = torch.unsqueeze(x1[:,0], 1)
        x1_var = torch.unsqueeze(x1[:,1], 1)

        x2_mean = torch.unsqueeze(x2[:,0], 1)
        x2_var = torch.unsqueeze(x2[:,1], 1)

        self.ard_num_dims = len(x1_mean[0]) # always has to have ard_num_dims as specified in paper

        return UncertaintyCovariance.apply(
            x1_mean,
            x2_mean,
            x1_var,
            x2_var,
            self.lengthscale,
            lambda x1_mean, x2_mean: self.covar_dist(
                x1_mean, x2_mean, square_dist=True, diag=diag, postprocess=False, **params
            ),
        )