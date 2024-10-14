import torch
import gpytorch

class UncertainKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, **params):

        # preprocessing
        x1_mean = torch.unsqueeze(x1[:,0], 1)
        x1_var = torch.unsqueeze(x1[:,1], 1)

        x2_mean = torch.unsqueeze(x2[:,0], 1)
        x2_var = torch.unsqueeze(x2[:,1], 1)

        if len(x1_mean.shape)>2:
            x1_mean = torch.squeeze(x1_mean)
            x1_var = torch.squeeze(x1_var)

            x2_mean = torch.squeeze(x2_mean)
            x2_var = torch.squeeze(x2_var)

        l2 = self.lengthscale * self.lengthscale


        """
        Numerator Only
        """

        # print(x1_var.shape)
        # print(x2_var.shape)
        # print(self.lengthscale.shape)
        var_dist = self.lengthscale + x1_var + x2_var#self.covar_dist(x1_var, -x2_var, square_dist=False)

        # print(var_dist.shape)

        # temp = self.lengthscale + var_dist

        x1_mean_ = x1_mean.div(var_dist)
        x2_mean_ = x2_mean.div(var_dist)

        temp2 = self.covar_dist(x1_mean_, x2_mean_, square_dist=True)

        dist_mat = temp2.div_(-2).exp_()

        return dist_mat


        """
        Denominator Only
        """

        x1_var_ = x1_var.div(self.lengthscale)
        x2_var_ = x2_var.div(self.lengthscale)

        denomDist = self.covar_dist(x1_var_, x2_var_, square_dist=False)
        denomDist = denomDist.abs().sqrt()


        """
        Combined
        """

        return dist_mat.div(denomDist)