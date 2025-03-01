import gpytorch, torch

import numpy as np


def _random(low = -2.5, high = 2.5):
    return torch.tensor(np.exp(float(np.random.uniform(low, high, size = 1)[0])))


# Gaussian Process for Regression
class _GPR(gpytorch.models.ExactGP):
    def __init__(self, X_, y_, g_, _like, kernel, degree, RC, hrzn, random_init = True, multiple_length_scales = False):
        super(_GPR, self).__init__(X_, y_, _like)
        self.mean_module = gpytorch.means.ConstantMean()
        # Random Parameters Initialization
        self.random_init            = random_init
        self.multiple_length_scales = multiple_length_scales
        # Define features index
        idx_dim_  = torch.linspace(0, g_.shape[0] - 1, g_.shape[0], dtype = int)
        # Treat features and index independently
        idx_      = idx_dim_[g_ != torch.unique(g_)[-1]]
        idx_bias_ = idx_dim_[g_ == torch.unique(g_)[-1]]

        # Define features kernel
        _K = self.__define_kernel(kernel, degree, idx_dim_ = idx_)

        # Define constant kernel for bias
        _K_bias = self.__define_kernel(kernel   = 'linear',
                                       degree   = 0,
                                       idx_dim_ = idx_bias_)
        # Multiple-kernel learning
        if (RC == 1) and (hrzn > 0):
            idx_rc_ = idx_dim_[g_ == torch.unique(g_)[-2]]
            # Define kernel for recursive predictions
            _K_chain = self.__define_kernel(kernel   = 'linear',
                                            degree   = 0,
                                            idx_dim_ = idx_rc_)
            # Combine features and bias kernels
            self.covar_module = _K + _K_chain + _K_bias
        else:
            # Combine features and bias kernels
            self.covar_module = _K + _K_bias

    # Define a kernel
    def __define_kernel(self, kernel, degree, idx_dim_ = None):
        if self.multiple_length_scales:
            dim = int(idx_dim_.shape[0])
        else:
            dim = None
        # Random Initialization Covariance Matrix
        if self.random_init:
            self.likelihood.noise_covar.raw_noise.data.fill_(self.likelihood.noise_covar.raw_noise_constraint.inverse_transform(_random()))
        if self.random_init:
            self.mean_module.constant.data.fill_(_random())
        # Linear kernel
        if kernel == 'linear':
            _K = gpytorch.kernels.LinearKernel(active_dims = idx_dim_)
            # Linear kernel parameter
            if self.random_init: _K.raw_variance.data.fill_(_K.raw_variance_constraint.inverse_transform(_random()))
            return _K
        # Radian Basis Function Kernel
        if kernel == 'RBF':
            _K = gpytorch.kernels.RBFKernel(active_dims  = idx_dim_,
                                            ard_num_dims = dim)
            # RBF Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Polynomial Expansion Kernel
        if kernel == 'poly':
            _K = gpytorch.kernels.PolynomialKernel(power       = degree,
                                                   active_dims = idx_dim_)
            # Polynomial Kernel parameter
            if self.random_init:
                _K.raw_offset.data.fill_(_K.raw_offset_constraint.inverse_transform(_random()))
        # Matern Kernel
        if kernel == 'matern':
            _K = gpytorch.kernels.MaternKernel(nu           = degree,
                                               active_dims  = idx_dim_,
                                               ard_num_dims = dim)
            # Matern Kernel parameter
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]): _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        # Rational Quadratic Kernel
        if kernel == 'RQ':
            _K = gpytorch.kernels.RQKernel(active_dims  = idx_dim_,
                                           ard_num_dims = dim)
            # RQ Kernel parameters
            if self.random_init:
                _K.raw_alpha.data.fill_(_K.raw_alpha_constraint.inverse_transform(_random()))
            if self.random_init:
                for i in range(_K.raw_lengthscale.shape[1]):
                    _K.raw_lengthscale[0, i].data.fill_(_K.raw_lengthscale_constraint.inverse_transform(_random()))
        _K = gpytorch.kernels.ScaleKernel(_K)
        # Amplitude coefficient parameter
        if self.random_init:
            _K.raw_outputscale.data.fill_(_K.raw_outputscale_constraint.inverse_transform(_random()))
        return _K

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Gaussian Process Regression model fit...
def _GPR_fit(X_, y_, g_, params_, random_init = True):
    # Optimize Kernel hyperparameters
    def __optimize(_model, _like, X_, y_, max_training_iter, early_stop):
        # Storage Variables Initialization
        nmll_ = []
        # Find optimal model hyperparameters
        _model.train()
        # Use the adam optimizer
        _optimizer = torch.optim.Adam(_model.parameters(), lr = .1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_like, _model)
        # Begins Iterative Optimization
        for i in range(max_training_iter):
            # Zero gradients from previous iteration
            _optimizer.zero_grad()
            # Output from model
            f_hat_ = _model(X_)
            # Calc loss and backprop gradients
            _nmll = - _mll(f_hat_, y_)
            _nmll.backward()
            _optimizer.step()
            #print(i, np.around(float(_error.detach().numpy()), 2))
            nmll_.append(np.around(float(_nmll.detach().numpy()), 2) )
            if np.isnan(nmll_[-1]):
                return _model, _like, np.inf
            if i > early_stop:
                if np.unique(nmll_[-early_stop:]).shape[0] == 1:
                    break
        return _model, _like, nmll_[-1]

    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = params_
    # Add dummy feature for the bias
    X_ = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    g_ = np.concatenate((g_, np.ones((1,))*np.unique(g_)[-1] + 1), axis = 0)
    # Numpy yo pyTorch
    X_p_ = torch.tensor(X_, dtype = torch.float)
    y_p_ = torch.tensor(y_, dtype = torch.float)
    g_p_ = torch.tensor(g_, dtype = torch.float)
    # initialize likelihood and model
    _like  = gpytorch.likelihoods.GaussianLikelihood()
    _model = _GPR(X_p_, y_p_, g_p_, _like, kernel, degree, RC, hrzn, random_init)
    return __optimize(_model, _like, X_p_, y_p_, max_training_iter, early_stop)

# Select the best model using multiple initializations
def _fit(X_, y_, g_, param_):
    kernel, degree, RC, hrzn, max_training_iter, n_random_init, early_stop = param_
    # Storage Variables Initialization
    model_ = []
    nmll_  = []
    # No Random Initialization
    _GPR, _like, nmll = _GPR_fit(X_, y_, g_, param_, random_init = False)
    # Get Results
    model_.append([_GPR, _like])
    nmll_.append(nmll)
    # Perform multiple Random Initializations
    for i in range(n_random_init):
        _GPR, _like, nmll = _GPR_fit(X_, y_, g_, param_, random_init = True)
        # Get Results
        model_.append([_GPR, _like])
        nmll_.append(nmll)
    # Best Results of all different Initialization
    _GPR, _like = model_[np.argmin(nmll_)]
    nmll         = nmll_[np.argmin(nmll_)]
    return [_GPR, _like, nmll]

# Calculating prediction for new sample
def _GPR_predict(GP_, X_):
    _model, _like, nmll = GP_
    X_   = np.concatenate((X_, np.ones((X_.shape[0], 1))), axis = 1)
    X_p_ = torch.tensor(X_, dtype = torch.float)
    _model.eval()
    _like.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        _f_hat = _like(_model(X_p_))
        return _f_hat.mean.numpy(), np.sqrt(_f_hat.variance.numpy()), np.sqrt(_like.noise.numpy())


__all__ = ['_GPR_predict',
           '_fit']
