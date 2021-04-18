r"""
Linear models

When you build larger models,
it would be cumbersome to have to include every parameter as an argument in the logp function.

To avoid this, you can declare the size of variables when passing in the starting state.
For instance, with a linear model it would be great to pass the coefficients as one parameter.
"""

import matplotlib.pyplot as plt
import sampyl as smp
from autograd import grad
from sampyl import np
from sampyl.samplers import NUTS
from sampyl.samplers import metropolis

# First, we'll make some fake data, then infer the coefficients.
N = 200  # Number of data points
sigma = 1  # True parameters
true_B = np.array([2, 1, 4])

# Simulated features, including a constant
X = np.ones((N, len(true_B)))
X[:, 1:] = np.random.rand(N, 2) * 2

# Simulated outcomes with normally distributed noise
y = np.dot(X, true_B) + np.random.randn(N) * sigma

data = np.ones((N, len(true_B) + 1))
data[:, :-1] = X
data[:, -1] = y

fig, axes = plt.subplots(figsize=(7, 4), ncols=2)
for i, ax in enumerate(axes):
    ax.scatter(X[:, i + 1], y)
axes[0].set_ylabel('y')
axes[0].set_xlabel('X1')
axes[1].set_xlabel('X2')

plt.savefig('linear_model_data.png')
plt.close()


def logp(β, sig):
    """
    Here, β is a length 3 array of coefficients
    """
    model = smp.Model()

    # Estimate from our data and coefficients
    y_hat = np.dot(X, β)

    # Add log-likelihood
    model.add(smp.normal(y, mu=y_hat, sig=sig))

    # Add prior for estimate error
    model.add(smp.exponential(sig))

    # Uniform priors on coefficients
    model.add(smp.uniform(β, lower=-100, upper=100))

    return model()


start = smp.find_MAP(
    logp, {'β': np.ones(3), 'sig': 1.},
    bounds={'β': (-5, 10), 'sig': (0.01, None)}
)
sampler = smp.samplers.metropolis.Metropolis(logp, start)
chain = sampler(20000, burn=5000, thin=4)

plt.plot(chain.β, alpha=0.8)
plt.hlines(np.median(chain.β, axis=0), 0, 3500, colors='k')
plt.savefig('beta-chains.png')
plt.close()

plt.hist(chain.β, alpha=0.8, histtype='stepfilled')
plt.vlines(np.median(chain.β, axis=0), 0, 2500, linestyles='--', colors='k')
plt.savefig('beta-hist.png')
plt.close()

# And using NUTS too.
start = smp.find_MAP(logp, {'β': np.ones(3), 'sig': 1.})
nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts.sample(2100, burn=100)

fig, axes = plt.subplots(figsize=(8, 5), nrows=2, ncols=2)
for i, (row, param) in enumerate(zip(axes, [chain.β, chain.sig])):
    row[0].plot(param)
    row[0].set_ylabel('Sample value')
    # row[0].set_xlabel('Sample')
    row[0].set_title(['β', 'sig'][i])
    row[1].set_title(['β', 'sig'][i])
    if len(param.shape) > 1:
        for eachT in param.T:
            row[1].hist(eachT, alpha=0.8, histtype='stepfilled')
        row[1].set_yticklabels('')
        row[1].vlines([2, 1, 4], 0, 600, linestyles='--', colors='k')
    else:
        row[1].hist(param, alpha=0.8, histtype='stepfilled')
        row[1].set_yticklabels('')
        row[1].vlines(np.median(param), 0, 600, linestyles='--', colors='k')
    # row[1].set_xlabel('Sample value')
fig.tight_layout(pad=0.1, h_pad=1.5, w_pad=1)
fig.savefig('linear_model_posterior.png')

grads = [grad(logp, 0), grad(logp, 1)]


def single_logp(theta):
    r"""
    Using one logp function for both logp and gradient
    You can also use one `logp` function that returns both the logp value and the gradient.
    To let the samplers know about this, set `grad_logp = True`.
    I'm also using one argument `theta` as the parameter which contains the five $\beta$ coefficients and $\sigma$.
    """
    b, sig = theta[:3], theta[-1]
    logp_val = logp(b, sig)
    grad_val = np.hstack([each(b, sig) for each in grads])
    return logp_val, grad_val


start = {'theta': np.ones(4)}
nuts = smp.samplers.NUTS.NUTS(single_logp, start, grad_logp=True)
chain = nuts.sample(2000, burn=1000, thin=2)

plt.plot(chain.theta, alpha=0.7)
plt.hlines(np.median(chain.theta, axis=0), 0, 500, colors='k')
plt.savefig('single-logp-chain-NUTS.png')
plt.close()
