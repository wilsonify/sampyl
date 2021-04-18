r"""
Sampling in parallel

We can make use of our multicore CPUs by running chains in parallel.
To do this, simply request the number of chains you want when you call `sample`: `nuts.sample(1000, n_chains=4)`.
Each chain is given its own process and the OS decides how to run the processes.
Typically this means that each process will run on its own core.
So, if you have four cores and four chains, they will all run in parallel.
But, if you have two cores, only two will run at a time.
"""

import matplotlib.pyplot as plt
import sampyl as smp
from sampyl import np
from sampyl.samplers import NUTS

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


start = smp.find_MAP(logp, {'β': np.ones(3), 'sig': 1.})
nuts = smp.samplers.NUTS.NUTS(logp, start)
chains = nuts.sample(1100, burn=100, n_chains=8)

fig, axes = plt.subplots(figsize=(10, 3), ncols=8)
for ax, chain in zip(axes, chains):
    ax.plot(chain.β, alpha=0.75)
    ax.hlines(np.median(chain.β, axis=0), 0, 1000)

