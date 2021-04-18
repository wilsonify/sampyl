r"""
Sampyl Examples
Here I will have some examples showing how to use Sampyl.
This is for version 0.2.2. Let's import it and get started.
Sampyl is a Python package used to sample from probability distributions using Markov Chain Monte Carlo (MCMC).
This is most useful when sampling from the posterior distribution of a Bayesian model.

Every sampler provided by Sampyl works the same way.
Define $ \log P(\theta) $ as a function, then pass it to the sampler class.
The class returns a sampler object, which you can then use to sample from $P(\theta)$.
For samplers which use the gradient, $\nabla_{\theta} \log P(\theta)$,
Sampyl uses [autograd](https://github.com/HIPS/autograd) to automatically calculate the gradients.
However, you can pass in your own $\nabla_{\theta} \log P(\theta)$ functions.

Starting out simple, let's sample from a normal distribution.
"""

import matplotlib.pyplot as plt
import sampyl as smp
from sampyl import np
from sampyl.samplers import NUTS
from sampyl.samplers import metropolis


def logp(x):
    r"""
    A normal distribution with mean $\mu$ and variance $\sigma^2$ is defined as:
    $$
    P(x,\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \; \mathrm{Exp}\left( \frac{-(x - \mu)^2}{2\sigma^2} \right)
    $$

    For numerical stability, it is typically better to deal with log probabilities, $\log{P(\theta)}$.
    Then for the normal distribution with known mean and variance,
    $$
    \log{P(x \mid \mu, \sigma)} = -\log{\sigma} - \frac{(x - \mu)^2}{2\sigma^2}
    $$

    where we can drop constant terms since the MCMC samplers only require something proportional to $\log{P(\theta)}$.
    We can easily write this as a Python function.
    """
    mu, sig = 3, 2
    return -np.log(sig) - (x - mu) ** 2 / (2 * sig ** 2)


# First we'll use a Metropolis-Hastings sampler.
# Each sampler requires a $\log{P(\theta)}$ function and a starting state.
# We have included a function to calculate the *maximum a posteriori* (MAP) to find the peak of the distribution
# for use as the starting state.
# Then you call the sampler and a chain of samples is returned.

start = smp.find_MAP(logp, {'x': 1.})
metro = smp.samplers.metropolis.Metropolis(logp, start)
chain = metro(10000, burn=2000, thin=4)

# We can retrieve the chain by accessing the attributes defined by the parameter name(s) of `logp`.

plt.plot(chain.x, alpha=0.7)
plt.hlines(np.nanmedian(chain.x), -0, 2001, colors='k')
plt.savefig("normal-chain.png")
plt.close()

_ = plt.hist(chain.x, bins=30, alpha=0.7)
_ = plt.vlines(np.nanmedian(chain.x), 0, 250, linestyles='--')
plt.savefig("normal-hist.png")
plt.close()

# Here we have sampled from a normal distribution with a mean of 3, indicated with the dashed vertical line.
# There is also a No-U-Turn Sampler (NUTS), which avoids the random-walk nature of Metropolis samplers.
# NUTS uses the gradient of $\log{P(\theta)}$ to make intelligent state proposals.
# You'll notice here that we don't pass in any information about the gradient.
# Instead, it is calculated automatically with [autograd](https://github.com/HIPS/autograd).

nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts(2100, burn=100)

plt.plot(chain, alpha=0.7)
plt.hlines(np.nanmedian(chain.x), 0, 2001)
plt.savefig("normal-chain-NUTS.png")
plt.close()

_ = plt.hist(chain.x, bins=30, alpha=0.75)
_ = plt.vlines(np.nanmedian(chain.x), 0, 250, linestyles='--')
plt.savefig("normal-hist-NUTS.png")
plt.close()
