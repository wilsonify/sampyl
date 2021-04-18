# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sampyl Examples
#
# Here I will have some examples showing how to use Sampyl. This is for version 0.2.2. Let's import it and get started. Sampyl is a Python package used to sample from probability distributions using Markov Chain Monte Carlo (MCMC). This is most useful when sampling from the posterior distribution of a Bayesian model.
#
# Every sampler provided by Sampyl works the same way. Define $ \log P(\theta) $ as a function, then pass it to the sampler class. The class returns a sampler object, which you can then use to sample from $P(\theta)$. For samplers which use the gradient, $\nabla_{\theta} \log P(\theta)$, Sampyl uses [autograd](https://github.com/HIPS/autograd) to automatically calculate the gradients. However, you can pass in your own $\nabla_{\theta} \log P(\theta)$ functions.
#
# Starting out simple, let's sample from a normal distribution.

# +
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import sampyl as smp
from sampyl import np
from sampyl.samplers import metropolis
from sampyl.samplers import NUTS

# Autograd throws some warnings that are useful, but this is
# a demonstration, so I'll squelch them.
import warnings
warnings.filterwarnings('ignore')
# -

# A normal distribution with mean $\mu$ and variance $\sigma^2$ is defined as:
#
# $$
# P(x,\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \; \mathrm{Exp}\left( \frac{-(x - \mu)^2}{2\sigma^2} \right)
# $$
#
# For numerical stability, it is typically better to deal with log probabilities, $\log{P(\theta)}$. Then for the normal distribution with known mean and variance,
#
# $$
# \log{P(x \mid \mu, \sigma)} = -\log{\sigma} - \frac{(x - \mu)^2}{2\sigma^2} 
# $$
#
# where we can drop constant terms since the MCMC samplers only require something proportional to $\log{P(\theta)}$. We can easily write this as a Python function.

mu, sig = 3, 2
def logp(x):
    return  -np.log(sig) - (x - mu)**2/(2*sig**2)


# First we'll use a Metropolis-Hastings sampler. Each sampler requires a $\log{P(\theta)}$ function and a starting state. We have included a function to calculate the *maximum a posteriori* (MAP) to find the peak of the distribution for use as the starting state. Then you call the sampler and a chain of samples is returned.

start = smp.find_MAP(logp, {'x':1.})
metro = smp.samplers.metropolis.Metropolis(logp, start)
chain = metro(10000, burn=2000, thin=4)

# We can retrieve the chain by accessing the attributes defined by the parameter name(s) of `logp`.

plt.plot(chain.x,alpha=0.7)
plt.hlines(np.nanmedian(chain.x),-0,2001,colors='k')

_ = plt.hist(chain.x, bins=30, alpha=0.7)
_ = plt.vlines(np.nanmedian(chain.x), 0, 250, linestyles='--')

# Here we have sampled from a normal distribution with a mean of 3, indicated with the dashed vertical line.
#
# There is also a No-U-Turn Sampler (NUTS), which avoids the random-walk nature of Metropolis samplers. NUTS uses the gradient of $\log{P(\theta)}$ to make intelligent state proposals. You'll notice here that we don't pass in any information about the gradient. Instead, it is calculated automatically with [autograd](https://github.com/HIPS/autograd).

nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts(2100, burn=100)

plt.plot(chain,alpha=0.7)
plt.hlines(np.nanmedian(chain.x),0,2001)

_ = plt.hist(chain.x, bins=30,alpha=0.75)
_ = plt.vlines(np.nanmedian(chain.x), 0, 250, linestyles='--')

# ## Bayesian estimation of phone call rates
#
# Let's try something a little more complicated. Let's say you run a business and you put an advertisement in the paper. Then, to judge the effectiveness of the ad, you want to compare the number of incoming phone calls per hour before and after the placement of the add. Then we can build a Bayesian model using a Poisson likelihood with exponential priors for $\lambda_1$ and $\lambda_2$.
#
#
# \begin{align}
# P(\lambda_1, \lambda_2 \mid D) &\propto P( D \mid \lambda_1, \lambda_2)\, P(\lambda_1)\, P(\lambda_2) \\
# P( D \mid \lambda_1, \lambda_2) &\sim \mathrm{Poisson}(D\mid\lambda_1)\,\mathrm{Poisson}(D\mid\lambda_2) \\
# P(\lambda_1) &\sim \mathrm{Exp}(1) \\
# P(\lambda_2) &\sim \mathrm{Exp}(1) 
# \end{align}
#
# This analysis method is known as Bayesian inference or Bayesian estimation. We want to know likely values for $\lambda_1$ and $\lambda_2$. This information is contained in the posterior distribution $P(\lambda_1, \lambda_2 \mid D)$. To infer values for $\lambda_1$ and $\lambda_2$, we can sample from the posterior using our MCMC samplers.

# +
# Fake data for the day before and after placing the ad.
# We'll make the calls increase by 2 an hour. Record data for each
# hour over two work days.
before = np.random.poisson(7, size=16)
after = np.random.poisson(9, size=16)

# Define the log-P function here
def logp(λ1, λ2):
    model = smp.Model()
    # Poisson log-likelihoods
    model.add(smp.poisson(before, rate=λ1),
              smp.poisson(after, rate=λ2))

    # Exponential log-priors for rate parameters
    model.add(smp.exponential(λ1),
              smp.exponential(λ2))
    
    return model()


# -

start = smp.find_MAP(logp, {'λ1':1., 'λ2':1.})
sampler = smp.samplers.metropolis.Metropolis(logp, start)
chain = sampler(10000, burn=2000, thin=4)

# Sampling returns a numpy record array which you can use to access samples by name. Variable names are taken directly from the argument list of `logp`.

print(sampler.var_names)

plt.plot(chain.λ1,alpha=0.7)
plt.hlines(np.median(chain.λ1),0,2000)
plt.plot(chain.λ2,alpha=0.7)
plt.hlines(np.median(chain.λ2),0,2000)

# Now to see if there is a significant difference between the two days. We can find the difference $\delta =  \lambda_2 - \lambda_1$, then find the probability that $\delta > 0$.

delta = chain.λ2 - chain.λ1

_ = plt.hist(delta, bins=30,alpha=0.75)
_ = plt.vlines(np.median(delta), 0, 250, linestyle='--')

p = np.mean(delta > 0)
effect = np.mean(delta)
CR = np.percentile(delta, (2.5, 97.5))
print("{:.3f} probability the rate of phone calls increased".format(p))
print("delta = {:.3f}, 95% CR = {{{:.3f} {:.3f}}}".format(effect, *CR))

# There true difference in rates is two per hour, marked with the dashed line. Our posterior is showing an effect, but our best estimate is that the rate increased by only one call per hour. The 95% credible region is {-0.735 2.743} which idicates that there is a 95% probability that the true effect lies with the region, as it indeed does.
#
# We can also use NUTS to sample from the posterior.

nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts.sample(2100, burn=100)
plt.plot(chain.λ1,alpha=0.7)
plt.hlines(np.median(chain.λ1),0,2000)
plt.plot(chain.λ2,alpha=0.7)
plt.hlines(np.median(chain.λ2),0,2000)

delta = chain.λ2 - chain.λ1
plt.hist(delta, bins=30, alpha=0.75)
plt.vlines(np.median(delta), 0, 250, linestyle='--')

p = np.mean(delta > 0)
effect = np.mean(delta)
CR = np.percentile(delta, (2.5, 97.5))
print("{:.3f} probability the rate of phone calls increased".format(p))
print("delta = {:.3f}, 95% CR = {{{:.3f} {:.3f}}}".format(effect, *CR))

# ## Linear models too
#
# When you build larger models, it would be cumbersome to have to include every parameter as an argument in the logp function. To avoid this, you can declare the size of variables when passing in the starting state.
#
# For instance, with a linear model it would be great to pass the coefficients as one parameter. First, we'll make some fake data, then infer the coefficients.

# +
# Number of data points
N = 200
# True parameters
sigma = 1
true_B = np.array([2, 1, 4])

# Simulated features, including a constant
X = np.ones((N, len(true_B)))
X[:,1:] = np.random.rand(N, 2)*2

# Simulated outcomes with normally distributed noise
y = np.dot(X, true_B) + np.random.randn(N)*sigma

data = np.ones((N, len(true_B) + 1))
data[:, :-1] = X
data[:, -1] = y
# -

fig, axes = plt.subplots(figsize=(7,4),ncols=2)
for i, ax in enumerate(axes):
    ax.scatter(X[:,i+1], y)
axes[0].set_ylabel('y')
axes[0].set_xlabel('X1')
axes[1].set_xlabel('X2')

fig.savefig('linear_model_data.png')


# Here, β is a length 3 array of coefficients
def logp(β, sig):
    
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
    bounds={'β':(-5, 10), 'sig':(0.01, None)}
)
sampler = smp.samplers.metropolis.Metropolis(logp, start)
chain = sampler(20000, burn=5000, thin=4)

plt.plot(chain.β,alpha=0.8)
plt.hlines(np.median(chain.β,axis=0),0,3500,colors='k')

plt.hist(chain.β, alpha=0.8, histtype='stepfilled')
plt.vlines(np.median(chain.β,axis=0), 0, 2500, linestyles='--', colors='k')

# And using NUTS too.

start = smp.find_MAP(logp, {'β': np.ones(3), 'sig': 1.})
nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts.sample(2100, burn=100)

fig, axes = plt.subplots(figsize=(8,5), nrows=2, ncols=2)
for i, (row, param) in enumerate(zip(axes, [chain.β, chain.sig])):
    row[0].plot(param)
    row[0].set_ylabel('Sample value')
    #row[0].set_xlabel('Sample')
    row[0].set_title(['β', 'sig'][i])
    row[1].set_title(['β', 'sig'][i])
    if len(param.shape) > 1:
        for each in param.T:
            row[1].hist(each, alpha=0.8, histtype='stepfilled')
        row[1].set_yticklabels('')
        row[1].vlines([2,1,4], 0, 600, linestyles='--', colors='k')        
    else:
        row[1].hist(param, alpha=0.8, histtype='stepfilled')
        row[1].set_yticklabels('')
        row[1].vlines(np.median(param), 0, 600, linestyles='--', colors='k')
    #row[1].set_xlabel('Sample value')
fig.tight_layout(pad=0.1, h_pad=1.5, w_pad=1)

fig.savefig('linear_model_posterior.png')

# ## Using one logp function for both logp and gradient
#
# You can also use one `logp` function that returns both the logp value and the gradient. To let the samplers know about this, set `grad_logp = True`. I'm also using one argument `theta` as the parameter which contains the five $\beta$ coefficients and $\sigma$.

from autograd import grad
grads = [grad(logp, 0), grad(logp, 1)]
def single_logp(theta):
    b, sig = theta[:3], theta[-1]
    logp_val = logp(b, sig)
    grad_val = np.hstack([each(b, sig) for each in grads])
    return logp_val, grad_val


start = {'theta': np.ones(4)}
nuts = smp.samplers.NUTS.NUTS(single_logp, start, grad_logp=True)
chain = nuts.sample(2000, burn=1000, thin=2)

plt.plot(chain.theta,alpha=0.7)
plt.hlines(np.median(chain.theta,axis=0),0,500,colors='k')

# ## Sampling in parallel
#
# We can make use of our multicore CPUs by running chains in parallel. To do this, simply request the number of chains you want when you call `sample`: `nuts.sample(1000, n_chains=4)`. Each chain is given its own process and the OS decides how to run the processes. Typically this means that each process will run on its own core. So, if you have four cores and four chains, they will all run in parallel. But, if you have two cores, only two will run at a time.

start = smp.find_MAP(logp, {'β': np.ones(3), 'sig': 1.})
nuts = smp.samplers.NUTS.NUTS(logp, start)
chains = nuts.sample(1100, burn=100, n_chains=8)

fig, axes = plt.subplots(figsize=(10,3), ncols=8)
for ax, chain in zip(axes, chains):
    ax.plot(chain.β,alpha=0.75)
    ax.hlines(np.median(chain.β,axis=0),0,1000)

# The future
# ----------
#
# * Write a module that makes it easier to build logp functions from distributions
# * Add various functions such as autocorrelation, HPD, etc.
# * Plots!
