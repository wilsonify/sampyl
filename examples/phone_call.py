"""
Bayesian estimation of phone call rates
this one is a little more complicated.

Let's say you run a business and you put an advertisement in the paper.
Then, to judge the effectiveness of the ad, you want to compare
the number of incoming phone calls per hour before and after the placement of the add.
Then we can build a Bayesian model using a Poisson likelihood with exponential priors for


"""

import matplotlib.pyplot as plt
import sampyl as smp
from sampyl import np
from sampyl.samplers import NUTS
from sampyl.samplers import metropolis

# Fake data for the day before and after placing the ad.
# We'll make the calls increase by 2 an hour. Record data for each
# hour over two work days.
before = np.random.poisson(7, size=16)
after = np.random.poisson(9, size=16)


def logp(λ1, λ2):
    r"""
    Define the log-P function here.

    $\lambda_1$ and $\lambda_2$.

    \begin{align}
    P(\lambda_1, \lambda_2 \mid D) &\propto P( D \mid \lambda_1, \lambda_2)\, P(\lambda_1)\, P(\lambda_2) \\
    P( D \mid \lambda_1, \lambda_2) &\sim \mathrm{Poisson}(D\mid\lambda_1)\,\mathrm{Poisson}(D\mid\lambda_2) \\
    P(\lambda_1) &\sim \mathrm{Exp}(1) \\
    P(\lambda_2) &\sim \mathrm{Exp}(1)
    \end{align}

    This analysis method is known as Bayesian inference or Bayesian estimation.
    We want to know likely values for $\lambda_1$ and $\lambda_2$.
    This information is contained in the posterior distribution $P(\lambda_1, \lambda_2 \mid D)$.
    To infer values for $\lambda_1$ and $\lambda_2$, we can sample from the posterior using our MCMC samplers.
    """
    model = smp.Model()

    model.add(
        smp.poisson(before, rate=λ1),  # Poisson log-likelihood
        smp.poisson(after, rate=λ2)
    )

    model.add(
        smp.exponential(λ1),  # Exponential log-priors for rate parameters
        smp.exponential(λ2)
    )

    return model()


start = smp.find_MAP(logp, {'λ1': 1., 'λ2': 1.})
sampler = smp.samplers.metropolis.Metropolis(logp, start)
chain = sampler(10000, burn=2000, thin=4)

# Sampling returns a numpy record array which you can use to access samples by name.
# Variable names are taken directly from the argument list of `logp`.

print(sampler.var_names)
plt.plot(chain.λ1, alpha=0.7)
plt.hlines(np.median(chain.λ1), 0, 2000)
plt.savefig("lambda-chain.png")
plt.close()

plt.plot(chain.λ2, alpha=0.7)
plt.hlines(np.median(chain.λ2), 0, 2000)
plt.savefig("lambda-hist.png")
plt.close()

# Now to see if there is a significant difference between the two days.
# We can find the difference $\delta =  \lambda_2 - \lambda_1$, then find the probability that $\delta > 0$.

delta = chain.λ2 - chain.λ1

_ = plt.hist(delta, bins=30, alpha=0.75)
_ = plt.vlines(np.median(delta), 0, 250, linestyle='--')
plt.savefig("delta-hist.png")
plt.close()

p = np.mean(delta > 0)
effect = np.mean(delta)
CR = np.percentile(delta, (2.5, 97.5))
print("{:.3f} probability the rate of phone calls increased".format(p))
print("delta = {:.3f}, 95% CR = {{{:.3f} {:.3f}}}".format(effect, *CR))

# There true difference in rates is two per hour, marked with the dashed line.
# Our posterior is showing an effect, but our best estimate is that the rate increased by only one call per hour.
# The 95% credible region is {-0.735 2.743} which idicates that
# there is a 95% probability that the true effect lies with the region, as it indeed does.
# We can also use NUTS to sample from the posterior.

nuts = smp.samplers.NUTS.NUTS(logp, start)
chain = nuts.sample(2100, burn=100)
plt.plot(chain.λ1, alpha=0.7)
plt.hlines(np.median(chain.λ1), 0, 2000)
plt.plot(chain.λ2, alpha=0.7)
plt.hlines(np.median(chain.λ2), 0, 2000)
plt.savefig("chains-NUTS.png")
plt.close()

delta = chain.λ2 - chain.λ1
plt.hist(delta, bins=30, alpha=0.75)
plt.vlines(np.median(delta), 0, 250, linestyle='--')
plt.savefig("delta-hist-NUTS.png")
plt.close()

p = np.mean(delta > 0)
effect = np.mean(delta)
CR = np.percentile(delta, (2.5, 97.5))
print("{:.3f} probability the rate of phone calls increased".format(p))
print("delta = {:.3f}, 95% CR = {{{:.3f} {:.3f}}}".format(effect, *CR))
