import sampyl as smp
from sampyl.core import np, auto_grad_logp

__all__ = [
    "normal_1D_logp",
    "normal_1D_grad_logp",
    "normal_logp",
    "poisson_logp",
    "poisson_with_grad",
    "linear_model_logp",
]

mu, sig = 3, 2


def normal_1D_logp(x):
    return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(sig ** 2)
            - np.sum((x - mu) ** 2) / (2 * sig ** 2)
    )


def normal_1D_grad_logp(x=0.0):
    return -2 * (x - mu) / (2 * sig ** 2)


mu, sig = 10, 3
data = np.random.randn(20) * sig + mu
n = len(data)


def normal_logp(mu, sig):
    likelihood = -n * 0.5 * np.log(sig ** 2) - np.sum((data - mu) ** 2) / (2 * sig ** 2)
    mu_prior = smp.uniform(mu, 5, 15)
    sig_prior = -np.log(np.abs(sig))
    return likelihood + mu_prior + sig_prior


###### Poisson model ######
before = np.random.poisson(7, size=12)
after = np.random.poisson(9, size=12)


def poisson_logp(lam1, lam2):
    # Rates for Poisson must be > 0
    if lam1 <= 0 or lam2 <= 0:
        return -np.inf
    else:
        # logps for likelihoods
        llh1 = np.sum(before * np.log(lam1)) - before.size * lam1
        llh2 = np.sum(after * np.log(lam2)) - after.size * lam2

        # logps for priors
        lam1_prior = -lam1
        lam2_prior = -lam2
        return llh1 + llh2 + lam1_prior + lam2_prior


def poisson_with_grad(lam1, lam2):
    grad_logp = auto_grad_logp(poisson_logp)
    grad = np.array([grad_logp[each](lam1, lam2) for each in ["lam1", "lam2"]])
    return poisson_logp(lam1, lam2), grad


###### Linear model ##########
true_b = np.random.randn(5)
x = np.random.rand(5, 10)
data = np.dot(true_b, x)




def linear_model_logp(b, sig):
    if smp.fails_constraints(sig > 0):
        return -np.inf
    mu = np.dot(b, x)
    n = len(data)
    likelihood = (
            -n * 0.5 * np.log(2 * np.pi)
            - n * 0.5 * np.log(sig ** 2)
            - np.sum((data - mu) ** 2) / (2 * sig ** 2)
    )
    prior_sig = -np.log(np.abs(sig))
    prior_b = smp.uniform(b, lower=-5, upper=10)
    return likelihood + prior_sig + prior_b
