import matplotlib
from sampyl import np
from sampyl.diagnostics import diagnostics
from sampyl.samplers.NUTS import NUTS
from sampyl.samplers.metropolis import Metropolis

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# correlated gaussian
def logp(x, y):
    icov = np.linalg.inv(np.array([[1., .8], [.8, 1.]]))
    d = np.array([x, y])
    return -.5 * np.dot(np.dot(d, icov), d)


# logp_xy = lambda(th): logp(th[0], th[1])

start = {'x': 1., 'y': 1.}
# compare the performance of NUTS and Metropolis by effective sample size
nuts = NUTS(logp, start)
nuts_trace = nuts.sample(1000)

met = Metropolis(logp, start)
met_trace = met.sample(1000)

# compute effective sample size based on autocorrelation
nuts_eff = diagnostics.compute_n_eff_acf(nuts_trace.x)
met_eff = diagnostics.compute_n_eff_acf(met_trace.x)
print("NUTS effective sample size: {:0.2f}".format(nuts_eff))
print("MH   effective sample size: {:0.2f}".format(met_eff))

# graphically compare samples
fig, axarr = plt.subplots(1, 2)
axarr[0].scatter(nuts_trace.x, nuts_trace.y)
axarr[0].set_title("NUTS samples")
axarr[1].scatter(met_trace.x, met_trace.y)
axarr[1].set_title("MH samples")
plt.show()
