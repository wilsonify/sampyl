import pytest
from sampyl.exceptions import AutogradError
from sampyl.logps import poisson_with_grad, linear_model_logp, poisson_logp, normal_1D_logp, normal_1D_grad_logp

from sampyl.core import np
from sampyl import samplers as smp
from sampyl.samplers.NUTS import NUTS
from sampyl.samplers.chain import Chain
from sampyl.samplers.hamiltonian import Hamiltonian
from sampyl.samplers.metropolis import Metropolis
from sampyl.samplers.slice import Slice

n_samples = 100


def test_np_source():
    np_source = np.__package__
    assert np_source == "autograd.numpy"


def test_logp_with_grad():
    logp = poisson_with_grad
    start = {"lam1": 1.0, "lam2": 1.0}
    nuts = NUTS(logp, start, grad_logp=True)
    chain = nuts.sample(n_samples)

    assert len(chain) == n_samples


def test_parallel_lin_model():
    logp = linear_model_logp
    start = {"b": np.zeros(5), "sig": 1.0}
    metro = Metropolis(logp, start)
    nuts = NUTS(logp, start)

    metro_chains = metro.sample(n_samples, n_chains=2)
    nuts_chains = nuts.sample(n_samples, n_chains=2)

    assert len(metro_chains) == 2
    assert len(nuts_chains) == 2


def test_parallel_2D():
    start = {"lam1": 1.0, "lam2": 1.0}
    metro = Metropolis(poisson_logp, start)
    nuts = NUTS(poisson_logp, start)

    metro_chains = metro.sample(n_samples, n_chains=2)
    nuts_chains = nuts.sample(n_samples, n_chains=2)

    assert len(metro_chains) == 2
    assert len(nuts_chains) == 2


def test_sample_chain():
    start = {"lam1": 1.0, "lam2": 1.0}
    step1 = Metropolis(poisson_logp, start, condition=["lam2"])
    step2 = NUTS(poisson_logp, start, condition=["lam1"])

    chain = Chain([step1, step2], start)
    trace = chain.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_conditional_chain():
    logp = poisson_logp
    start = {"lam1": 1.0, "lam2": 2.0}
    metro = Metropolis(logp, start, condition=["lam2"])
    nuts = NUTS(logp, start, condition=["lam1"])

    state = metro._conditional_step()
    assert state["lam2"] == 2.0
    nuts.state.update(state)
    state = nuts._conditional_step()
    assert len(state) == 2


def test_conditional():
    logp = poisson_logp
    start = {"lam1": 1.0, "lam2": 2.0}
    metro = Metropolis(logp, start, condition=["lam2"])
    state = metro._conditional_step()
    assert len(state) == 2
    assert state["lam2"] == 2.0


def test_metropolis_linear_model():
    logp = linear_model_logp
    start = {"b": np.zeros(5), "sig": 1.0}
    metro = Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_hamiltonian_linear_model():
    logp = linear_model_logp
    start = {"b": np.zeros(5), "sig": 1.0}
    hmc = Hamiltonian(logp, start)
    trace = hmc.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_nuts_linear_model():
    logp = linear_model_logp
    start = {"b": np.zeros(5), "sig": 1.0}
    nuts = NUTS(logp, start)
    trace = nuts.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_metropolis():
    logp = normal_1D_logp
    start = {"x": 1.0}
    metro = Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert trace.shape == (n_samples,)


@pytest.mark.parametrize(
    "np_source",
    ("autograd.numpy", "numpy")
)
def test_hmc_autograd(np_source):
    logp = normal_1D_logp
    start = {"x": 1.0}
    hmc = Hamiltonian(logp, start)
    trace = hmc.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_hmc_pass_grad_logp():
    start = {"x": 1.0}
    hmc = Hamiltonian(normal_1D_logp, start, grad_logp=normal_1D_grad_logp)
    trace = hmc.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_NUTS_autograd():
    logp = normal_1D_logp
    start = {"x": 1.0}

    nuts = NUTS(logp, start)
    trace = nuts.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_NUTS_pass_grad_logp():
    logp, grad_logp = normal_1D_logp, normal_1D_grad_logp
    start = {"x": 1.0}
    nuts = NUTS(logp, start, grad_logp=grad_logp)
    trace = nuts.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_sampler_num_logp():
    logp = 1.0
    start = {"x": None}
    with pytest.raises(TypeError):
        metro = Metropolis(logp, start)


def test_sampler_no_args_logp():
    def logp():
        return x

    start = {"x": None}
    with pytest.raises(ValueError):
        metro = Metropolis(logp, start)


def test_metropolis_two_vars():
    logp = poisson_logp
    start = {"lam1": 1.0, "lam2": 1.0}
    metro = Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_metropolis_two_vars_start():
    logp = poisson_logp
    start = {"lam1": 1.0, "lam2": 1.0}
    metro = Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_slice():
    logp = normal_1D_logp
    start = {"x": 1.0}
    _slice = Slice(logp, start)
    trace = _slice.sample(n_samples)
    assert trace.shape == (n_samples,)


def test_slice_two_vars():
    logp = poisson_logp
    start = {"lam1": 1.0, "lam2": 1.0}
    _slice = Slice(logp, start)
    trace = _slice.sample(n_samples)
    assert trace.shape == (n_samples,)
