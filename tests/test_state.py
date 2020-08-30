from sampyl.core import np
from sampyl import state


def test_add_states():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state3 + state5
    print(new)
    # assert type(new) == state.State
    # assert np.all(new["x"] == np.array([3, 3, 7]))
    # assert new["y"] == 3


def test_add_list():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state3 + [np.array([2, 3, 4]), 2]
    assert type(new) == state.State
    assert np.all(new["x"] == np.array([3, 5, 7]))
    assert new["y"] == 3


def test_add_multi_array():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state2 + np.array([2, 3, 4])
    assert type(new) == state.State
    assert np.all(new["x"] == np.array([3, 5, 7]))


def test_add_single_array():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state4 + np.array([1.0])
    assert type(new) == state.State
    assert len(new) == 1
    assert new["x"] == np.array([3.0])


def test_add_int():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state1 + 1
    assert type(new) == state.State
    assert new["x"] == 2


def test_add_float():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state1 + 1.0
    assert type(new) == state.State
    assert new["x"] == 2.0


def test_radd_int():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = 1 + state1
    assert type(new) == state.State
    assert new["x"] == 2


def test_radd_float():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = 1.0 + state1
    assert type(new) == state.State
    assert new["x"] == 2.0


def test_mul_int():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state1 * 2
    assert type(new) == state.State
    assert new["x"] == 2


def test_mul_float():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = state1 * 2.0
    assert type(new) == state.State
    assert new["x"] == 2.0


def test_rmul_int():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = 2 * state1
    assert type(new) == state.State
    assert new["x"] == 2


def test_rmul_float():
    state1 = state.State([("x", 1)])
    state2 = state.State([("x", np.array([1, 2, 3]))])
    state3 = state.State([("x", np.array([1, 2, 3])), ("y", 1)])
    state4 = state.State([("x", np.array([2.0]))])
    state5 = state.State([("x", np.array([2, 1, 4])), ("y", 2)])

    new = 2.0 * state1
    assert type(new) == state.State
    assert new["x"] == 2.0
