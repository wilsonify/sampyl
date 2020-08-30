from sampyl.core import np

import sampyl as smp
import pytest
from sampyl.logps import normal_1D_logp


def test_1d_MAP():
    logp = normal_1D_logp
    start = {"x": 1.0}
    state = smp.find_MAP(logp, start)
