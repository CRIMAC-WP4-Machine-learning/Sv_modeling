import numpy as np
import pytest

from sv_modeling.fembem import func_FEM_BEM
from sv_modeling.geometry import Get_Angle
from sv_modeling.paths import FEMBEM_DIR
from sv_modeling.signals import func_Gen_SweepSig_Explicit
from sv_modeling.spectrum import ESD_BKH


@pytest.mark.parametrize(
    "u, expected",
    [
        ([1.0, 0.0, 0.0], 90.0),   # horizontal fish below transducer: broadside
        ([0.0, 0.0, 1.0], 0.0),    # pointing at transducer: end-on
        ([0.0, 0.0, -1.0], 0.0),   # pointing away: folded to end-on
        ([1.0, 0.0, 1.0], 45.0),
    ],
)
def test_get_angle(u, expected):
    p0 = np.array([0.0, 0.0, 5.0])
    p1 = np.array([0.0, 0.0, 0.0])
    assert Get_Angle(p0, p1, np.array(u)) == pytest.approx(expected)


def test_esd_conserves_energy():
    Fs = 512 * 512 * 16 * 2
    t, p = func_Gen_SweepSig_Explicit(12 * 1.024e-3, 10 * 1.024e-3, 2 * 1.024e-3, Fs, 160e3, 260e3, 25)
    dt = np.mean(np.diff(t))
    out = ESD_BKH(p, dt)
    E_time = np.sum(np.abs(p) ** 2) * dt
    E_freq = np.sum(out[:, 1]) / (len(p) * dt)
    assert E_freq == pytest.approx(E_time, rel=1e-3)


@pytest.mark.skipif(not any(FEMBEM_DIR.glob("*.txt")), reason="FEM/BEM data not present (not stored in git)")
def test_fembem_reader():
    f, ts = func_FEM_BEM(FEMBEM_DIR, "FEM_BEM_II_TwoTargets_TS_a1cm_b4mm_rho7p34_90deg_Delta_1m.txt")
    assert f.shape == ts.shape
    assert f.size > 0
    assert np.all(np.diff(f) > 0)
