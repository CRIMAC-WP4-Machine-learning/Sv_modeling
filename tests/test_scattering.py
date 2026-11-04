import numpy as np
import pandas as pd
import pytest

from sv_modeling.database import Extract_database_metadata
from sv_modeling.paths import TS_DATABASE_150HZ_DIR
from sv_modeling.scattering import SOUND_SPEED, coherent_backscatter, total_ts

pytestmark = pytest.mark.skipif(
    not any(TS_DATABASE_150HZ_DIR.glob("*.csv")),
    reason="TS database not present (not stored in git)",
)

DEPTH = 50.0
HORIZONTAL = [1.0, 0.0, 0.0]  # broadside to a transducer straight above


@pytest.fixture(scope="module")
def df_database():
    return Extract_database_metadata(TS_DATABASE_150HZ_DIR)


@pytest.fixture(scope="module")
def broadside_entry(df_database):
    """freq (Hz) and f_bs of the 90-degree database entry."""
    row = df_database[df_database["IncAngle"] == 90.0].iloc[0]
    df_csv = pd.read_csv(TS_DATABASE_150HZ_DIR / row["filename"])
    return df_csv["Freq_kHz"].to_numpy() * 1000, df_csv["f_bs"].to_numpy().astype(complex)


def run(df_database, points, orientations):
    return coherent_backscatter(points, orientations, df_database, TS_DATABASE_150HZ_DIR)


def test_single_fish_returns_database_fbs(df_database, broadside_entry):
    freq_hz, f_bs = broadside_entry
    res = run(df_database, [[0, 0, -DEPTH]], [HORIZONTAL])

    np.testing.assert_array_equal(res.freq_hz, freq_hz)
    assert res.incident_angles[0] == pytest.approx(90.0)
    np.testing.assert_allclose(np.abs(DEPTH**2 * res.p_far), np.abs(f_bs), rtol=1e-12)
    np.testing.assert_allclose(res.sigma_bs[0], np.abs(f_bs) ** 2, rtol=1e-12)
    np.testing.assert_allclose(total_ts(res.p_far, DEPTH), 20 * np.log10(np.abs(f_bs)), atol=1e-9)


def test_two_fish_at_same_range_add_in_phase(df_database):
    one = run(df_database, [[0, 0, -DEPTH]], [HORIZONTAL])
    # same slant range, mirrored across the beam axis -> identical phase
    x = 0.5
    z = -np.sqrt(DEPTH**2 - x**2)
    two = run(df_database, [[x, 0, z], [-x, 0, z]], [HORIZONTAL, HORIZONTAL])

    # incident angles are equal for both fish, so 2x the single-fish field at that angle
    single_at_angle = run(df_database, [[x, 0, z]], [HORIZONTAL])
    np.testing.assert_allclose(two.p_far, 2 * single_at_angle.p_far, rtol=1e-12)
    assert two.distances == pytest.approx([DEPTH, DEPTH])
    assert one.p_far.shape == two.p_far.shape


def test_four_targets_match_closed_form(df_database, broadside_entry):
    """The COMSOL comparison geometry: four horizontal fish near 50 m."""
    freq_hz, f_bs = broadside_entry
    points = np.array(
        [[0, 0, -DEPTH], [0, 0, -DEPTH - 0.1], [0, 0, -DEPTH - 0.25], [0, -0.25, -DEPTH - 0.25]]
    )
    res = run(df_database, points, np.tile(HORIZONTAL, (4, 1)))

    k = 2 * np.pi * freq_hz / SOUND_SPEED
    R = np.linalg.norm(points, axis=1)
    expected = sum(f_bs / r**2 * np.exp(-2j * k * r) for r in R)

    np.testing.assert_allclose(res.p_far, expected, rtol=1e-12)
    np.testing.assert_allclose(res.distances, R)
    assert res.sigma_bs.shape == (4, len(freq_hz))


def test_sound_speed_changes_phase_only(df_database):
    points = [[0, 0, -DEPTH], [0, 0, -DEPTH - 0.1]]
    orient = [HORIZONTAL, HORIZONTAL]
    a = coherent_backscatter(points, orient, df_database, TS_DATABASE_150HZ_DIR, sound_speed=1500)
    b = coherent_backscatter(points, orient, df_database, TS_DATABASE_150HZ_DIR, sound_speed=1480)
    np.testing.assert_array_equal(a.sigma_bs, b.sigma_bs)
    assert not np.allclose(a.p_far, b.p_far)


def test_shape_mismatch_raises(df_database):
    with pytest.raises(ValueError, match="shape"):
        run(df_database, [[0, 0, -DEPTH], [0, 0, -DEPTH - 1]], [HORIZONTAL])
