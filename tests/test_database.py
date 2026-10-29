import numpy as np
import pytest

from sv_modeling.database import (
    Extract_database_metadata,
    Find_closestfile_in_database,
    func_get_frq_TS_fbs_Dict,
)
from sv_modeling.paths import TS_DATABASE_150HZ_DIR


pytestmark = pytest.mark.skipif(
    not any(TS_DATABASE_150HZ_DIR.glob("*.csv")),
    reason="TS database not present (not stored in git)",
)


@pytest.fixture(scope="module")
def df_database():
    return Extract_database_metadata(TS_DATABASE_150HZ_DIR)


def test_metadata_parsed_from_filenames(df_database):
    assert len(df_database) == 18
    assert set(df_database.columns) == {"filename", "a", "b", "IncAngle", "depth", "length"}
    assert df_database["depth"].unique().tolist() == [50.0]
    assert df_database["IncAngle"].min() == 30.0
    assert df_database["IncAngle"].max() == 90.0


def test_closest_angle_selected(df_database):
    row = Find_closestfile_in_database(df_database, 50, 0.0, 88.6)
    assert row["IncAngle"] == 89.0
    row = Find_closestfile_in_database(df_database, 50, 0.0, 31.0)
    assert row["IncAngle"] == 30.0


def test_fbs_loaded_as_complex(df_database):
    row = Find_closestfile_in_database(df_database, 50, 0.0, 90.0)
    freq, ts, fbs = func_get_frq_TS_fbs_Dict(TS_DATABASE_150HZ_DIR, row, 0.0)
    assert np.iscomplexobj(fbs)
    assert len(freq) == len(ts) == len(fbs)
