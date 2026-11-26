import textwrap

import pytest

from sv_modeling.config import (
    BeamShellSchool,
    ConfigError,
    FixedSchool,
    SpheroidSchool,
    config_from_cli,
    load_config,
)
from sv_modeling.paths import CONFIG_DIR, REPO_ROOT

CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))


def write(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --- Shipped configs ---------------------------------------------------------

def test_config_dir_is_not_empty():
    assert CONFIG_FILES


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.path == path.resolve()
    assert cfg.description


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_has_a_default_config(script):
    assert (CONFIG_DIR / f"{script.stem}.yaml").is_file()


# --- Parsing -------------------------------------------------------------------

def test_empty_file_gives_empty_config(tmp_path):
    cfg = load_config(write(tmp_path, ""))
    assert cfg.seed is None
    assert cfg.school is None


def test_school_geometry_selects_class(tmp_path):
    cfg = load_config(write(tmp_path, """
        seed: 42
        school:
          geometry: beam_shell
          n_targets: 10
          mean_range_m: 50.0
          thickness_m: 1
          min_dist_m: 0.1
        """))
    assert cfg.seed == 42
    assert isinstance(cfg.school, BeamShellSchool)
    assert cfg.school_as(BeamShellSchool).n_targets == 10
    with pytest.raises(ConfigError, match="school.geometry: spheroid"):
        cfg.school_as(SpheroidSchool)
    with pytest.raises(ConfigError, match="school.geometry: spheroid or fixed"):
        cfg.school_as(SpheroidSchool, FixedSchool)


def test_lists_become_tuples(tmp_path):
    cfg = load_config(write(tmp_path, """
        orientation:
          distribution: uniform
          theta_range_deg: [-1.0, 1.0]
          phi_range_deg: [80.0, 100.0]
        """))
    assert cfg.orientation.theta_range_deg == (-1.0, 1.0)


def test_integers_are_kept_as_integers(tmp_path):
    # keeps results identical to the previously hard-coded int values
    cfg = load_config(write(tmp_path, "analysis:\n  band_hz: [45000, 90000]\n"))
    assert cfg.analysis.band_hz == (45000, 90000)
    assert isinstance(cfg.analysis.band_hz[0], int)


def test_fixed_school(tmp_path):
    cfg = load_config(write(tmp_path, """
        school:
          geometry: fixed
          points_m: [[0.0, 0.0, -50.0], [0.0, 0.0, -50.1]]
          theta_deg: [0.0, 0.0]
          phi_deg: [90.0, 90.0]
        """))
    assert isinstance(cfg.school, FixedSchool)
    assert cfg.school.points_m[1] == (0.0, 0.0, -50.1)


def test_optional_field_default(tmp_path):
    cfg = load_config(write(tmp_path, """
        bubbles:
          gas: CO2
          radius_min_m: 0.0005
          radius_max_m: 0.002
          radius_resolution_m: 0.0001
          sound_speed_m_s: 1500
        """))
    assert cfg.bubbles.transducer_depth_m == 0.0


# --- Validation ------------------------------------------------------------------

@pytest.mark.parametrize("text, match", [
    ("seeed: 1\n", r"unknown key\(s\) \['seeed'\]"),
    ("school:\n  geometry: spheroid\n  n_targets: 1\n  semi_axis_horizontal_m: 2.0\n"
     "  semi_axis_vertical_m: 0.6\n  depth_m: 50\n  min_dsit_m: 0.1\n", "min_dsit_m"),
    ("orientation:\n  distribution: normal\n  theta_range_deg: [0.0, 1.0]\n",
     "orientation.phi_range_deg: missing required key"),
    ("orientation:\n  distribution: gaussian\n  theta_range_deg: [0.0, 1.0]\n  phi_range_deg: [0.0, 1.0]\n",
     "orientation.distribution: must be one of"),
    ("school:\n  geometry: cube\n", "school.geometry: must be one of"),
    ("school:\n  n_targets: 3\n", "school.geometry: missing required key"),
    ("analysis:\n  band_hz: [5e4, 9e4]\n", "YAML read it as text"),
    ("analysis:\n  band_hz: [50000]\n", "expected a list of 2 values"),
    ("analysis:\n  band_hz: [90000, 50000]\n", "lower limit must be below"),
    ("seed: 4.2\n", "seed: expected an integer"),
    ("school:\n  geometry: beam_shell\n  n_targets: true\n  mean_range_m: 50.0\n"
     "  thickness_m: 1\n  min_dist_m: 0.1\n", "school.n_targets: expected an integer"),
    ("school:\n  geometry: fixed\n  points_m: [[0.0, 0.0, -50.0]]\n  theta_deg: [0.0, 0.0]\n"
     "  phi_deg: [90.0]\n", "same length"),
])
def test_invalid_config_is_rejected(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, text))


def test_error_names_the_file(tmp_path):
    path = write(tmp_path, "seeed: 1\n")
    with pytest.raises(ConfigError, match="run.yaml"):
        load_config(path)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_require_checks_dotted_names(tmp_path):
    cfg = load_config(write(tmp_path, "echosounder:\n  position_m: [0.0, 0.0, 0.0]\n"))
    cfg.require("echosounder", "echosounder.position_m")
    with pytest.raises(ConfigError, match="echosounder.beam_half_angle_deg, analysis"):
        cfg.require("echosounder.beam_half_angle_deg", "analysis")


# --- Command line ------------------------------------------------------------------

def test_config_from_cli_uses_config_argument(tmp_path):
    path = write(tmp_path, "description: from the command line\n")
    cfg = config_from_cli("DistributeFish.yaml", argv=["--config", str(path)], copy_to_outputs=False)
    assert cfg.description == "from the command line"


def test_config_from_cli_default_and_unknown_args():
    # unknown arguments (e.g. from Spyder/IPython) are ignored
    cfg = config_from_cli("DistributeFish.yaml", argv=["-f", "kernel.json"], copy_to_outputs=False)
    assert cfg.path == (CONFIG_DIR / "DistributeFish.yaml").resolve()
