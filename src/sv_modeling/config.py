"""Run configurations (YAML) for the scripts in ``scripts/``.

Each script reads its parameters from ``configs/<script name>.yaml``, or from
the file given with ``--config``. The file is checked against the dataclasses
below: unknown keys, missing keys and wrong types raise ``ConfigError`` naming
the offending key, so a typo never falls back silently to a default.

Units are part of the key names (``_m``, ``_hz``, ``_deg``, ``_ms``).

Example
-------
>>> from sv_modeling.config import BeamShellSchool, config_from_cli
>>> cfg = config_from_cli("DistributeFish_IdealSpect_sv_beam.yaml")  # doctest: +SKIP
>>> school = cfg.school_as(BeamShellSchool)                          # doctest: +SKIP

YAML note: PyYAML reads ``5e4`` as a string. Write numbers in exponent form
with a decimal point and a signed exponent (``5.0e+4``) or in full (``50000``).
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

import yaml

from sv_modeling.paths import CONFIG_DIR, output_path


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransmitSignal:
    """FM sweep passed to ``func_Gen_SweepSig_Explicit``."""

    effective_duration_ms: float
    center_time_ms: float
    total_duration_ms: float
    sampling_rate_hz: float
    f_start_hz: float
    f_end_hz: float
    ramp_exponent: float


@dataclass(frozen=True)
class Echosounder:
    """Transducer position and beam.

    ``beam_direction`` and ``beam_half_angle_deg`` are optional because not
    every script uses them; scripts that do call ``cfg.require(...)``.
    """

    position_m: tuple[float, float, float]
    beam_direction: tuple[float, float, float] | None = None
    beam_half_angle_deg: float | None = None


@dataclass(frozen=True)
class SpheroidSchool:
    """Targets placed uniformly in a spheroid centred at (0, 0, -depth_m)."""

    geometry: Literal["spheroid"]
    n_targets: int
    semi_axis_horizontal_m: float
    semi_axis_vertical_m: float
    depth_m: float
    min_dist_m: float


@dataclass(frozen=True)
class BeamShellSchool:
    """Targets placed uniformly in the beam between ranges R - DR/2 and R + DR/2.

    The beam half-angle is taken from ``echosounder.beam_half_angle_deg``.
    """

    geometry: Literal["beam_shell"]
    n_targets: int
    mean_range_m: float
    thickness_m: float
    min_dist_m: float


@dataclass(frozen=True)
class FixedSchool:
    """Explicit target positions and orientations (one entry per target)."""

    geometry: Literal["fixed"]
    points_m: list[tuple[float, float, float]]
    theta_deg: list[float]
    phi_deg: list[float]

    def __post_init__(self):
        n = len(self.points_m)
        if n == 0:
            raise ConfigError("points_m: at least one target is required")
        if len(self.theta_deg) != n or len(self.phi_deg) != n:
            raise ConfigError(
                f"points_m, theta_deg and phi_deg must have the same length "
                f"(got {n}, {len(self.theta_deg)}, {len(self.phi_deg)})"
            )


@dataclass(frozen=True)
class Orientation:
    """Fish orientation ranges for ``orient_fish``.

    theta: azimuth in the XY plane from +X. phi: polar angle from +Z.
    """

    distribution: Literal["normal", "uniform"]
    theta_range_deg: tuple[float, float]
    phi_range_deg: tuple[float, float]


@dataclass(frozen=True)
class Analysis:
    """Frequency band used for band averages and density estimates."""

    band_hz: tuple[float, float]

    def __post_init__(self):
        if not self.band_hz[0] < self.band_hz[1]:
            raise ConfigError(f"band_hz: lower limit must be below upper limit, got {list(self.band_hz)}")


@dataclass(frozen=True)
class FemBem:
    """FEM/BEM (COMSOL) result file in ``data/fembem/``."""

    file: str


@dataclass(frozen=True)
class Bubbles:
    """Spherical gas bubbles (shared by the bubble simulation and library generator)."""

    gas: str
    radius_min_m: float
    radius_max_m: float
    radius_resolution_m: float
    sound_speed_m_s: float
    transducer_depth_m: float = 0.0


@dataclass(frozen=True)
class BubbleLibrary:
    """Depth/frequency grid and water properties for the f_bs library."""

    depth_start_m: float
    depth_end_m: float
    depth_resolution_m: float
    freq_start_hz: float
    freq_end_hz: float
    freq_resolution_hz: float
    surface_pressure_pa: float
    water_density_kg_m3: float
    temperature_c: float


School = Union[SpheroidSchool, BeamShellSchool, FixedSchool]


@dataclass(frozen=True)
class RunConfig:
    """A full run configuration. Sections not in the file are ``None``."""

    path: Path
    description: str = ""
    seed: int | None = None
    transmit_signal: TransmitSignal | None = None
    echosounder: Echosounder | None = None
    school: School | None = None
    orientation: Orientation | None = None
    analysis: Analysis | None = None
    fembem: FemBem | None = None
    bubbles: Bubbles | None = None
    bubble_library: BubbleLibrary | None = None

    def require(self, *names: str) -> None:
        """Raise ConfigError unless every (dotted) name is set, e.g. ``"echosounder.beam_half_angle_deg"``."""
        missing = []
        for name in names:
            obj = self
            for part in name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is None:
                missing.append(name)
        if missing:
            raise ConfigError(f"{self.path.name}: this script needs {', '.join(missing)}")

    def school_as(self, cls: type) -> School:
        """Return ``school`` if it has the geometry this script needs, else raise ConfigError."""
        self.require("school")
        if not isinstance(self.school, cls):
            wanted = get_args(get_type_hints(cls)["geometry"])[0]
            raise ConfigError(
                f"{self.path.name}: this script needs school.geometry: {wanted}, "
                f"got {self.school.geometry}"
            )
        return self.school


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _number_error(value, where: str, kind: str) -> ConfigError:
    msg = f"{where}: expected {kind}, got {value!r}"
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            pass
        else:
            msg += " (YAML read it as text; write e.g. 5.0e+4 or 50000 instead of 5e4)"
    return ConfigError(msg)


def _convert(value, tp, where: str):
    origin, args = get_origin(tp), get_args(tp)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _convert(value, options[0], where)
        # Union of dataclasses: choose by the "geometry" key
        by_geometry = {get_args(get_type_hints(o)["geometry"])[0]: o for o in options}
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        if "geometry" not in value:
            raise ConfigError(f"{_join(where, 'geometry')}: missing required key (one of {sorted(by_geometry)})")
        geometry = value["geometry"]
        if geometry not in by_geometry:
            raise ConfigError(
                f"{_join(where, 'geometry')}: must be one of {sorted(by_geometry)}, got {geometry!r}"
            )
        return _build(by_geometry[geometry], value, where)

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: must be one of {list(args)}, got {value!r}")
        return value

    if tp is float:
        # ints are kept as ints so results are identical to the old hard-coded values
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _number_error(value, where, "a number")
        return value

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _number_error(value, where, "an integer")
        return value

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value

    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{where}: expected a list of {len(args)} values, got {value!r}")
        return tuple(_convert(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))

    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_convert(v, args[0], f"{where}[{i}]") for i, v in enumerate(value)]

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)

    raise TypeError(f"Unsupported config type {tp!r} at {where}")


def _build(cls, data, where: str, **extra):
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {data!r}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.name not in extra}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(
            f"{where or 'top level'}: unknown key(s) {unknown}; allowed: {sorted(fields)}"
        )

    kwargs = {}
    for name, f in fields.items():
        key = _join(where, name)
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{key}: missing required key")

    try:
        return cls(**kwargs, **extra)
    except ConfigError as exc:
        raise ConfigError(f"{where}: {exc}" if where else str(exc)) from None


def load_config(path) -> RunConfig:
    """Load and validate a YAML run configuration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    try:
        return _build(RunConfig, data, "", path=path.resolve())
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def config_from_cli(default_name: str, description: str | None = None,
                    argv: list[str] | None = None, copy_to_outputs: bool = True) -> RunConfig:
    """Load the config given with ``--config``, or ``configs/<default_name>``.

    Unknown command-line arguments are ignored, so scripts also run inside
    Spyder/IPython. With ``copy_to_outputs`` the config file is copied to
    ``outputs/used_<name>.yaml`` so the figures can be traced to their settings.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR / default_name,
        help=f"YAML run configuration (default: configs/{default_name})",
    )
    args, _unknown = parser.parse_known_args(argv)

    cfg = load_config(args.config)
    print(f"Config: {cfg.path}")
    if cfg.description:
        print(f"        {cfg.description}")

    if copy_to_outputs:
        target = output_path(f"used_{cfg.path.name}")
        if target.resolve() != cfg.path:
            shutil.copyfile(cfg.path, target)
    return cfg
