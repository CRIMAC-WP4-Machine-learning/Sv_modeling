"""Simulation of acoustic backscatter from fish aggregations and s_v estimation."""

from sv_modeling.database import (
    Extract_database_metadata,
    Find_closestfile_in_database,
    func_get_frq_TS_fbs_Dict,
)
from sv_modeling.fembem import func_FEM_BEM
from sv_modeling.geometry import Get_Angle
from sv_modeling.signals import func_Gen_SweepSig_Explicit
from sv_modeling.spectrum import ESD_BKH

__version__ = "0.1.0"

__all__ = [
    "ESD_BKH",
    "Extract_database_metadata",
    "Find_closestfile_in_database",
    "Get_Angle",
    "func_FEM_BEM",
    "func_Gen_SweepSig_Explicit",
    "func_get_frq_TS_fbs_Dict",
]
