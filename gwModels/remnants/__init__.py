from .HLZ_kick import (bbh_final_kick_precessing_CLZM2007,
                       bbh_final_kick_nonprecessing_HLZ2014)
from .HBR_mass_spin import (bbh_final_mass_precessing_BMR2012,
                            bbh_final_spin_precessing_HBR2016)
from .UIB2016_mass_spin import (bbh_final_mass_non_precessing_UIB2016,
                                bbh_final_spin_non_precessing_UIB2016)
from .IW2025_kick_nonprecessing import gwModel_kick_q200
from .IW2025_kick_precessing import gwModel_kick_prec_flow
from .IW2025_kick_gpr import gwModel_kick_q200_GPR
from .remnant_utils import symmetric_mass_ratio

# Kerr geodesic quantities: ISCO, eccentric separatrix, and the generic
# inclined-eccentric separatrix solver.
from .Kerr import (clip_spin,
                   kerr_isco_radius,
                   kerr_isco_energy,
                   kerr_isco_angular_momentum,
                   kerr_ell,
                   separatrix_energy,
                   separatrix_angular_momentum,
                   separatrix_ell,
                   separatrix_EL,
                   SPIN_CLIP)

# gwModelRem family: unified remnant models for aligned-spin, precessing and
# eccentric binaries, plus the point-particle limit.
from .gwModelRemS import (gwModelRemS,
                          gwModelRemS_mf,
                          gwModelRemS_chif,
                          gwModelRemS_Lpeak,
                          gwModelRemS_omega_peak,
                          gwModelRemS_kick)
from .gwModelRemP import (gwModelRemP,
                          gwModelRemP_mf,
                          gwModelRemP_chif,
                          gwModelRemP_Lpeak,
                          spin_projections)
from .gwModelRemSE import (gwModelRemSE,
                           gwModelRemSE_mf,
                           gwModelRemSE_chif,
                           gwModelRemSE_kick,
                           gwModelRemSE_Lpeak)
from .gwModelRemPE import (gwModelRemPE,
                            gwModelRemPE_mf,
                            gwModelRemPE_chif,
                            gwModelRemPE_Lpeak)
from .gwModelEMRI import gwModelEMRI


def __getattr__(name):
    """
    Lazily expose gwModelRemP_flow so torch and nflows stay optional.

    Importing it eagerly would make `import gwModels` fail for users who have
    not installed the "kicks" extra.
    """
    if name == "gwModelRemP_flow":
        import importlib
        module = importlib.import_module(".gwModelRemP_flow", __name__)
        globals()["gwModelRemP_flow"] = module.gwModelRemP_flow
        return module.gwModelRemP_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
