"""Synthesizers module."""

from ctgan.synthesizers.ctgan import CTGAN
from ctgan.synthesizers.tvae import TVAE
from ctgan.synthesizers.maectgan import MAECTGAN
from ctgan.synthesizers.reducectgan import REDUCEMAECTGAN
from ctgan.synthesizers.rectgan import RECTGAN
from ctgan.synthesizers.outlinehaectgan import OUTLINEHAECTGAN
from ctgan.synthesizers.cleanmaectgan import CleanMAECTGAN
from ctgan.synthesizers.maeafterctgan import MAEAfterCTGAN
from ctgan.synthesizers.m01aectgan import M01AECTGAN
from ctgan.synthesizers.m01ae_2phase_ctgan import M01AE2PhaseCTGAN
from ctgan.synthesizers.m01ae_org_ctgan import M01AEOrgCTGAN
from ctgan.synthesizers.m01ae_2phase_org_ctgan import M01AE2PhaseOrgCTGAN

__all__ = ('CTGAN', 'TVAE', 'MAECTGAN', 'REDUCEMAECTGAN', 'RECTGAN', 'OUTLINEHAECTGAN', 'CleanMAECTGAN', 'MAEAfterCTGAN', 'M01AECTGAN', 'M01AE2PhaseCTGAN', 'M01AEOrgCTGAN', 'M01AE2PhaseOrgCTGAN')



def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
