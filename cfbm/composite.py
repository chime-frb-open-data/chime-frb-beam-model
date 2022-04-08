from .formed import (
    FFTFormedSincNSBeamModel,
)
from .primary import (
    DataDrivenPrimaryBeamModel,
)

import os.path
import numpy as np


class _BaseCompositeMixin:
    """Base class for Mixins that add a parimary beam model to a formed beam model."""

    def get_sensitivity(self, beams, positions, freqs):
        """
        For a list of beam IDs return the sensitivites for a list of positions.

        Parameters
        ----------

        beams : array_like
             IDs for the beams for which to calculate sensitivities.
             Has shape (N).
        positions : array_like
             Positions in ``x`` and ``y`` coordinates in degrees.
             Has shape (M,2).
             See beam model doc for coordinate description.
        freqs: array_like
             Frequencies in MHz.
             Has shape (K).

        Returns
        -------

        ndarray
            Relative sensitivities for each beam and position. The on-axis
            sensitivity is 1. Has shape (M,N,K).
        """

        formed_sensitivity = super().get_sensitivity(beams, positions, freqs)

        primary_sensitivity = self.primary_bm.get_sensitivity(positions, freqs)

        return np.moveaxis(
            primary_sensitivity[..., np.newaxis]
            * np.moveaxis(formed_sensitivity, 1, 2),
            2,
            1,
        )

class _DataDrivenPrimaryMixin(_BaseCompositeMixin):
    def __init__(self, config=None, interpolate_bad_freq=False):

        super().__init__(config)

        primary_params = ["datapath_xpol", "datapath_ypol"]
        primary_config = {k: v for k, v in self.config.items() if k in primary_params}
        self.primary_bm = DataDrivenPrimaryBeamModel(
            primary_config, interpolate_bad_freq
        )


class CompositeBeamModel(_DataDrivenPrimaryMixin, FFTFormedSincNSBeamModel):
    """
    Beam model that combines reponse from synthesized FFT beams and model
    of primary beam. The models are chosen to be fast to run.

    Uses `formed.FFTFormedSincNSBeamModel` for the formed beam and 
    `primary.DataDrivenPrimaryBeamModel` for the primary beam model.

    Will replace CompositeBeamModel in the future.

    Parameters
    ----------

    config : dict
        A dictionary containing:

        'beam_sep_x' : separation in degress of E-W beams.

        'feed_sep' : physical N-S separation of feeds in meters.

        'Nx' : number of E-W beam columns.

        'Ny' : number of N-S beam rows.

        'clamp_pad' : Factor that the FFT is padded in beam forming.

        'clamp_tile' : Factor that the FFT is extended beyond Nyquist freq.

        'clamp_freq' : Frequency at which reference angles to clamp to are determined.

        'datapath_xpol' : Path to the .h5 data file for the x polarization

        'datapath_ypol' : Path to the .h5 data file for the y polarization
    """
