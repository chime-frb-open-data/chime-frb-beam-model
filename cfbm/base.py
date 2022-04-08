"""
Base class for models decsribing CHIME beams.

Example
-------

To load beam model as per the current pipeline confiuration:

```
import cfbm
beammod = cfbm.current_model_class(cfbm.current_config)
```

Then do things with `beammod` to your hearts content.

"""
from __future__ import division

from builtins import object
import numpy as np
from scipy import constants as phys_const
from cfbm import utils
from cfbm.config import current_config

_SPEED_OF_LIGHT = phys_const.speed_of_light


class BeamModel(object):
    """
    Provides a description of the synthesized CHIME beams.

    Parameters
    ----------

    config : dict
        A dictionary of any configuration parameters needed in the beam model.
    """

    def __init__(self, config=None):
        if config is None:
            config = current_config
        self.config = config

    def get_beam_positions(self, beam_ids):
        """
        Given a list of beam_ids return the ``[x, y]`` positions of those beams.

        Parameters
        ----------

        beam_ids : array_like
            An array of beam_ids of length N.

        Returns
        -------

        ndarray
            An array of positions for each beam_id, has shape (N,2)
        """

        raise NotImplementedError(
            "get_beam_positions should be implemented in sub-class"
        )

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
        raise NotImplementedError("get_sensitivity should be implemented in sub-class")

    def get_beam_widths(self, beams, freq):
        """
        For a list of beam IDs return the N/S and E/W beam widths.

        Parameters
        ----------

        beams : array_like
             IDs for the beams for which to calculate sensitivities.
             Has shape (N).
        freqs: float or array_like
             Frequency(ies) in MHz.
             Has shape (K).

        Returns
        -------

        ndarray
            Beam width in ``x'' (E/W) and ``y'' (N/S) for each beam and frequency.
            Has shape (N,K,2).
        """
        raise NotImplementedError("get_beam_widths should be implemented in sub-class")

    def get_equatorial_from_position(self, x, y, time):
        return utils.get_equatorial_from_position(x, y, time)

    def get_position_from_equatorial(self, ra_deg, dec_deg, time):
        return utils.get_position_from_equatorial(ra_deg, dec_deg, time)

    def get_position_from_cartesian(self, x_tel, y_tel):
        return utils.get_position_from_cartesian(x_tel, y_tel)

    def get_cartesian_from_position(self, x, y):
        return utils.get_cartesian_from_position(x, y)

    def is_position_above_horizon(self, x, y):
        return utils.is_position_above_horizon(x, y)

    def is_equatorial_above_horizon(self, ra_deg, dec_deg, time):
        return utils.is_equatorial_above_horizon(ra_deg, dec_deg, time)
