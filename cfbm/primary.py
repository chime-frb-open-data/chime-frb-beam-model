"""
Provides a description of the CHIME primary beam.

"""
from __future__ import division

from builtins import range
import numpy as np
import scipy.constants as phys_const
from scipy.interpolate import interp1d, RegularGridInterpolator
import h5py

from . import BeamModel

_SPEED_OF_LIGHT = phys_const.speed_of_light

class PrimaryBeamModel(BeamModel):
    """
    Provides a description of the primary CHIME beam.

    Parameters
    ----------

    config : dict
        A dictionary of any configuration parameters needed in the beam model.
    """

class DataDrivenPrimaryBeamModel(PrimaryBeamModel):
    """
    This class loads and interpolates the data driven primary beam model from 
    CHIME/cosmo curated by Saurabh. It is based on a coupling model applied 
    to holography data.

    Parameters:

    config : dict
        A dictionary containing: 'datapath_xpol', 'datapath_ypol'
            'datapath_xpol' : Path to the .h5 data file for the x polarization
            'datapath_ypol' : Path to the .h5 data file for the y polarization

    """

    def __init__(self, config, interpolate_bad_freq=False):

        super(DataDrivenPrimaryBeamModel, self).__init__(config)

        datafile_xpol = h5py.File(config["datapath_xpol"], "r")
        datafile_ypol = h5py.File(config["datapath_ypol"], "r")

        index_map_xpol = datafile_xpol["index_map"]
        index_map_ypol = datafile_ypol["index_map"]

        assert (
            np.all(index_map_xpol["x"][:] == index_map_ypol["x"][:])
            and np.all(index_map_xpol["y"][:] == index_map_ypol["y"][:])
            and np.all(index_map_xpol["frequency"][:] == index_map_ypol["frequency"][:])
        )

        bad_freq_xpol = self._get_bad_freq_mask(
            datafile_xpol, include_nans=interpolate_bad_freq
        )
        bad_freq_ypol = self._get_bad_freq_mask(
            datafile_ypol, include_nans=interpolate_bad_freq
        )

        good_freq = np.logical_not(np.logical_or(bad_freq_xpol, bad_freq_ypol))

        total_power_beam = (
            datafile_xpol["voltage_beam_amp"][:] ** 2
            + datafile_ypol["voltage_beam_amp"][:] ** 2
        ) / 2.0  # divide by npol so that max power is 1

        # Is linear inpterpolation enough, should we use splines?
        # (Linear would be more well-behaved, splines potentially smoother).
        # Or could get a higher resolution map from Saurabh
        self.beam_map_interpolator = RegularGridInterpolator(
            (
                index_map_xpol["frequency"][good_freq][
                    ::-1
                ],  # reverse so freq increasing
                index_map_xpol["x"][:],
                index_map_xpol["y"][:],
            ),
            total_power_beam[good_freq][::-1],  # reverse so freq increasing
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _get_bad_freq_mask(self, datafile, tolerance=0.05, include_nans=False):
        """
        Return a boolean array of bad frequecies to be masked in the beam
        map.

        The frequencies are "bad" if the response of CygA deviates more than
        a the fractional tolerance from 1.0. 
        """
        cyga_siny = -0.15334373

        cyga_y_ind = np.argmin(np.abs(datafile["index_map"]["y"][:] - cyga_siny))
        meridian_x_ind = np.argmin(np.abs(datafile["index_map"]["x"][:] - 0.0))
        cyga_response = datafile["voltage_beam_amp"][
            :, meridian_x_ind, cyga_y_ind
        ].copy()
        nans = np.isnan(cyga_response)
        if include_nans:
            cyga_response[nans] = 0
        else:
            cyga_response[nans] = 1
        bad_freq = abs(cyga_response - 1) > tolerance

        return bad_freq

    def get_sensitivity(self, positions, freqs):
        """
        Return the sensitivites for a list of positions.

        Parameters
        ----------

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
            Relative sensitivities for each beam and position. The
            sensitivity is 1 at the position of CygA for all frequencies. 
            Has shape (M,K).
        """

        positions = np.atleast_2d(positions)
        x = positions[..., 0]
        y = positions[..., 1]

        x_tel, y_tel = self.get_cartesian_from_position(x, y)

        freqs = np.atleast_1d(freqs)

        grid_points = np.transpose(
            [
                np.tile(freqs, len(positions)),
                np.repeat(x_tel, len(freqs)),
                np.repeat(y_tel, len(freqs)),
            ]
        )

        sens = self.beam_map_interpolator(grid_points)

        return sens.reshape([len(positions), len(freqs)])
