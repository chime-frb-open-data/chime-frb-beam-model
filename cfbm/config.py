"""
Configuration for beam_model

End goal is to read eveything in from constants.py, kotekan configuration,
or some other centralized configuration.
"""
from __future__ import print_function
from __future__ import division

import numpy as np
import ephem
from os import path
from datetime import datetime

import chime_frb_constants as constants
from scipy import constants as phys_const

# Some of this needs to be the same as the L0 config:
#    i.e. clamp_pad, clamp_tile, clamp_freq, beam_sep_x
#
# Nx and Ny are the size of the beam grid you have... could be different from
# N feeds.
#
# feed_sep is fixed.

# beamformer config as a function of time, should be able to look this up
# somewhere, but for now hardcoded.
beamformer_configs = [
    (datetime(2018, 1, 1), 4, 256, np.array([-0.1, 0.0, 0.1, 0.2]), 90.0),
    (datetime(2018, 8, 29), 4, 256, np.array([-0.4, 0.0, 0.4, 0.8]), 60.0),
]


def get_L0_config(date=datetime.now()):
    """
    Get beamformer configuration parameters for a date of interest.

    Parameters
    ----------
        date: datetime object
            Date for which to get config. Default is now.

    Returns
    -------
        Nx : int
            Number of columns of beams formed in the EW direction.
        Ny : int
            Number of rows of beams formed in the NS direction.
        ew_beam_spacing : np.array
            Spacing of beams in the EW direction in degrees from meridian.
        northmost_beam : float
            Position of northmost formed beam in NS direction in degrees 
            from zenith.
    """
    latest_config = beamformer_configs[0]
    for beamformer_config in beamformer_configs:
        if date > beamformer_config[0]:
            latest_config = beamformer_config

    return latest_config[1:]


def get_clamping_freq_from_northmost_beam(
    northmost_beam, speed_of_light=phys_const.speed_of_light
):
    """
    Given the position of the northmost beam, calculate the resulting reference
    frequency for clamping of the FFT formed beams.

    Parameters
    ----------
        northmost_beam : float
        Angle from zenith of northmost beam.

    Returns
    -------
        float
        Clamping reference frequency.
    """

    # calculate clamping freq from northmost beam
    # should use northmost_beam as the parameter and calc in model itself?
    freq_ref = (
        speed_of_light
        * (128)
        / (np.sin(northmost_beam * np.pi / 180.0) * constants.DELTA_Y_FEED_M * 256)
    ) / 1.0e6
    return freq_ref


Nx, Ny, ew_beam_spacing, northmost_beam = get_L0_config()
freq_ref = get_clamping_freq_from_northmost_beam(northmost_beam)

data_path = path.abspath(path.join(path.abspath(path.dirname(__file__)), "bm_data/"))

current_config = {
    "Nx": Nx,  # int; number of EW beams
    "Ny": Ny,  # int; number of NS beams
    "ew_spacing": ew_beam_spacing,  # array; EW beam position in degrees
    "ns_feed_sep": constants.DELTA_Y_FEED_M,  # float; Physical feed separation in meters
    "ew_feed_sep": constants.DELTA_X_CYL_M,  # float; Physical feed separation in meters
    "clamp_pad": 2,  # int; Padding factor used in clamping
    "clamp_tile": 2,  # int; Tiling factor used in clamping
    "clamp_freq": freq_ref,  # float; Frequency for reference beams used in clamping
    "speed_of_light_mode": "current",  # str; "current" for c=299792458m/s or "legacy" for c=3e8m/s that matches data before July 10th, 2019.
    "use_1k_freqs_for_clamping": False,  # bool; Whether to use 1k resolution freq array to calculate clamping freqs (matches beamformer clamping exactly). Can only be used with 16k input freqs.
    "datapath_xpol": path.join(data_path, "beam_XX_v1.h5"),
    "datapath_ypol": path.join(data_path, "beam_YY_v1.h5"),
}

# Define which model to load
current_model_name = "CompositeBeamModel"

# set up ephem
chime = ephem.Observer()
chime.lat = np.deg2rad(constants.CHIME_LATITUDE_DEG)
chime.long = np.deg2rad(constants.CHIME_LONGITUDE_DEG)

current_config.update({"location": chime})
