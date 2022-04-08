"""
Provides a description of the synthesized CHIME beams.

"""
from __future__ import division
import numpy as np
from scipy import constants as phys_const
import chime_frb_constants as constants
from . import BeamModel
from . import config as bm_config

_SPEED_OF_LIGHT = phys_const.speed_of_light
_LEGACY_SPEED_OF_LIGHT = 3.0e8
_VARIABLE_SPEED_OF_LIGHT = phys_const.speed_of_light


class FFTFormedBeamModel(BeamModel):
    """
    A parent class for beam models that describe FFT formed beams.
    Takes into account:

    * Spacing in sin(y) as provided by FFT beamforming
    * Elongation of beam as it approaches horizon
    * Frequency dependant beam centers and how they are clamped

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

    """

    def __init__(self, config=None):

        super(FFTFormedBeamModel, self).__init__(config)

        global _VARIABLE_SPEED_OF_LIGHT

        self.ew_spacing = self.config["ew_spacing"]
        self.ns_feed_sep = self.config["ns_feed_sep"]
        self.ew_feed_sep = self.config["ew_feed_sep"]
        self.Nx = self.config["Nx"]
        self.Ny = self.config["Ny"]
        self.clamp_pad = self.config["clamp_pad"]
        self.clamp_tile = self.config["clamp_tile"]
        self.clamping_1k = self.config["use_1k_freqs_for_clamping"]

        if self.config["speed_of_light_mode"] == "current":
            self.clamp_freq = self.config["clamp_freq"]
            _VARIABLE_SPEED_OF_LIGHT = _SPEED_OF_LIGHT
        elif self.config["speed_of_light_mode"] == "legacy":
            self.clamp_freq = bm_config.get_clamping_freq_from_northmost_beam(
                bm_config.northmost_beam, speed_of_light=_LEGACY_SPEED_OF_LIGHT
            )
            _VARIABLE_SPEED_OF_LIGHT = _LEGACY_SPEED_OF_LIGHT
        else:
            raise Exception(
                "config['speed_of_light_mode'] must be either 'current' (c=299792458m/s) or 'legacy' (c=3e8m/s)."
            )

        self.ns_offset = 0
        self.feed_sep = self.ns_feed_sep

        # Hack since beam_sep is not necessarily constant
        # But not sure if `beam_sep_x` is used much anyways...
        self.beam_sep_x = np.mean(self.ew_spacing[1:] - self.ew_spacing[:-1])
        self.beam_sep = self.beam_sep_x

        reference_indices = np.arange(self.Ny) + 1 - self.Ny / 2.0

        self.reference_angles = np.rad2deg(
            np.arcsin(
                _VARIABLE_SPEED_OF_LIGHT
                * reference_indices
                / (self.clamp_freq * 1.0e6 * self.Ny * self.feed_sep)
            )
        )

        self.x_offsets = self.ew_spacing

    def _clamping(self, beam_ids, freqs):
        """
        Return positions of beams for given beam_ids and frequencies taking
        into account clamping.

        Parameters
        ----------

        beam_ids : int or array_like
            Beam id or array of beam ids. Has length N.
        freqs : float or array_like
            Frequency or list of frequencies in MHz. Has length M.

        Returns
        -------

        output_angles : ndarray
            NxM array of y positions of beams in grid for each beam and
            frequency.
        """

        global _VARIABLE_SPEED_OF_LIGHT
        beam_ids = np.array(beam_ids)
        freqs = np.array(freqs)

        y_idx = beam_ids % 1000

        if self.clamping_1k and (len(freqs) == 16 * 1024):
            # Use beamformer resolution (1K) frequencies to calculate clamping
            # Note: right now this only works if the input freqs is 16k
            freqs_1k = np.array(constants.FPGA_FREQ)[::-1]
            freqs_16k = np.repeat(freqs_1k, 16)
        elif self.clamping_1k and (len(freqs) != 16 * 1024):
            raise Exception(
                "config['use_1k_freqs_for_clamping']=True but input freqs array is not 16k resolution."
            )
        else:
            freqs_16k = freqs

        beam_angles_at_ref = self.reference_angles[y_idx]

        padded_indices = (
            np.arange(self.Ny * self.clamp_pad * self.clamp_tile)
            - self.Ny * self.clamp_pad * self.clamp_tile // 2
        )

        t = (
            self.Ny
            * self.clamp_pad
            * (self.clamp_freq * 1.0e6)
            * (
                self.feed_sep
                / _VARIABLE_SPEED_OF_LIGHT
                * np.sin(np.deg2rad(beam_angles_at_ref))
            )
            + 0.5
        )

        delta_t = (
            self.Ny
            * self.clamp_pad
            * self.feed_sep
            / _VARIABLE_SPEED_OF_LIGHT
            * 1e6
            * np.outer(
                freqs_16k - self.clamp_freq, np.sin(np.deg2rad(beam_angles_at_ref))
            )
        )

        beam_indices_in_padded = np.array(
            np.floor(t + delta_t) + self.Ny * self.clamp_tile * self.clamp_pad // 2,
            dtype=np.int,
        ).T

        output_angles = np.rad2deg(
            np.arcsin(
                _SPEED_OF_LIGHT
                * padded_indices[beam_indices_in_padded]
                / (freqs * 1.0e6 * self.Ny * self.clamp_pad * self.feed_sep)
            )
        )

        return output_angles

    def _single_beam_signal(self, x_off, y_off, beam_za, freq):
        """
        Get beam signal at position ``x_off``, ``y_off`` in degrees from beam
        center for a single beam.

        Parameters
        ----------

        x_off : float or array_like
            x offset or array of x offsets from beam center in degrees.
            Must be same shape as ``y_off``.

        y_off : float or array_like
            y offset or array of y offsets from beam center in degrees.
            Must be same shape as ``x_off``.

        beam_za: float or array_like
            Zenith angle for each beam and freq whose signal is being calculated.

        freq : float or array_like
            frequency or array of frequencies to calculate signal at in MHz.

        Returns
        -------

        ndarray
            Relative beam signal (centre of beam is 1).
            Array is the same shape as ``x_off`` and ``y_off`` with an
            axis added for ``freq``, i.e. ``(shape(x_off),len(freq))``

        Notes
        -----

        Can calculate signal for several positions if arrays of x and y
        positions are given as ``x_off`` and ``y_off``.

        Can also calculate for several frequencies if ``freq`` is an array.

        """

        sincsq_halfmax = 0.44295

        beam_fwhm_x = _beam_fwhm(self.Ny, self.feed_sep, freq)
        beam_fwhm_y = _beam_fwhm(self.Ny, self.feed_sep, freq) / np.cos(
            np.deg2rad(beam_za)
        )

        sensitivities = (
            np.sinc(
                sincsq_halfmax
                * np.moveaxis(x_off, 2, 0)
                / (0.5 * np.array(beam_fwhm_x))
            )
            ** 2
            * np.sinc(
                sincsq_halfmax
                * np.moveaxis(y_off, 2, 0)
                / (0.5 * np.array(beam_fwhm_y))
            )
            ** 2
        )

        return sensitivities

    def get_beam_positions(self, beam_ids, freqs=None):
        """
        Given a list of beam_ids return the ``[x, y]`` positions of those beams.

        Parameters
        ----------

        beam_ids : array_like
            An array of beam_ids of length N.
        freqs : array_like
            An array of frequencies at which to calculate beam positions.
            Has length M.

        Returns
        -------

        ndarray
            NxMx2 array of positions for each beam_id, and frequency.


        Note
        ----

        `beam_id`s outside of the beam grid will get position [0,0].
        """

        beam_ids = np.atleast_1d(beam_ids)
        freqs = np.atleast_1d(freqs)

        x_idx = beam_ids // 1000
        beams_in_grid = x_idx < self.Nx

        y_beam_positions = np.zeros([len(beam_ids), len(freqs)])
        x_beam_positions = np.zeros([len(beam_ids)])

        y_beam_positions[beams_in_grid] = self._clamping(beam_ids[beams_in_grid], freqs)

        x_beam_positions[beams_in_grid] = self.x_offsets[x_idx[beams_in_grid]]
        x_beam_positions = np.repeat(x_beam_positions, y_beam_positions.shape[1])
        x_beam_positions.shape = y_beam_positions.shape

        return np.array([x_beam_positions.T, y_beam_positions.T]).T

    def get_beam_widths(self, beams, freq):
        """
        For a list of beam IDs return the N/S and E/W beam widths.

        Parameters
        ----------

        beams : int or array_like
             IDs for the beams for which to calculate sensitivities.
             Has shape (N).
        freq: float or array_like
             Frequency(ies) in MHz.
             Has shape (K).

        Returns
        -------

        ndarray
            Beam width in ``x`` (E/W) and ``y`` (N/S) for each beam and frequency.
            Has shape (N,K,2).
        """
        freq = np.atleast_1d(freq)
        beams = np.atleast_1d(beams)

        offsets = self.get_beam_positions(beams, freq)
        beam_zas = np.abs(offsets[:, :, 1])  # y=0 is zenith

        widths = _beam_fwhm(self.Ny, self.feed_sep, freq)

        x_widths = np.tile(widths, len(beams))
        x_widths.shape = (len(beams), len(freq))

        y_widths = widths / np.cos(np.deg2rad(beam_zas))

        widths = np.array([x_widths.T, y_widths.T])

        return widths.T

class FFTFormedActualBeamModel(FFTFormedBeamModel):
    """
    A beam model with positions and shapes calculated analytically from the
    angular delays. Also takes into account clamping.

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

    """

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

        freqs = np.atleast_1d(freqs)

        x = positions[..., 0]
        y = positions[..., 1]
        x_tel, y_tel = self.get_cartesian_from_position(x, y)

        offsets = self.get_beam_positions(beams, freqs)
        offsets_x_tel, offsets_y_tel = self.get_cartesian_from_position(
            offsets[:, :, 0], offsets[:, :, 1]
        )

        x_sens = _get_beam_shape_analytic(
            x_tel, freqs, offsets_x_tel, self.ew_feed_sep, self.Nx
        )
        y_sens = _get_beam_shape_analytic(
            y_tel, freqs, offsets_y_tel, self.ns_feed_sep, self.Ny
        )
        return x_sens * y_sens


class FFTFormedSincNSBeamModel(FFTFormedBeamModel):
    """
    A beam model with EW beam shapes calculated analytically from the
    angular delays. NS beams are approximated as sinc**2 functions.
    Also takes into account clamping.

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
    """

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

        x = positions[..., 0]
        y = positions[..., 1]
        freqs = np.atleast_1d(freqs)

        offsets = self.get_beam_positions(beams, freqs)
        beam_zas = np.abs(offsets[:, :, 1])  # y=0 is zenith

        x_tel, y_tel = self.get_cartesian_from_position(x, y)

        offsets_x_tel = self.get_cartesian_from_position(
            offsets[:, :, 0], offsets[:, :, 1]
        )[0]

        x_sens = _get_beam_shape_analytic(
            x_tel, freqs, offsets_x_tel, self.ew_feed_sep, self.Nx
        )
        y_sens = _get_sincsq_signal_NS(
            self.Ny,
            self.feed_sep,
            np.array(y)[np.newaxis, ...] - offsets[:, :, 1][..., np.newaxis],
            beam_zas,
            freqs,
        )

        return x_sens * y_sens

def _beam_fwhm(Ny, feed_sep, freq):
    """
    Calculate FWHM of synthesized beam for a given frequnecy

    Parameters
    ----------

    freq : float
        Frequency in MHz

    Returns
    -------

    float
        FWHM in degrees

    Note
    ----

    The FWHM of the beam pattern for an FFT telescope of side D is
    equivalent to a disk with diameter D/0.87.

    See Tegmark & Zaldarriaga (2009) Phys. Rev. D. 79, 083530

    """

    airy_diameter = (Ny - 1) * feed_sep / 0.87

    wavelength = _SPEED_OF_LIGHT / (np.array(freq) * 1.0e6)
    return 180.0 / np.pi * np.arcsin(wavelength / airy_diameter)


def _get_sincsq_signal_NS(Ny, feed_sep, y_off, beam_za, freq):
    """
    Get beam signal in NS dimension, y,  at position ``y_off`` in degrees from beam
    center for a single beam.

    Parameters
    ----------

    y_off : float or array_like
        y offset or array of y offsets from beam center in degrees.
        Must be same shape as ``x_off``.

    beam_za: float or array_like
        Zenith angle for each beam and freq whose signal is being calculated.

    freq : float or array_like
        frequency or array of frequencies to calculate signal at in MHz.

    Returns
    -------

    ndarray
        Relative beam signal (centre of beam is 1).
        Array is the same shape as ``y_off`` with an
        axis added for ``freq``, i.e. ``(shape(y_off),len(freq))``

    Notes
    -----

    Can calculate signal for several positions if arrays of y
    positions are given as ``y_off``.

    Can also calculate for several frequencies if ``freq`` is an array.

    """

    sincsq_halfmax = 0.44295

    beam_fwhm_y = _beam_fwhm(Ny, feed_sep, freq) / np.cos(np.deg2rad(beam_za))

    return (
        np.sinc(
            sincsq_halfmax * np.moveaxis(y_off, 2, 0) / (0.5 * np.array(beam_fwhm_y))
        )
        ** 2
    )


def _get_beam_shape_analytic(x_tel, freq, offset, feed_sep, N):
    """
    Return the relative power beam (i.e. peak = 1.0) given a telescope
    coordinate (in the unit-spehere cartesian system)
    ``x_tel`` and a frequency, ``freq``.

    Parameters
    ----------

    x_tel : float or array_like
        Telescope coordinate or array of coordinates. Has length N.
    freq : array_like
        Array of frequencies in MHz. Has length M.
    offset : array_like
        Array of beam offsets in telescope coordinates per frequency
        in ``freq``. Has shape K*M.

    Returns
    -------

    N*K*M array of beam sensitivities for each ``x_tel``, ``freq``, ``offset``

    """

    # Implement Equation 36 from 1710.08591.
    sep_in_wavelengths = feed_sep * freq * 1e6 / _SPEED_OF_LIGHT

    sin_arg = offset[None, :, :] - x_tel[..., None, None]
    sin_arg = sin_arg * np.pi * sep_in_wavelengths
    sin_fact = np.sin(sin_arg)
    sin_Nfact = np.sin(N * sin_arg)
    # Take care of limit sin_arg -> n * pi
    tol = 1e-6 / N
    take_lim = np.abs(sin_fact) < tol
    sin_fact[take_lim] = 1
    sin_Nfact[take_lim] = N
    return (sin_Nfact / sin_fact) ** 2 / N ** 2
