"""
Useful functions for calculating things that are independent of what beam
model is being used.
"""

import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation
import ephem

import chime_frb_constants as constants
from cfbm import config


def get_great_circle_distance(long1, long2, lat1, lat2):
    """
    For a longitude (e.g. RA) and latitude (e.g. dec) in a spherical coordinate
    system, calculate the great circle distance between two points in degrees.
    """

    long1_rad = np.deg2rad(long1)
    long2_rad = np.deg2rad(long2)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)

    return np.rad2deg(
        np.arccos(
            np.sin(lat1_rad) * np.sin(lat2_rad)
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(long1_rad - long2_rad)
        )
    )


def get_equatorial_from_position(
    x, y, time, telescope_rotation_angle=constants.TELESCOPE_ROTATION_ANGLE
):
    """
    For a position x, y in the beam grid return the RA and Dec

    Parameters
    ----------

    x : float or iterable
         ``x`` position in coordinate grid.
    y : float or iterable
         ``y`` position in coordinate grid.
    time : float or datetime object or astropy.Time object
         Time at observatory to perform transform. If float, it is considered unix time
    telescope_rotation_angle : float
         Rotation of the coordinate system with respect the telescope axis (deg). Default: value calculated by cosmology team

    Returns
    -------

    ra : float or ndarray
         Right ascension in celestial coordinates.
    dec : float or ndarray
         Declination in celestial coordinates.

    Notes
    -----

    In this base class these translations assume the beam grid laid out in
    the beam model sphinx doc. If a beam model uses a different convention
    this function will need to be overridden.
    """

    chime = config.chime
    lat = chime.lat  # radians

    # use frame where poles are on horizon to avoid singularity at zenith
    phi = np.pi / 2.0 - np.deg2rad(y)  # polar angle
    theta = np.deg2rad(x) / np.sin(phi)  # azimuthal angle

    if telescope_rotation_angle is not None:
        # correct for telescope rotation
        # note: frame has 'z' axis at North horizon, and 'x' axis at zenith
        rot = Rotation.from_euler("x", -telescope_rotation_angle, degrees=True)
        phi, theta = _cart2sph(*rot.apply(np.array(_sph2cart(phi, theta)).T).T)

    dec = np.arcsin(
        np.cos(lat) * np.cos(phi) + np.sin(lat) * np.sin(phi) * np.cos(theta)
    )

    cosh = (
        -1.0 * (np.cos(phi) - np.cos(lat) * np.sin(dec)) / (np.sin(lat) * np.cos(dec))
    )
    sinh = np.sin(phi) * np.sin(theta) / np.cos(dec)

    hour_angle = np.arctan2(sinh, cosh)

    if isinstance(time, float):
        time = datetime.utcfromtimestamp(time)
    chime.date = time

    ra_deg = np.rad2deg(chime.sidereal_time() - hour_angle) % 360
    dec_deg = np.rad2deg(dec)

    # precession correction
    try:
        coord_ephem = [
            ephem.Equatorial(
                ephem.Equatorial(np.deg2rad(ra_i), np.deg2rad(dec_i), epoch=time),
                epoch=ephem.J2000,
            )
            for ra_i, dec_i in zip(ra_deg, dec_deg)
        ]
        ra_rad, dec_rad = zip(*[(n.ra, n.dec) for n in coord_ephem])
    except TypeError:
        coord_ephem = ephem.Equatorial(
            ephem.Equatorial(np.deg2rad(ra_deg), np.deg2rad(dec_deg), epoch=time),
            epoch=ephem.J2000,
        )
        ra_rad = coord_ephem.ra
        dec_rad = coord_ephem.dec

    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)
    return ra_deg, dec_deg


def get_position_from_equatorial(
    ra_deg, dec_deg, time, telescope_rotation_angle=constants.TELESCOPE_ROTATION_ANGLE
):
    """
    For RA and Dec coordinates return position an x, y position in the beam grid Dec

    Parameters
    ----------

    ra_deg : float or iterable
         Right ascension in celestial coordinates in degrees.
    dec_deg : float or iterable
         Declination in celestial coordinates in degrees.
    time : float or datetime object or astropy.Time object
         Time at observatory to perform transform. If float, it is considered unix time
    telescope_rotation_angle : float
         Rotation of the coordinate system with respect the telescope axis (deg). Default: value calculated by cosmology team

    Returns
    -------

    x : float or ndarray
         ``x`` position in coordinate grid.
    y : float or ndarray
         ``y`` position in coordinate grid.
    Notes
    -----

    In this base class these translations assume the beam grid laid out in
    the beam model sphinx doc. If a beam model uses a different convention
    this function will need to be overridden.
    """

    chime = config.chime
    lat = chime.lat  # radians

    if isinstance(time, float):
        time = datetime.utcfromtimestamp(time)
    chime.date = time

    # precession correction
    try:
        coord_ephem = [
            ephem.Equatorial(
                ephem.Equatorial(
                    np.deg2rad(ra_i), np.deg2rad(dec_i), epoch=ephem.J2000
                ),
                epoch=time,
            )
            for ra_i, dec_i in zip(ra_deg, dec_deg)
        ]
        ra_rad, dec_rad = zip(*[(n.ra, n.dec) for n in coord_ephem])
    except TypeError:
        coord_ephem = ephem.Equatorial(
            ephem.Equatorial(
                np.deg2rad(ra_deg), np.deg2rad(dec_deg), epoch=ephem.J2000
            ),
            epoch=time,
        )
        ra_rad = coord_ephem.ra
        dec_rad = coord_ephem.dec
    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)

    hour_angle = (chime.sidereal_time() - np.deg2rad(ra_deg)) % (np.pi * 2.0)
    dec = np.deg2rad(dec_deg)

    phi = np.arccos(
        np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(hour_angle)
    )

    sintheta = np.cos(dec) * np.sin(hour_angle) / np.sin(phi)
    costheta = (np.sin(dec) - np.cos(lat) * np.cos(phi)) / (np.sin(lat) * np.sin(phi))

    theta = np.arctan2(sintheta, costheta)

    if telescope_rotation_angle is not None:
        # correct for telescope rotation
        # note: frame has 'z' axis at North horizon, and 'x' axis at zenith
        rot = Rotation.from_euler("x", telescope_rotation_angle, degrees=True)
        phi, theta = _cart2sph(*rot.apply(np.array(_sph2cart(phi, theta)).T).T)

    x_deg = np.rad2deg(theta * np.sin(phi))
    y_deg = np.rad2deg(np.pi / 2.0 - phi)
    return x_deg, y_deg


def get_position_from_cartesian(x_tel, y_tel):
    """
    Get beam model (x,y) position from telescope
    cartesian unit-sphere coordinates (x_tel,y_tel).

    Parameters
    ----------

    x_tel : float
         Telescope x coordinate on unit sphere.
    y_tel : float
         Telescope y coordinate on unit sphere.

    Returns
    -------

    x : float
         ``x`` position in coordinate grid.
    y : float
         ``y`` position in coordinate grid.

    """

    z_tel = np.sqrt(1 - x_tel ** 2 - y_tel ** 2)

    phi, theta = _cart2sph(z_tel, x_tel, y_tel)

    x_rad = -theta * np.sin(phi)
    y_rad = np.pi / 2 - phi

    return np.rad2deg(x_rad), np.rad2deg(y_rad)


def get_cartesian_from_position(x, y):
    """
    Get telescope cartesian unit-sphere coordinates 
    (x_tel,y_tel) from beam model (x,y) position.

    Parameters
    ----------

    x : float
         ``x`` position in coordinate grid.
    y : float
         ``y`` position in coordinate grid.

    Returns
    -------

    x_tel : float
         Telescope x coordinate on unit sphere.
    y_tel : float
         Telescope y coordinate on unit sphere.

    Note
    ----

    Returns np.nan for (x,y) positions below horizon.

    """

    phi = np.pi / 2.0 - np.deg2rad(y)
    theta = np.deg2rad(x) / np.sin(phi)

    z_tel, x_tel, y_tel = _sph2cart(phi, -theta)

    x_tel = np.atleast_1d(x_tel)
    y_tel = np.atleast_1d(y_tel)

    below_horizon = np.logical_not(np.atleast_1d(is_position_above_horizon(x, y)))

    x_tel[below_horizon] = np.nan
    y_tel[below_horizon] = np.nan

    if type(x) is np.ndarray:
        return x_tel, y_tel
    else:
        return x_tel[0], y_tel[0]


def is_position_above_horizon(x, y):
    """
    Test whether `(x,y)` position is above the horizon at the telescope.

    Parameters
    ----------

    x : float
         ``x`` position in coordinate grid.
    y : float
         ``y`` position in coordinate grid.

    Returns
    -------

    bool
        Whether position is above horizon or not.
    """
    phi = np.pi / 2.0 - np.deg2rad(y)  # polar angle
    theta = np.deg2rad(x) / np.sin(phi)  # azimuthal angle

    return np.logical_and(theta > -1.0 * np.pi / 2.0, theta < np.pi / 2.0)


def is_equatorial_above_horizon(ra_deg, dec_deg, time):
    """
    Test whether `(ra,dec)` equatorial position is above the horizon at 
    the telescope.

    Parameters
    ----------

    ra_deg : float
         Right ascension in celestial coordinates in degrees.
    dec_deg : float
         Declination in celestial coordinates in degrees.
    time : datetime object or astropy.Time object
         Time at observatory to perform transform.

    Returns
    -------

    bool
        Whether position is above horizon or not.
    """
    x, y = get_position_from_equatorial(ra_deg, dec_deg, time)
    return is_position_above_horizon(x, y)


def _sph2cart(phi, theta):
    """
    Converts spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    phi: float
        polar angle in [0, pi)
    theta: float
        azimuthal angle in [0, 2*pi)

    Returns
    -------
    x, y, z: float
        Cartesian coordinates

    """
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z


def _cart2sph(x, y, z):
    """
    Converts cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z: float
        coordinates on unit sphere

    Returns
    -------
    phi, theta: float
        polar and azimuthal angles

    """
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return phi, theta
