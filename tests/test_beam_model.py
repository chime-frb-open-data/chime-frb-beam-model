import pytest
import numpy as np
from numpy import random
from datetime import datetime as dt
import pytz

import cfbm
import chime_frb_constants as constants


@pytest.fixture(
    params=[
        "FFTFormedActualBeamModel",
        "FFTFormedSincNSBeamModel",
        "CompositeBeamModel",
    ]
)
def bm(request):
    return getattr(cfbm, request.param)()


@pytest.fixture(
    params=[
        "FFTFormedActualBeamModel",
        "FFTFormedSincNSBeamModel",
    ]
)
def formed_bm(request):
    return getattr(cfbm, request.param)()


@pytest.fixture
def base_bm():
    config = cfbm.current_config
    return cfbm.BeamModel(config)


@pytest.fixture
def basic_data():

    beam_ids = np.array([1127, 1128])
    freqs = np.array([400.0, 600.0, 800.0])
    positions = np.array([[0.0, 0.0], [-1.0, -1.0]])

    return [beam_ids, freqs, positions]


def test_get_equatorial_from_position(base_bm):
    """
    Test bm.get_equatorial_from_position
    """

    x, y = 0.9, 27.3
    time = dt(2019, 1, 1, 12, 0, 0, tzinfo=pytz.utc)

    ra, dec = base_bm.get_equatorial_from_position(x, y, time)

    assert isinstance(ra, float)
    assert isinstance(dec, float)

    assert ra == pytest.approx(157.0692509868087)
    assert dec == pytest.approx(76.69461175016062)


def test_get_position_from_equatorial(base_bm):
    """
    Test bm.get_position_from_equatorial
    """

    ra, dec = 33.0, -10.1
    time = dt(2019, 1, 1, 12, 0, 0, tzinfo=pytz.utc)

    x, y = base_bm.get_position_from_equatorial(ra, dec, time)

    assert isinstance(x, float)
    assert isinstance(y, float)

    assert x == pytest.approx(116.52026429926534)
    assert y == pytest.approx(20.21033634709244)


def test_get_cartesian_from_position(base_bm):
    """
    Test bm.get_cartesian_from_position
    """
    x, y = -10.0, 33.0

    x_tel, y_tel = base_bm.get_cartesian_from_position(x, y)

    assert isinstance(x_tel, float)
    assert isinstance(y_tel, float)

    assert x_tel == pytest.approx(0.1732758606763881)
    assert y_tel == pytest.approx(0.5446390350150272)


def test_get_position_from_cartesian(base_bm):
    """
    Test bm.get_position_from_cartesian
    """
    x_tel, y_tel = -0.1, 0.4

    x, y = base_bm.get_position_from_cartesian(x_tel, y_tel)

    assert isinstance(x, float)
    assert isinstance(y, float)

    assert x == pytest.approx(5.741007497738201)
    assert y == pytest.approx(23.578178478201835)


def test_is_position_above_horizon(base_bm):
    """
    Test bm.is_position_above_horizon
    """

    x, y = 10.0, 10.0

    output = base_bm.is_position_above_horizon(x, y)

    assert isinstance(output, bool) or isinstance(output, np.bool_)

    assert output == True


def test_is_equatorial_above_horizon(base_bm):
    """
    Test bm.is_equatorial_above_horizon
    """

    ra, dec = 66.0, 20.0
    time = dt(2019, 1, 1, 12, 0, 0, tzinfo=pytz.utc)

    output = base_bm.is_equatorial_above_horizon(ra, dec, time)

    assert isinstance(output, bool) or isinstance(output, np.bool_)

    assert output == True


def test_get_beam_positions(bm, basic_data):
    """
    Test bm.get_beam_positions
    """

    beam_ids = basic_data[0]
    freqs = basic_data[1]

    position = bm.get_beam_positions(beam_ids, freqs)

    assert isinstance(position, np.ndarray)
    assert position.shape == (len(beam_ids), len(freqs), 2)


def test_get_beam_widths(bm, basic_data):
    """
    Test bm.get_beam_widths
    """

    beam_ids = basic_data[0]
    freqs = basic_data[1]

    widths = bm.get_beam_widths(beam_ids, freqs)

    assert isinstance(widths, np.ndarray)
    assert widths.shape == (len(beam_ids), len(freqs), 2)


def test_get_sensitivity(bm, basic_data):
    """
    Test bm.get_sensitivity
    """

    beam_ids = basic_data[0]
    freqs = basic_data[1]
    positions = basic_data[2]

    sensitivity = bm.get_sensitivity(beam_ids, positions, freqs)

    assert isinstance(sensitivity, np.ndarray)
    assert sensitivity.shape == (len(positions), len(beam_ids), len(freqs))


def test_speed_of_light_config():
    """
    Test different speed of light options for before and after July 10th, 2019
    """

    # Dummy data
    freqs = np.array(constants.FREQ)
    beam_id = 1013
    position = np.array([[3.68069918e-05, -5.03180833e01]])

    # Get sensitivities with "legacy" speed of light: 3e8m/s
    config = cfbm.current_config
    config["speed_of_light_mode"] = "legacy"
    bm = cfbm.FFTFormedSincNSBeamModel(config)
    legacy_sensitivities = bm.get_sensitivity(beam_id, position, freqs)

    # Get sensitivities with "current" speed of light: 299792458m/s
    config["speed_of_light_mode"] = "current"
    bm = cfbm.FFTFormedSincNSBeamModel(config)
    current_sensitivities = bm.get_sensitivity(beam_id, position, freqs)

    assert isinstance(legacy_sensitivities, np.ndarray)
    assert isinstance(current_sensitivities, np.ndarray)
    assert legacy_sensitivities.shape == (1, 1, len(freqs))
    assert current_sensitivities.shape == (1, 1, len(freqs))
    assert not np.allclose(legacy_sensitivities, current_sensitivities, rtol=1e-2)

    # Trigger exception
    with pytest.raises(Exception):
        config["speed_of_light_mode"] = "not_a_valid_option"
        bm = cfbm.FFTFormedSincNSBeamModel(config)
    # Change the config back to something acceptable for further tests
    config["speed_of_light_mode"] = "current"


def test_clamping_1k_config(basic_data):
    """
    Test calculating clamping using the 1k freq resolution at L0
    """

    # Dummy data
    freqs = np.array(constants.FREQ)
    beam_id = 1013
    position = np.array([[3.68069918e-05, -5.03180833e01]])

    # Get sensitivities with "legacy" speed of light: 3e8m/s
    config = cfbm.current_config
    config["use_1k_freqs_for_clamping"] = True
    bm = cfbm.FFTFormedSincNSBeamModel(config)
    clamping_1k_sensitivities = bm.get_sensitivity(beam_id, position, freqs)

    # Get sensitivities with "current" speed of light: 299792458m/s
    config["use_1k_freqs_for_clamping"] = False
    bm = cfbm.FFTFormedSincNSBeamModel(config)
    sensitivities = bm.get_sensitivity(beam_id, position, freqs)

    assert isinstance(sensitivities, np.ndarray)
    assert isinstance(clamping_1k_sensitivities, np.ndarray)
    assert clamping_1k_sensitivities.shape == (1, 1, len(freqs))
    assert sensitivities.shape == (1, 1, len(freqs))
    assert not np.allclose(clamping_1k_sensitivities, sensitivities, rtol=1e-2)

    # Trigger exception
    with pytest.raises(Exception):
        freqs = np.array(constants.FPGA_FREQ)[::-1]
        config["use_1k_freqs_for_clamping"] = True
        bm = cfbm.FFTFormedSincNSBeamModel(config)
        sensitivities = bm.get_sensitivity(beam_id, position, freqs)
