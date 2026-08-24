"""Classical orbital elements: closed-form checks, a published case, and singular geometry.

Nothing here asserts a golden number produced by an earlier run of ``rpo_core.elements``.
The oracles are, in order of strength:

1. A worked example from Curtis with printed element values (cited at the test).
2. ``scipy.spatial.transform.Rotation`` as an independent implementation of the 3-1-3
   perifocal-to-inertial rotation, so the forward map is checked against somebody else's
   arithmetic rather than against its own inverse.
3. ``rpo_core.propagate.propagate_two_body``, already validated to conserve energy and
   angular momentum to better than 1e-10 relative over ten orbits. Under two-body motion
   five of the six elements are constants of the motion, which makes the propagator a
   completely independent check on the conversion.
4. Closed-form geometry: ``r_p = a(1-e)``, ``r_a = a(1+e)``, vis-viva, and the definitions
   of polar and retrograde.
"""

import math

import numpy as np
import pytest
from rpo_core.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M, orbital_period_s
from rpo_core.elements import (
    CIRCULAR_ECCENTRICITY_TOL,
    EQUATORIAL_SINE_TOL,
    MAX_CLOSED_ECCENTRICITY,
    ClassicalElements,
    NonClosedOrbitError,
    UndefinedOrbitalElementError,
    argument_of_latitude_rad,
    cartesian_to_classical,
    classical_to_cartesian,
    eccentricity_vector,
    inclination_rad,
    longitude_of_periapsis_rad,
    true_longitude_rad,
)
from rpo_core.exceptions import DegenerateGeometryError
from rpo_core.propagate import propagate_two_body
from scipy.spatial.transform import Rotation

MU = MU_EARTH_M3_S2
A_LEO_M = R_EARTH_EQUATORIAL_M + 420.0e3

#: Functions that take ``(state_eci, mu_m3_s2)``. Every input-validation branch is checked
#: against all of them, so a new entry point cannot skip the shared validation.
STATE_FUNCTIONS = (
    cartesian_to_classical,
    eccentricity_vector,
    inclination_rad,
    argument_of_latitude_rad,
    true_longitude_rad,
    longitude_of_periapsis_rad,
)

#: A generic well-posed orbit: inclined, eccentric, every angle in a different quadrant.
GENERIC = ClassicalElements(
    semi_major_axis_m=9.0e6,
    eccentricity=0.25,
    inclination_rad=math.radians(53.0),
    raan_rad=math.radians(200.0),
    arg_periapsis_rad=math.radians(310.0),
    true_anomaly_rad=math.radians(120.0),
)


def state_via_scipy(elements: ClassicalElements, mu: float = MU) -> np.ndarray:
    """Build the ECI state from elements using scipy's rotation as an independent oracle.

    The perifocal conic is written out from its definition; the 3-1-3 rotation comes from
    ``Rotation.from_euler("ZXZ", ...)``, i.e. an intrinsic Z-X-Z sequence, which is exactly
    ``R_z(Omega) R_x(i) R_z(omega)``. No ``rpo_core.elements`` code is involved.
    """
    e = elements.eccentricity
    nu = elements.true_anomaly_rad
    p = elements.semi_major_axis_m * (1.0 - e**2)
    r_pf = (p / (1.0 + e * math.cos(nu))) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    rotation = Rotation.from_euler(
        "ZXZ",
        [elements.raan_rad, elements.inclination_rad, elements.arg_periapsis_rad],
    ).as_matrix()
    return np.concatenate((rotation @ r_pf, rotation @ v_pf))


def angle_difference(a: float, b: float) -> float:
    """Return the smallest absolute difference between two angles, radians."""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


# --------------------------------------------------------------------------------------
# 1. The forward map against an independent implementation of the rotation
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "elements",
    [
        GENERIC,
        ClassicalElements(7.5e6, 0.001, math.radians(98.0), 0.1, 3.0, 5.5),
        ClassicalElements(4.2164e7, 0.7, math.radians(140.0), 4.9, 1.7, 0.4),
        ClassicalElements(2.6e7, 0.6, math.radians(63.4), 2.2, math.pi / 2.0, math.pi),
    ],
)
def test_forward_map_matches_an_independently_constructed_rotation(elements):
    """COE2RV must agree with scipy's 3-1-3 rotation applied to the perifocal conic.

    This pins the rotation *convention*, which a round-trip test cannot: a consistently
    transposed or wrongly-ordered rotation inverts itself perfectly.
    """
    reference = state_via_scipy(elements)
    produced = classical_to_cartesian(elements, MU)
    error = float(np.linalg.norm(produced - reference) / np.linalg.norm(reference))
    # Measured max over these four cases and 200 random orbits: 5.8e-16, i.e. a few ULP.
    # 1e-13 leaves ~2.5 decades of headroom while still catching any real convention error,
    # which would show up at order 1.
    assert error < 1e-13, f"forward map deviates from the scipy oracle by {error:.3e} relative"


# --------------------------------------------------------------------------------------
# 2. Round trip over many random orbits
# --------------------------------------------------------------------------------------

#: Bound for the random round trip, set from measurement rather than feel. Worst per-element
#: relative error measured over 3000 orbits at each of the seeds 1, 7, 20260824 and 99999
#: was 4.9e-13, always in the eccentricity channel and always on the smallest-e samples
#: (relative error there is the double-precision floor divided by e ~ 1e-4). 1e-10 is a
#: little over two decades of headroom: loose enough not to fail on seed variation, tight
#: enough that any genuine formula or quadrant error -- which lands at order 1e-1 to 1 --
#: is caught by several orders of magnitude. See
#: ``test_round_trip_bound_rejects_a_perturbed_element`` for proof it is not a plateau.
ROUND_TRIP_REL_BOUND = 1e-10


def random_elements(rng, count):
    """Draw ``count`` well-posed, non-degenerate element sets.

    Bounds chosen to stay clear of the singular geometries by a wide margin: ``e`` no
    smaller than 1e-4 (four decades above :data:`CIRCULAR_ECCENTRICITY_TOL`) and ``i`` no
    closer than 0.5 deg to 0 or 180 deg (six decades above
    :data:`EQUATORIAL_SINE_TOL`). Degeneracy is the subject of its own tests below; mixing
    it into the accuracy measurement would only blur the bound.
    """
    return [
        ClassicalElements(
            semi_major_axis_m=float(a),
            eccentricity=float(e),
            inclination_rad=float(i),
            raan_rad=float(raan),
            arg_periapsis_rad=float(argp),
            true_anomaly_rad=float(nu),
        )
        for a, e, i, raan, argp, nu in zip(
            rng.uniform(7.0e6, 4.3e7, count),
            rng.uniform(1.0e-4, 0.85, count),
            rng.uniform(math.radians(0.5), math.radians(179.5), count),
            rng.uniform(0.0, 2.0 * math.pi, count),
            rng.uniform(0.0, 2.0 * math.pi, count),
            rng.uniform(0.0, 2.0 * math.pi, count),
            strict=True,
        )
    ]


def round_trip_errors(elements_list):
    """Return the worst per-element relative error over an elements -> state -> elements trip."""
    worst = np.zeros(6)
    for reference in elements_list:
        recovered = cartesian_to_classical(classical_to_cartesian(reference, MU), MU)
        worst = np.maximum(
            worst,
            np.array(
                [
                    abs(recovered.semi_major_axis_m - reference.semi_major_axis_m)
                    / reference.semi_major_axis_m,
                    abs(recovered.eccentricity - reference.eccentricity) / reference.eccentricity,
                    abs(recovered.inclination_rad - reference.inclination_rad)
                    / reference.inclination_rad,
                    # Angles are compared as a fraction of a full turn: a relative error on
                    # an angle that happens to be near zero is meaningless.
                    angle_difference(recovered.raan_rad, reference.raan_rad) / (2.0 * math.pi),
                    angle_difference(recovered.arg_periapsis_rad, reference.arg_periapsis_rad)
                    / (2.0 * math.pi),
                    angle_difference(recovered.true_anomaly_rad, reference.true_anomaly_rad)
                    / (2.0 * math.pi),
                ]
            ),
        )
    return worst


@pytest.mark.unit
def test_round_trip_over_three_thousand_random_orbits(capsys):
    rng = np.random.default_rng(20260824)
    worst = round_trip_errors(random_elements(rng, 3000))
    with capsys.disabled():
        labels = ("a", "e", "i", "raan", "argp", "nu")
        print("\nround-trip worst relative error by element:")
        for label, value in zip(labels, worst, strict=True):
            print(f"  {label:>4s}: {value:.3e}")
        print(f"  max : {worst.max():.3e} (bound {ROUND_TRIP_REL_BOUND:.0e})")
    assert worst.max() < ROUND_TRIP_REL_BOUND


@pytest.mark.unit
@pytest.mark.parametrize("seed", [1, 7, 99999])
def test_round_trip_bound_holds_across_seeds(seed):
    """The bound must not depend on the one seed it was measured with."""
    worst = round_trip_errors(random_elements(np.random.default_rng(seed), 500))
    assert worst.max() < ROUND_TRIP_REL_BOUND


@pytest.mark.unit
@pytest.mark.parametrize("field", ["eccentricity", "inclination_rad", "raan_rad"])
def test_round_trip_bound_rejects_a_perturbed_element(field):
    """Complement test: the bound is a knife edge, not a plateau.

    A round-trip assertion is only meaningful if it would fail on a wrong answer. Perturbing
    one element by 1e-8 relative -- far below anything a real bug would produce, and far
    below the resolution of a plotted trajectory -- must already breach the bound.
    """
    perturbed = ClassicalElements(
        **{
            **{
                "semi_major_axis_m": GENERIC.semi_major_axis_m,
                "eccentricity": GENERIC.eccentricity,
                "inclination_rad": GENERIC.inclination_rad,
                "raan_rad": GENERIC.raan_rad,
                "arg_periapsis_rad": GENERIC.arg_periapsis_rad,
                "true_anomaly_rad": GENERIC.true_anomaly_rad,
            },
            field: getattr(GENERIC, field) * (1.0 + 1e-8),
        }
    )
    recovered = cartesian_to_classical(classical_to_cartesian(perturbed, MU), MU)
    drift = max(
        abs(recovered.eccentricity - GENERIC.eccentricity) / GENERIC.eccentricity,
        angle_difference(recovered.inclination_rad, GENERIC.inclination_rad) / (2.0 * math.pi),
        angle_difference(recovered.raan_rad, GENERIC.raan_rad) / (2.0 * math.pi),
    )
    assert drift > ROUND_TRIP_REL_BOUND, (
        f"a 1e-8 relative perturbation of {field} moved the round trip by only {drift:.3e}; "
        "the bound is too loose to detect a real error"
    )


# --------------------------------------------------------------------------------------
# 3. A published worked example
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_curtis_example_4_3_published_elements():
    """Reproduce Curtis, *Orbital Mechanics for Engineering Students*, Example 4.3.

    Given (in the textbook's units) ``r = [-6045, -3490, 2500]`` km and
    ``v = [-3.457, 6.618, 2.533]`` km/s with ``mu = 398600`` km^3/s^2, Curtis prints:

    ======================= ==================
    ``h``                   58,310 km^2/s
    ``i``                   153.2 deg
    ``Omega``               255.3 deg
    ``e``                   0.1712
    ``omega``               20.07 deg
    ``nu``                  28.45 deg
    ``a``                   8788 km
    ======================= ==================

    Each value is asserted by rounding this implementation's result to the number of digits
    the book prints and comparing for exact equality. That is stronger than picking a
    tolerance and it cannot be tuned after the fact: either the printed digits come out or
    they do not.

    Curtis also prints ``r_p = 7284`` km, ``r_a = 10,292`` km and ``T = 2.278`` h. Those are
    *not* asserted here: he computes them from the already-rounded ``h`` and ``e`` above, so
    they differ from the exact values in the last printed digit (7283.5 km, 10,292.7 km,
    2.2775 h). Asserting them would be asserting the textbook's intermediate rounding, not
    its physics. The same geometry is checked exactly in
    ``test_periapsis_and_apoapsis_radii_follow_from_a_and_e``.
    """
    mu_curtis = 398600.0 * 1e9  # km^3/s^2 -> m^3/s^2; Curtis's value, not WGS-84's.
    state = np.concatenate(
        (np.array([-6045.0, -3490.0, 2500.0]) * 1e3, np.array([-3.457, 6.618, 2.533]) * 1e3)
    )

    elements = cartesian_to_classical(state, mu_curtis)
    h_km2_s = float(np.linalg.norm(np.cross(state[:3], state[3:]))) / 1e6

    assert round(h_km2_s, -1) == 58310.0
    assert round(math.degrees(elements.inclination_rad), 1) == 153.2
    assert round(math.degrees(elements.raan_rad), 1) == 255.3
    assert round(elements.eccentricity, 4) == 0.1712
    assert round(math.degrees(elements.arg_periapsis_rad), 2) == 20.07
    assert round(math.degrees(elements.true_anomaly_rad), 2) == 28.45
    assert round(elements.semi_major_axis_m / 1e3) == 8788


@pytest.mark.unit
def test_curtis_example_4_3_round_trips_back_to_the_published_state():
    """Complement to the previous test: the published elements must regenerate the state."""
    mu_curtis = 398600.0 * 1e9
    state = np.concatenate(
        (np.array([-6045.0, -3490.0, 2500.0]) * 1e3, np.array([-3.457, 6.618, 2.533]) * 1e3)
    )
    rebuilt = classical_to_cartesian(cartesian_to_classical(state, mu_curtis), mu_curtis)
    error = float(np.linalg.norm(rebuilt - state) / np.linalg.norm(state))
    assert error < 1e-13, f"state -> elements -> state drifted {error:.3e} relative"


# --------------------------------------------------------------------------------------
# 4. Cross-check against the validated two-body propagator
# --------------------------------------------------------------------------------------

PROPAGATED = ClassicalElements(
    semi_major_axis_m=8.0e6,
    eccentricity=0.2,
    inclination_rad=math.radians(40.0),
    raan_rad=math.radians(63.0),
    arg_periapsis_rad=math.radians(137.0),
    true_anomaly_rad=math.radians(17.0),
)


@pytest.fixture(scope="module")
def one_orbit():
    """Propagated states and their elements at 61 points over exactly one period."""
    state0 = classical_to_cartesian(PROPAGATED, MU)
    period = orbital_period_s(PROPAGATED.semi_major_axis_m, MU)
    states = propagate_two_body(state0, np.linspace(0.0, period, 61), MU)
    return states, [cartesian_to_classical(state, MU) for state in states]


@pytest.fixture(scope="module")
def one_orbit_elements(one_orbit):
    """Just the elements from :func:`one_orbit`."""
    return one_orbit[1]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("field", "bound"),
    [
        # Bounds measured over this exact propagation. a/e drift at 1e-11 relative is the
        # DOP853 integration error at rtol=1e-12, not a conversion error -- the propagator's
        # own energy-conservation test bounds it at 1e-10. The angles hold at ~1e-10 rad.
        ("semi_major_axis_m", 1e-9),
        ("eccentricity", 1e-9),
        ("inclination_rad", 1e-9),
        ("raan_rad", 1e-9),
        ("arg_periapsis_rad", 1e-9),
    ],
)
def test_five_elements_are_constants_of_two_body_motion(one_orbit_elements, field, bound):
    """Independent oracle: under two-body motion only the true anomaly changes.

    ``propagate_two_body`` is validated separately (energy and angular momentum conserved to
    better than 1e-10 relative over ten orbits). If the conversion had a quadrant flip, a
    swapped angle, or a sign error, the "constant" elements would jump as the spacecraft
    crossed the node, the equator, or apoapsis -- which is exactly what sampling a whole
    revolution exercises.
    """
    values = np.array([getattr(element, field) for element in one_orbit_elements])
    reference = getattr(PROPAGATED, field)
    if field.endswith("_rad"):
        drift = max(angle_difference(float(value), reference) for value in values)
    else:
        drift = float(np.ptp(values) / abs(reference))
    assert drift < bound, f"{field} drifted by {drift:.3e} over one orbit"


@pytest.mark.integration
def test_true_anomaly_sweeps_exactly_one_revolution_over_one_period(one_orbit_elements):
    """Complement to the invariance test: something must actually move.

    Five constants would also be produced by a conversion that ignored its input entirely.
    The true anomaly must advance monotonically through a full 2*pi -- including the wrap at
    periapsis, which is where a quadrant error in ``nu`` would show up as a fold rather than
    a wrap.
    """
    nu = np.array([element.true_anomaly_rad for element in one_orbit_elements])
    unwrapped = np.unwrap(nu)
    assert np.all(np.diff(unwrapped) > 0.0), "true anomaly is not monotonically increasing"
    assert float(unwrapped[-1] - unwrapped[0]) == pytest.approx(2.0 * math.pi, abs=1e-8)


@pytest.mark.integration
def test_true_anomaly_places_apoapsis_and_periapsis_on_the_right_side_of_the_orbit(one_orbit):
    """The propagated radius must peak at ``nu = pi`` and bottom out at ``nu = 0``.

    A sign error in the ``nu`` quadrant test maps the outbound leg onto the inbound one.
    That leaves the round trip and the five invariants intact -- both are symmetric under
    ``nu -> -nu`` -- while putting apoapsis on the wrong side of the orbit. The oracle here
    is the *propagated* position, which knows nothing about the conversion:

    1. Every propagated radius must satisfy the conic equation at its recovered ``nu``.
    2. The largest radius must occur at the sample whose ``nu`` is nearest ``pi``, and the
       smallest at the sample whose ``nu`` is nearest ``0``.
    """
    states, elements = one_orbit
    radii = np.linalg.norm(states[:, :3], axis=1)
    nu = np.array([element.true_anomaly_rad for element in elements])

    for radius, element in zip(radii, elements, strict=True):
        a, e = element.semi_major_axis_m, element.eccentricity
        expected = a * (1.0 - e**2) / (1.0 + e * math.cos(element.true_anomaly_rad))
        assert float(radius) == pytest.approx(expected, rel=1e-12)

    distance_to_apoapsis = np.array([angle_difference(value, math.pi) for value in nu])
    distance_to_periapsis = np.array([angle_difference(value, 0.0) for value in nu])
    assert int(np.argmax(radii)) == int(np.argmin(distance_to_apoapsis))
    assert int(np.argmin(radii)) == int(np.argmin(distance_to_periapsis))


# --------------------------------------------------------------------------------------
# 5. Known geometry
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_circular_equatorial_orbit_has_zero_eccentricity_and_zero_inclination():
    """Hand-built, no elements involved: r along x, v along y at exactly circular speed."""
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, speed, 0.0])
    assert float(np.linalg.norm(eccentricity_vector(state, MU))) < 1e-15
    assert inclination_rad(state, MU) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_polar_orbit_has_inclination_of_exactly_ninety_degrees():
    """Position along x, velocity along z: the orbit plane contains the polar axis."""
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, 0.0, speed])
    assert inclination_rad(state, MU) == pytest.approx(math.pi / 2.0, abs=1e-15)


@pytest.mark.unit
def test_retrograde_equatorial_orbit_has_inclination_of_exactly_pi():
    """Complement to the prograde case: reversing the velocity must flip i from 0 to pi."""
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, -speed, 0.0])
    assert inclination_rad(state, MU) == pytest.approx(math.pi, abs=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("eccentricity", [0.0, 0.05, 0.3, 0.75])
def test_periapsis_and_apoapsis_radii_follow_from_a_and_e(eccentricity):
    """``r(nu=0) = a(1-e)`` and ``r(nu=pi) = a(1+e)`` -- the definition of the conic."""
    a = 1.2e7
    periapsis = classical_to_cartesian(ClassicalElements(a, eccentricity, 0.6, 1.0, 2.0, 0.0), MU)
    apoapsis = classical_to_cartesian(
        ClassicalElements(a, eccentricity, 0.6, 1.0, 2.0, math.pi), MU
    )
    assert float(np.linalg.norm(periapsis[:3])) == pytest.approx(
        a * (1.0 - eccentricity), rel=1e-14
    )
    assert float(np.linalg.norm(apoapsis[:3])) == pytest.approx(a * (1.0 + eccentricity), rel=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize("eccentricity", [0.0, 0.05, 0.3, 0.75])
def test_vis_viva_holds_at_periapsis_and_apoapsis(eccentricity):
    """``v**2 = mu(2/r - 1/a)`` at both apsides, from the energy integral.

    Independent of everything the conversion does with angles: it only needs ``a`` and the
    radius to be right, so it catches a mis-scaled perifocal velocity that a round trip
    would happily invert.
    """
    a = 1.2e7
    for nu, radius in ((0.0, a * (1.0 - eccentricity)), (math.pi, a * (1.0 + eccentricity))):
        state = classical_to_cartesian(ClassicalElements(a, eccentricity, 0.6, 1.0, 2.0, nu), MU)
        expected = math.sqrt(MU * (2.0 / radius - 1.0 / a))
        assert float(np.linalg.norm(state[3:])) == pytest.approx(expected, rel=1e-14)


@pytest.mark.unit
def test_vis_viva_holds_everywhere_on_a_random_orbit():
    """Conservation law over a whole orbit, not just the apsides."""
    rng = np.random.default_rng(4242)
    for elements in random_elements(rng, 200):
        state = classical_to_cartesian(elements, MU)
        radius = float(np.linalg.norm(state[:3]))
        expected = math.sqrt(MU * (2.0 / radius - 1.0 / elements.semi_major_axis_m))
        assert float(np.linalg.norm(state[3:])) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------------------
# 6. Quadrant correctness
# --------------------------------------------------------------------------------------

QUADRANT_ANGLES = [math.radians(deg) for deg in (35.0, 125.0, 215.0, 305.0)]


@pytest.mark.unit
@pytest.mark.parametrize("raan", QUADRANT_ANGLES)
@pytest.mark.parametrize("argp", QUADRANT_ANGLES)
@pytest.mark.parametrize("nu", QUADRANT_ANGLES)
@pytest.mark.parametrize("inclination_deg", [23.0, 97.0, 152.0])
def test_all_angles_recover_in_every_quadrant(raan, argp, nu, inclination_deg):
    """Every combination of quadrants, prograde / polar / retrograde.

    The state is built by :func:`state_via_scipy`, so the expected angles are the inputs to
    an independent rotation, not the output of ``classical_to_cartesian``. A quadrant test
    dropped from RV2COE reflects the angle about 0 or pi, which is a change of order 1 rad
    here and cannot hide behind the tolerance.
    """
    reference = ClassicalElements(1.1e7, 0.3, math.radians(inclination_deg), raan, argp, nu)
    recovered = cartesian_to_classical(state_via_scipy(reference), MU)
    assert angle_difference(recovered.raan_rad, raan) < 1e-12
    assert angle_difference(recovered.arg_periapsis_rad, argp) < 1e-12
    assert angle_difference(recovered.true_anomaly_rad, nu) < 1e-12
    assert recovered.inclination_rad == pytest.approx(math.radians(inclination_deg), abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("nu_deg", [181.0, 200.0, 270.0, 359.0])
def test_true_anomaly_past_pi_is_not_reflected(nu_deg):
    """Inbound leg (``nu > pi``, ``r . v < 0``) must not be folded onto the outbound leg."""
    reference = ClassicalElements(9.5e6, 0.4, 1.0, 2.0, 3.0, math.radians(nu_deg))
    state = state_via_scipy(reference)
    assert float(np.dot(state[:3], state[3:])) < 0.0, "test case is not on the inbound leg"
    recovered = cartesian_to_classical(state, MU)
    assert angle_difference(recovered.true_anomaly_rad, math.radians(nu_deg)) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raan", "argp", "nu"),
    [
        (math.radians(215.0), math.radians(215.0), math.radians(215.0)),
        (math.radians(215.0), math.radians(305.0), math.radians(190.0)),
        (math.radians(305.0), math.radians(190.0), math.radians(305.0)),
    ],
)
def test_quadrant_tests_are_load_bearing(raan, argp, nu):
    """Complement test: a quadrant-blind RV2COE gives a measurably different answer.

    Recomputing each angle with ``arccos`` alone -- the bug this module is most likely to
    contain -- returns a value on ``[0, pi]``, so it can only be caught by an angle in the
    *lower* half plane. All three angles here are in ``(pi, 2*pi)``, and the blind value
    must differ from the real one by a wide margin. If it did not, the quadrant logic would
    be untested dead weight and every surrounding assertion would still pass with it
    deleted.
    """
    reference = ClassicalElements(1.1e7, 0.3, math.radians(47.0), raan, argp, nu)
    state = state_via_scipy(reference)
    r, v = state[:3], state[3:]
    h = np.cross(r, v)
    node = np.array([-h[1], h[0], 0.0])
    e_vec = (
        (float(np.dot(v, v)) - MU / float(np.linalg.norm(r))) * r - float(np.dot(r, v)) * v
    ) / MU

    blind = {
        "raan_rad": math.acos(node[0] / np.linalg.norm(node)),
        "arg_periapsis_rad": math.acos(
            float(np.dot(node, e_vec)) / (np.linalg.norm(node) * np.linalg.norm(e_vec))
        ),
        "true_anomaly_rad": math.acos(
            float(np.dot(e_vec, r)) / (np.linalg.norm(e_vec) * np.linalg.norm(r))
        ),
    }
    recovered = cartesian_to_classical(state, MU)
    for field, blind_value in blind.items():
        assert angle_difference(getattr(recovered, field), blind_value) > 0.1, (
            f"{field} agrees with the quadrant-blind formula; this case does not exercise "
            "the quadrant test"
        )


# --------------------------------------------------------------------------------------
# 7. Singular geometry: every raise, and every alternate accessor
# --------------------------------------------------------------------------------------


def circular_inclined_state(inclination, raan, argument_of_latitude, a=A_LEO_M):
    """Build a circular inclined state directly from geometry, with a known ``u``.

    The position is the in-plane unit vector at argument of latitude ``u`` rotated by the
    3-1 sequence; the velocity is the in-plane perpendicular at circular speed. No
    eccentricity vector and no ``rpo_core.elements`` code is involved, so the expected ``u``
    is genuinely independent.
    """
    rotation = Rotation.from_euler("ZX", [raan, inclination]).as_matrix()
    in_plane = np.array([math.cos(argument_of_latitude), math.sin(argument_of_latitude), 0.0])
    perpendicular = np.array([-math.sin(argument_of_latitude), math.cos(argument_of_latitude), 0.0])
    return np.concatenate(
        (a * (rotation @ in_plane), math.sqrt(MU / a) * (rotation @ perpendicular))
    )


def equatorial_periapsis_state(longitude_of_periapsis, e=0.2, a=1.0e7):
    """Build an equatorial ellipse at periapsis, with a known longitude of periapsis."""
    r_p = a * (1.0 - e)
    v_p = math.sqrt(MU * (2.0 / r_p - 1.0 / a))
    direction = np.array([math.cos(longitude_of_periapsis), math.sin(longitude_of_periapsis), 0.0])
    across = np.array([-math.sin(longitude_of_periapsis), math.cos(longitude_of_periapsis), 0.0])
    return np.concatenate((r_p * direction, v_p * across))


@pytest.mark.unit
def test_circular_inclined_orbit_refuses_argument_of_periapsis_and_true_anomaly():
    state = circular_inclined_state(math.radians(51.6), math.radians(40.0), math.radians(200.0))
    with pytest.raises(
        UndefinedOrbitalElementError,
        match=r"arg_periapsis_rad and true_anomaly_rad are undefined for a circular orbit",
    ) as excinfo:
        cartesian_to_classical(state, MU)
    assert excinfo.value.undefined_elements == ("arg_periapsis_rad", "true_anomaly_rad")
    assert "argument_of_latitude_rad" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("longitude_deg", [0.0, 70.0, 190.0, 300.0])
def test_argument_of_latitude_is_available_when_the_true_anomaly_is_not(longitude_deg):
    """The alternate must return the right angle for the orbit whose elements are undefined."""
    u = math.radians(longitude_deg)
    state = circular_inclined_state(math.radians(51.6), math.radians(40.0), u)
    with pytest.raises(UndefinedOrbitalElementError):
        cartesian_to_classical(state, MU)
    assert angle_difference(argument_of_latitude_rad(state, MU), u) < 1e-12


@pytest.mark.unit
def test_argument_of_latitude_equals_argp_plus_nu_for_a_non_circular_orbit():
    """Consistency: ``u = omega + nu`` wherever both sides exist.

    Ties the alternate to the primary elements, so the two cannot drift apart under a change
    to either.
    """
    state = state_via_scipy(GENERIC)
    expected = GENERIC.arg_periapsis_rad + GENERIC.true_anomaly_rad
    assert angle_difference(argument_of_latitude_rad(state, MU), expected) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("retrograde", [False, True])
def test_equatorial_elliptical_orbit_refuses_raan(retrograde):
    """Both i ~ 0 and i ~ pi are equatorial; the sine-based threshold must catch both."""
    state = equatorial_periapsis_state(math.radians(75.0))
    if retrograde:
        state = state.copy()
        state[3:] *= -1.0
    with pytest.raises(
        UndefinedOrbitalElementError,
        match=r"raan_rad is undefined for an equatorial orbit",
    ) as excinfo:
        cartesian_to_classical(state, MU)
    assert excinfo.value.undefined_elements == ("raan_rad",)
    assert "longitude_of_periapsis_rad" in str(excinfo.value)
    # Complement: inclination itself is still perfectly well defined, and distinguishes the
    # two cases the single error message covers.
    expected_inclination = math.pi if retrograde else 0.0
    assert inclination_rad(state, MU) == pytest.approx(expected_inclination, abs=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("longitude_deg", [0.0, 75.0, 210.0, 345.0])
def test_longitude_of_periapsis_is_available_when_raan_is_not(longitude_deg):
    varpi = math.radians(longitude_deg)
    state = equatorial_periapsis_state(varpi)
    with pytest.raises(UndefinedOrbitalElementError):
        cartesian_to_classical(state, MU)
    assert angle_difference(longitude_of_periapsis_rad(state, MU), varpi) < 1e-12


@pytest.mark.unit
def test_circular_equatorial_orbit_refuses_all_three_angles():
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, speed, 0.0])
    with pytest.raises(
        UndefinedOrbitalElementError,
        match=(
            r"raan_rad, arg_periapsis_rad and true_anomaly_rad are undefined for a "
            r"circular equatorial orbit"
        ),
    ) as excinfo:
        cartesian_to_classical(state, MU)
    assert excinfo.value.undefined_elements == (
        "raan_rad",
        "arg_periapsis_rad",
        "true_anomaly_rad",
    )
    assert "true_longitude_rad" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("longitude_deg", [0.0, 90.0, 185.0, 300.0])
def test_true_longitude_is_available_when_all_three_angles_are_not(longitude_deg):
    lam = math.radians(longitude_deg)
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array(
        [
            A_LEO_M * math.cos(lam),
            A_LEO_M * math.sin(lam),
            0.0,
            -speed * math.sin(lam),
            speed * math.cos(lam),
            0.0,
        ]
    )
    with pytest.raises(UndefinedOrbitalElementError):
        cartesian_to_classical(state, MU)
    assert angle_difference(true_longitude_rad(state, MU), lam) < 1e-12
    with pytest.raises(UndefinedOrbitalElementError, match=r"circular"):
        longitude_of_periapsis_rad(state, MU)
    with pytest.raises(UndefinedOrbitalElementError, match=r"equatorial"):
        argument_of_latitude_rad(state, MU)


@pytest.mark.unit
def test_argument_of_latitude_refuses_an_equatorial_orbit():
    with pytest.raises(
        UndefinedOrbitalElementError,
        match=r"argument of latitude is undefined for an equatorial orbit",
    ):
        argument_of_latitude_rad(equatorial_periapsis_state(math.radians(75.0)), MU)


@pytest.mark.unit
def test_longitude_of_periapsis_refuses_a_circular_orbit():
    state = circular_inclined_state(math.radians(51.6), math.radians(40.0), math.radians(200.0))
    with pytest.raises(
        UndefinedOrbitalElementError,
        match=r"longitude of periapsis is undefined for a circular orbit",
    ):
        longitude_of_periapsis_rad(state, MU)


@pytest.mark.unit
def test_true_longitude_refuses_a_position_directly_over_the_pole():
    """``atan2(0, 0)`` returns 0.0 silently; this must raise instead."""
    speed = math.sqrt(MU / A_LEO_M)
    state = np.array([0.0, 0.0, A_LEO_M, -speed, 0.0, 0.0])
    with pytest.raises(
        DegenerateGeometryError,
        match=r"position vector lies along the polar axis",
    ):
        true_longitude_rad(state, MU)


@pytest.mark.unit
def test_longitude_of_periapsis_refuses_periapsis_directly_over_the_pole():
    a, e = 1.0e7, 0.2
    r_p = a * (1.0 - e)
    v_p = math.sqrt(MU * (2.0 / r_p - 1.0 / a))
    state = np.array([0.0, 0.0, r_p, v_p, 0.0, 0.0])
    with pytest.raises(
        DegenerateGeometryError,
        match=r"eccentricity vector lies along the polar axis",
    ):
        longitude_of_periapsis_rad(state, MU)


@pytest.mark.unit
def test_longitude_of_periapsis_is_a_right_ascension_not_the_dogleg_sum():
    """Documents a deliberate API choice a reviewer should be able to see, not infer.

    For an *inclined* orbit this module returns ``atan2(e_y, e_x)``, not the classical
    ``varpi = Omega + omega``. The two agree only when the orbit is equatorial, which is the
    only geometry in which the classical element is used. Both ``Omega`` and ``omega`` are
    separately available for an inclined orbit, so nothing is lost.
    """
    equatorial = equatorial_periapsis_state(math.radians(75.0))
    assert angle_difference(longitude_of_periapsis_rad(equatorial, MU), math.radians(75.0)) < 1e-12

    inclined = state_via_scipy(GENERIC)
    dogleg = GENERIC.raan_rad + GENERIC.arg_periapsis_rad
    assert angle_difference(longitude_of_periapsis_rad(inclined, MU), dogleg) > 0.1


# --- threshold knife edges -------------------------------------------------------------


@pytest.mark.unit
def test_circular_threshold_is_a_knife_edge():
    """Just below the tolerance raises; just above converts and returns the eccentricity."""
    below = ClassicalElements(A_LEO_M, 0.4 * CIRCULAR_ECCENTRICITY_TOL, 0.9, 0.5, 1.2, 2.0)
    with pytest.raises(UndefinedOrbitalElementError, match=r"circular orbit"):
        cartesian_to_classical(classical_to_cartesian(below, MU), MU)

    above = ClassicalElements(A_LEO_M, 2.5 * CIRCULAR_ECCENTRICITY_TOL, 0.9, 0.5, 1.2, 2.0)
    recovered = cartesian_to_classical(classical_to_cartesian(above, MU), MU)
    assert recovered.eccentricity == pytest.approx(above.eccentricity, rel=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("retrograde", [False, True])
def test_equatorial_threshold_is_a_knife_edge(retrograde):
    """The threshold is on sin(i), so it must bite symmetrically at i ~ 0 and i ~ pi."""

    def inclination(factor):
        angle = factor * EQUATORIAL_SINE_TOL
        return math.pi - angle if retrograde else angle

    below = ClassicalElements(A_LEO_M, 0.1, inclination(0.4), 0.5, 1.2, 2.0)
    with pytest.raises(UndefinedOrbitalElementError, match=r"equatorial orbit"):
        cartesian_to_classical(classical_to_cartesian(below, MU), MU)

    above = ClassicalElements(A_LEO_M, 0.1, inclination(2.5), 0.5, 1.2, 2.0)
    recovered = cartesian_to_classical(classical_to_cartesian(above, MU), MU)
    assert recovered.inclination_rad == pytest.approx(above.inclination_rad, rel=1e-6)


# --- non-closed orbits -----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(("name", "speed_factor"), [("parabolic", 1.0), ("hyperbolic", 1.2)])
def test_non_closed_states_are_rejected(name, speed_factor):
    """Escape speed and above: ``a = -mu/(2 xi)`` would be infinite or negative.

    Built from the escape-speed condition ``v = sqrt(2 mu / r)``, which is where ``e = 1``
    by definition -- not from an element set, since ``ClassicalElements`` refuses to hold
    one.
    """
    escape_speed = math.sqrt(2.0 * MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, speed_factor * escape_speed, 0.0])
    with pytest.raises(NonClosedOrbitError, match=r"closed orbits only"):
        cartesian_to_classical(state, MU)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
def test_every_accessor_rejects_a_hyperbolic_state(function):
    """The closed-orbit guard lives in the shared preamble, so it must apply everywhere."""
    escape_speed = math.sqrt(2.0 * MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.3 * escape_speed, 1.2 * escape_speed, 0.4e3])
    with pytest.raises(NonClosedOrbitError):
        function(state, MU)


@pytest.mark.unit
def test_a_closed_orbit_just_below_escape_speed_is_still_accepted():
    """Complement: the rejection must be at escape speed, not merely at 'fast'."""
    speed = 0.999 * math.sqrt(2.0 * MU / A_LEO_M)
    state = np.array([A_LEO_M, 0.0, 0.0, 0.0, speed, 1.0])
    elements = cartesian_to_classical(state, MU)
    assert 0.99 < elements.eccentricity < 1.0
    assert elements.semi_major_axis_m > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("eccentricity", [1.0, 1.5, MAX_CLOSED_ECCENTRICITY])
def test_elements_dataclass_rejects_non_closed_eccentricity(eccentricity):
    with pytest.raises(NonClosedOrbitError, match=r"MAX_CLOSED_ECCENTRICITY"):
        ClassicalElements(A_LEO_M, eccentricity, 0.5, 0.5, 0.5, 0.5)


# --------------------------------------------------------------------------------------
# 8. Input validation -- every branch
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize(
    "bad_shape", [np.zeros(3), np.zeros(7), np.zeros((2, 3)), np.zeros((6, 1))]
)
def test_state_functions_reject_a_wrong_shape(function, bad_shape):
    with pytest.raises(ValueError, match=r"must have shape \(6,\)"):
        function(bad_shape, MU)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_state_functions_reject_a_non_finite_state(function, bad_value):
    state = state_via_scipy(GENERIC).copy()
    state[4] = bad_value
    with pytest.raises(ValueError, match=r"must be finite"):
        function(state, MU)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("bad_mu", [0.0, -1.0, -3.986e14, math.nan, math.inf])
def test_state_functions_reject_a_non_positive_or_non_finite_mu(function, bad_mu):
    with pytest.raises(ValueError, match=r"finite positive gravitational parameter"):
        function(state_via_scipy(GENERIC), bad_mu)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
def test_state_functions_reject_a_zero_radius(function):
    with pytest.raises(DegenerateGeometryError, match=r"zero position magnitude"):
        function(np.array([0.0, 0.0, 0.0, 1.0e3, 2.0e3, 3.0e3]), MU)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
def test_state_functions_reject_a_zero_velocity(function):
    with pytest.raises(DegenerateGeometryError, match=r"zero specific angular momentum"):
        function(np.array([A_LEO_M, 0.0, 0.0, 0.0, 0.0, 0.0]), MU)


@pytest.mark.unit
@pytest.mark.parametrize("function", STATE_FUNCTIONS)
def test_state_functions_reject_a_rectilinear_trajectory(function):
    """``r`` parallel to ``v``: no orbit plane, so no inclination and no node line."""
    with pytest.raises(DegenerateGeometryError, match=r"rectilinear"):
        function(np.array([A_LEO_M, 0.0, 0.0, 1.0e3, 0.0, 0.0]), MU)


@pytest.mark.unit
@pytest.mark.parametrize("bad_mu", [0.0, -1.0, math.nan, math.inf])
def test_classical_to_cartesian_rejects_a_bad_mu(bad_mu):
    with pytest.raises(ValueError, match=r"finite positive gravitational parameter"):
        classical_to_cartesian(GENERIC, bad_mu)


@pytest.mark.unit
@pytest.mark.parametrize("not_elements", [None, (9.0e6, 0.25, 0.9, 3.5, 5.4, 2.1), "GENERIC"])
def test_classical_to_cartesian_rejects_a_non_elements_argument(not_elements):
    """A 6-tuple in the right order is the easy mistake; positional silence would be worse."""
    with pytest.raises(ValueError, match=r"must be a ClassicalElements"):
        classical_to_cartesian(not_elements, MU)


@pytest.mark.unit
@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_elements_dataclass_rejects_non_finite_fields(index, bad_value):
    fields = [A_LEO_M, 0.2, 0.5, 1.0, 2.0, 3.0]
    fields[index] = bad_value
    with pytest.raises(ValueError, match=r"must be finite"):
        ClassicalElements(*fields)


@pytest.mark.unit
@pytest.mark.parametrize("semi_major_axis", [0.0, -1.0, -7.0e6])
def test_elements_dataclass_rejects_a_non_positive_semi_major_axis(semi_major_axis):
    with pytest.raises(ValueError, match=r"must be > 0 for a closed orbit"):
        ClassicalElements(semi_major_axis, 0.2, 0.5, 1.0, 2.0, 3.0)


@pytest.mark.unit
def test_elements_dataclass_rejects_a_negative_eccentricity():
    with pytest.raises(ValueError, match=r"eccentricity must be >= 0"):
        ClassicalElements(A_LEO_M, -0.1, 0.5, 1.0, 2.0, 3.0)


@pytest.mark.unit
def test_elements_dataclass_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        GENERIC.eccentricity = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_state_functions_accept_a_plain_list():
    """``npt.ArrayLike`` at the boundary: a list must work exactly like an ndarray."""
    state = state_via_scipy(GENERIC)
    from_list = cartesian_to_classical(list(state), MU)
    from_array = cartesian_to_classical(state, MU)
    assert from_list == from_array
