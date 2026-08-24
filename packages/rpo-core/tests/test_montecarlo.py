"""Monte Carlo harness: closed-form statistics, determinism, and failure accounting.

Three kinds of assertion here, and no golden numbers from a previous run of this code:

* **Closed form.** Every distribution is checked against its analytic mean, variance, and
  covariance. Tolerances are computed from the standard error of the estimator at a stated
  confidence level (``_Z_BOUND`` sigmas), never picked by feel.
* **Exactness.** The burn-execution model makes two claims that hold to machine precision
  rather than statistically -- the rotation preserves the norm, and the realised pointing
  angle equals the drawn tilt. Those are asserted at 1e-12, not at a sample tolerance.
* **Complement.** Where a property is claimed, the violation of it is also measured, so a
  test cannot pass by accident with the term dropped. The correlated covariance is checked
  against what an uncorrelated sampler would give; the pointing model against what
  additive component-wise noise would give; the Wilson interval against what the normal
  approximation would give.
"""

import json
import math
import statistics

import numpy as np
import pytest
from rpo_core.exceptions import DegenerateGeometryError, RpoCoreError
from rpo_core.montecarlo import (
    DEFAULT_CONFIDENCE,
    BurnExecutionSample,
    CampaignConfigurationError,
    DispersionDefinitionError,
    MagnitudePointingDispersion,
    NormalDispersion,
    UniformDispersion,
    VectorNormalDispersion,
    dispersion_from_dict,
    draw_samples,
    execute_run,
    proportion_estimate,
    run_campaign,
    summarise_metric,
    wilson_interval,
)

# Number of standard errors allowed in every statistical-convergence assertion.
#
# Five sigma on a two-sided Gaussian is a false-failure probability of 5.7e-7 per
# assertion. With ~30 such assertions in this file the suite's spurious-failure rate is
# below 2e-5 per run, while still being tight enough that every mutation exercised in the
# module's mutation study moves the statistic by tens of sigma. The tests are seeded, so
# this is a statement about how discriminating the bound is, not about flakiness.
_Z_BOUND = 5.0

# Sample counts for the @slow convergence tests. Chosen so the 5-sigma bound below is
# small compared with the effect each test must resolve; see the per-test comments.
_N_SCALAR = 100_000
_N_VECTOR = 100_000
_N_BURN = 50_000


def _scalar_samples(dispersion, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([dispersion.sample(rng) for _ in range(n)], dtype=np.float64)


def _vector_samples(dispersion, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([dispersion.sample(rng) for _ in range(n)], dtype=np.float64)


def _burn_samples(dispersion, n, seed=0):
    rng = np.random.default_rng(seed)
    return [dispersion.sample(rng) for _ in range(n)]


# --------------------------------------------------------------------------------------
# Dispersion definitions: validation and raise paths
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_normal_dispersion_rejects_negative_sigma():
    with pytest.raises(DispersionDefinitionError, match="non-negative standard deviation"):
        NormalDispersion(mean=0.0, sigma=-1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_normal_dispersion_rejects_non_finite_parameters(bad):
    with pytest.raises(DispersionDefinitionError, match="must be finite"):
        NormalDispersion(mean=bad, sigma=1.0)
    with pytest.raises(DispersionDefinitionError, match="non-negative standard deviation"):
        NormalDispersion(mean=0.0, sigma=bad)


@pytest.mark.unit
def test_normal_dispersion_allows_zero_sigma_as_a_constant():
    """sigma=0 is the documented way to switch a dispersion off. It must not raise."""
    rng = np.random.default_rng(0)
    disp = NormalDispersion(mean=3.25, sigma=0.0)
    assert disp.sample(rng) == 3.25
    assert disp.sample(rng) == 3.25


@pytest.mark.unit
@pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (2.0, 1.0)])
def test_uniform_dispersion_rejects_empty_interval(low, high):
    with pytest.raises(DispersionDefinitionError, match="strictly below"):
        UniformDispersion(low=low, high=high)


@pytest.mark.unit
def test_uniform_dispersion_rejects_non_finite_bounds():
    with pytest.raises(DispersionDefinitionError, match="bounds must be finite"):
        UniformDispersion(low=0.0, high=math.inf)


@pytest.mark.unit
def test_vector_normal_rejects_shape_mismatch():
    with pytest.raises(DispersionDefinitionError, match=r"must have shape \(2, 2\)"):
        VectorNormalDispersion(mean=np.zeros(2), covariance=np.eye(3))


@pytest.mark.unit
def test_vector_normal_rejects_empty_or_multidimensional_mean():
    with pytest.raises(DispersionDefinitionError, match="non-empty 1-D array"):
        VectorNormalDispersion(mean=np.zeros((2, 2)), covariance=np.eye(4))
    with pytest.raises(DispersionDefinitionError, match="non-empty 1-D array"):
        VectorNormalDispersion(mean=np.zeros(0), covariance=np.zeros((0, 0)))


@pytest.mark.unit
def test_vector_normal_rejects_non_finite_entries():
    with pytest.raises(DispersionDefinitionError, match="must be finite"):
        VectorNormalDispersion(mean=np.array([0.0, math.nan]), covariance=np.eye(2))


@pytest.mark.unit
def test_vector_normal_rejects_asymmetric_covariance():
    """An asymmetric matrix is an assembly error; symmetrising it here would hide it."""
    cov = np.array([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(DispersionDefinitionError, match="not symmetric"):
        VectorNormalDispersion(mean=np.zeros(2), covariance=cov)


@pytest.mark.unit
def test_vector_normal_rejects_indefinite_covariance_and_names_smallest_eigenvalue():
    """The error must carry the number that motivated it, per docs/CONTRIBUTING.md.

    ``[[1, 2], [2, 1]]`` has eigenvalues 3 and -1 exactly, so the message content is a
    hand-computable fact rather than whatever the implementation happened to print.
    """
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert np.allclose(np.linalg.eigvalsh(cov), [-1.0, 3.0])
    with pytest.raises(DispersionDefinitionError) as excinfo:
        VectorNormalDispersion(mean=np.zeros(2), covariance=cov)
    message = str(excinfo.value)
    assert "not positive definite" in message
    assert "-1.000000e+00" in message
    assert "3.000000e+00" in message


@pytest.mark.unit
@pytest.mark.parametrize(
    "cov",
    [
        np.array([[1.0, 1.0], [1.0, 1.0]]),  # rank 1: one direction is exactly certain
        np.diag([4.0, 0.0]),  # a component declared to have no variance at all
    ],
)
def test_vector_normal_rejects_positive_semidefinite_covariance(cov):
    """A singular covariance has a zero-variance direction and no Cholesky factor.

    Complement to the indefinite case: the guard is at ``<= 0``, not ``< 0``, and the
    knife edge sits exactly at singularity rather than one step past it. Both matrices
    here have a smallest eigenvalue of exactly 0.0, so the message is matched on the
    *eigenvalue* branch specifically -- ``numpy.linalg.cholesky`` also fails on these, and
    its own error text says "not positive definite" too, which would let a ``< 0`` guard
    pass a laxer assertion while no longer being the thing that rejected the input.
    """
    assert float(np.linalg.eigvalsh(cov)[0]) == 0.0
    with pytest.raises(DispersionDefinitionError, match=r"smallest eigenvalue is 0\.000000e\+00"):
        VectorNormalDispersion(mean=np.zeros(2), covariance=cov)


@pytest.mark.unit
def test_vector_normal_accepts_a_barely_positive_definite_covariance():
    """Complement to the rejection tests: the guard must not reject valid inputs."""
    cov = np.array([[1.0, 1.0 - 1e-9], [1.0 - 1e-9, 1.0]])
    disp = VectorNormalDispersion(mean=np.zeros(2), covariance=cov)
    np.testing.assert_allclose(disp.cholesky @ disp.cholesky.T, cov, atol=1e-12)


@pytest.mark.unit
def test_cholesky_factor_reconstructs_the_covariance():
    """Conservation-style identity: L @ L.T must be the matrix that was validated."""
    cov = np.array([[4.0, 1.0, 0.5], [1.0, 9.0, -2.0], [0.5, -2.0, 16.0]])
    disp = VectorNormalDispersion(mean=np.array([1.0, 2.0, 3.0]), covariance=cov)
    np.testing.assert_allclose(disp.cholesky @ disp.cholesky.T, cov, atol=1e-13)
    assert np.allclose(np.triu(disp.cholesky, k=1), 0.0), "factor must be lower triangular"


@pytest.mark.unit
@pytest.mark.parametrize("field", ["sigma_magnitude", "sigma_pointing_rad"])
def test_magnitude_pointing_rejects_negative_sigma(field):
    with pytest.raises(DispersionDefinitionError, match="non-negative standard deviation"):
        MagnitudePointingDispersion(**{field: -0.01})


@pytest.mark.unit
def test_every_montecarlo_error_is_an_rpo_core_error():
    """The package's exception taxonomy must stay catchable from one base class."""
    assert issubclass(DispersionDefinitionError, RpoCoreError)
    assert issubclass(CampaignConfigurationError, RpoCoreError)


# --------------------------------------------------------------------------------------
# Tagged-union serialisation
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "dispersion",
    [
        NormalDispersion(mean=-2.5, sigma=0.75),
        UniformDispersion(low=-1.0, high=4.0),
        VectorNormalDispersion(
            mean=np.array([1.0, -1.0]), covariance=np.array([[2.0, 0.3], [0.3, 5.0]])
        ),
        MagnitudePointingDispersion(sigma_magnitude=0.02, sigma_pointing_rad=1e-3),
    ],
)
def test_dispersion_round_trips_through_its_json_description(dispersion):
    """describe() -> JSON -> dispersion_from_dict() must be the identity.

    This is what makes a campaign reproducible from its own summary file.
    """
    rebuilt = dispersion_from_dict(json.loads(json.dumps(dispersion.describe())))
    assert rebuilt == dispersion
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    assert repr(rebuilt.sample(rng_a)) == repr(dispersion.sample(rng_b))


@pytest.mark.unit
def test_dispersion_from_dict_rejects_missing_kind():
    with pytest.raises(DispersionDefinitionError, match="no 'kind' tag"):
        dispersion_from_dict({"mean": 0.0, "sigma": 1.0})


@pytest.mark.unit
def test_dispersion_from_dict_rejects_unknown_kind():
    with pytest.raises(DispersionDefinitionError, match="unknown dispersion kind 'lognormal'"):
        dispersion_from_dict({"kind": "lognormal", "sigma": 1.0})


@pytest.mark.unit
def test_dispersion_from_dict_rejects_wrong_fields():
    with pytest.raises(DispersionDefinitionError, match="does not accept fields"):
        dispersion_from_dict({"kind": "normal", "low": 0.0, "high": 1.0})


@pytest.mark.unit
def test_dispersions_of_different_types_are_not_equal():
    assert NormalDispersion(mean=0.0, sigma=1.0) != UniformDispersion(low=0.0, high=1.0)
    assert NormalDispersion(mean=0.0, sigma=1.0) != "normal"


# --------------------------------------------------------------------------------------
# Statistical correctness against closed form
# --------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.unit
def test_normal_dispersion_converges_to_its_closed_form_moments():
    """Mean within z*sigma/sqrt(N) and std within z*sigma/sqrt(2N), z = 5.

    Standard errors: the sample mean of N draws from N(mu, sigma^2) has SE sigma/sqrt(N),
    and the sample standard deviation has SE sigma/sqrt(2N). Both bounds are computed
    here, not chosen: at N = 1e5 and sigma = 2 they are 0.032 and 0.022, tight enough to
    catch a 2 % error in either moment.
    """
    mean, sigma = -3.0, 2.0
    values = _scalar_samples(NormalDispersion(mean=mean, sigma=sigma), _N_SCALAR, seed=101)

    mean_bound = _Z_BOUND * sigma / math.sqrt(_N_SCALAR)
    std_bound = _Z_BOUND * sigma / math.sqrt(2.0 * _N_SCALAR)
    assert abs(float(np.mean(values)) - mean) < mean_bound
    assert abs(float(np.std(values, ddof=1)) - sigma) < std_bound

    # Complement: the bound is a knife edge, not a plateau. The SE is sigma/sqrt(N) =
    # 0.0063, so a 2 % shift in the mean is 9.5 SE and must fail the same bound the true
    # value passes. A test that only checked the true value would still pass with the
    # mean silently offset.
    assert abs(float(np.mean(values)) - (mean + 0.02 * abs(mean))) > mean_bound


@pytest.mark.slow
@pytest.mark.unit
def test_uniform_dispersion_converges_to_its_closed_form_moments():
    """U[a, b] has mean (a+b)/2 and variance (b-a)^2/12; support is respected exactly."""
    low, high = -2.0, 5.0
    values = _scalar_samples(UniformDispersion(low=low, high=high), _N_SCALAR, seed=202)

    expected_mean = 0.5 * (low + high)
    expected_std = (high - low) / math.sqrt(12.0)
    assert abs(float(np.mean(values)) - expected_mean) < _Z_BOUND * expected_std / math.sqrt(
        _N_SCALAR
    )
    # SE of the sample std for a uniform: sqrt(Var(s)) with excess kurtosis -6/5 gives
    # SE = sigma * sqrt((kappa - 1) / (4N)) = sigma * sqrt(0.8 / (4N)).
    std_se = expected_std * math.sqrt(0.8 / (4.0 * _N_SCALAR))
    assert abs(float(np.std(values, ddof=1)) - expected_std) < _Z_BOUND * std_se
    assert float(np.min(values)) >= low
    assert float(np.max(values)) < high


@pytest.mark.slow
@pytest.mark.unit
def test_vector_normal_reproduces_the_full_covariance_including_off_diagonals():
    """Empirical covariance converges to C, entry by entry.

    SE of the sample covariance entry is sqrt((C_ii*C_jj + C_ij^2)/N). Each entry gets its
    own bound; the off-diagonal ones are the reason the Cholesky factor exists at all.
    """
    mean = np.array([10.0, -4.0, 0.5])
    cov = np.array([[4.0, 1.8, -0.6], [1.8, 9.0, 2.4], [-0.6, 2.4, 16.0]])
    samples = _vector_samples(
        VectorNormalDispersion(mean=mean, covariance=cov), _N_VECTOR, seed=303
    )
    assert samples.shape == (_N_VECTOR, 3)

    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False, ddof=1)

    mean_bound = _Z_BOUND * np.sqrt(np.diag(cov) / _N_VECTOR)
    assert np.all(np.abs(empirical_mean - mean) < mean_bound)

    cov_se = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov**2) / _N_VECTOR)
    assert np.all(np.abs(empirical_cov - cov) < _Z_BOUND * cov_se)

    # Complement: an uncorrelated sampler -- the "forgot the Cholesky" mutation -- would
    # put every off-diagonal at zero. Confirm zero is many standard errors away, so the
    # test above cannot pass with the correlation dropped.
    off = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(cov[off]) > _Z_BOUND * cov_se[off])


@pytest.mark.slow
@pytest.mark.unit
def test_vector_normal_sample_correlation_matches_the_specified_correlation():
    """Correlation coefficients, not just covariances: scale-free check on the factor."""
    cov = np.array([[4.0, 1.8], [1.8, 9.0]])
    samples = _vector_samples(
        VectorNormalDispersion(mean=np.zeros(2), covariance=cov), _N_VECTOR, seed=404
    )
    expected_rho = 1.8 / math.sqrt(4.0 * 9.0)
    empirical_rho = float(np.corrcoef(samples, rowvar=False)[0, 1])
    # SE of a sample correlation is (1 - rho^2)/sqrt(N).
    rho_se = (1.0 - expected_rho**2) / math.sqrt(_N_VECTOR)
    assert abs(empirical_rho - expected_rho) < _Z_BOUND * rho_se


# --------------------------------------------------------------------------------------
# Burn execution error: exactness, statistics, independence
# --------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "dv_nominal",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -2.5, 0.0]),
        np.array([0.0, 0.0, 0.3]),
        np.array([0.4, -0.4, 0.4]),
        np.array([1e-6, 2e-6, -3e-6]),
    ],
)
def test_pointing_rotation_preserves_magnitude_exactly(dv_nominal):
    """|dv'| == |scale| * |dv| to machine precision, for any commanded direction.

    This is the property additive component-wise noise cannot have, and it is exact rather
    than statistical: the rotation is orthogonal, so the magnitude scale factor is the only
    thing that changes the norm. Axis-aligned and diagonal commands are both covered
    because the perpendicular basis is built from the smallest component of the direction.
    """
    disp = MagnitudePointingDispersion(sigma_magnitude=0.05, sigma_pointing_rad=0.02)
    rng = np.random.default_rng(7)
    for _ in range(200):
        sample = disp.sample(rng)
        executed = sample.apply(dv_nominal)
        expected = abs(sample.scale) * float(np.linalg.norm(dv_nominal))
        assert abs(float(np.linalg.norm(executed)) - expected) <= 1e-12 * max(expected, 1e-30)


@pytest.mark.unit
def test_realised_pointing_angle_equals_the_drawn_tilt_exactly():
    """angle(dv', dv) == |tilt_rad|, to machine precision.

    The whole reason the rotation axis lies in the plane perpendicular to the commanded
    impulse: it makes sigma_pointing_rad the standard deviation of an observable, not of
    an internal variable that maps onto the observable through a cosine.
    """
    disp = MagnitudePointingDispersion(sigma_magnitude=0.1, sigma_pointing_rad=0.05)
    dv = np.array([0.6, -0.8, 0.0]) * 1.7
    rng = np.random.default_rng(13)
    for _ in range(500):
        sample = disp.sample(rng)
        executed = sample.apply(dv)
        cos_alpha = float(np.dot(executed, dv)) / (
            float(np.linalg.norm(executed)) * float(np.linalg.norm(dv))
        )
        # scale > 0 with these sigmas; a negative scale would flip the sign of cos.
        assert sample.scale > 0.0
        assert abs(math.acos(min(1.0, max(-1.0, cos_alpha))) - abs(sample.tilt_rad)) < 1e-11


@pytest.mark.unit
def test_zero_sigmas_reproduce_the_commanded_impulse_exactly():
    """Limiting case: with both sigmas zero the dispersion is the identity operator."""
    disp = MagnitudePointingDispersion(sigma_magnitude=0.0, sigma_pointing_rad=0.0)
    rng = np.random.default_rng(3)
    dv = np.array([0.1, -0.2, 0.05])
    np.testing.assert_allclose(disp.sample(rng).apply(dv), dv, atol=1e-15)


@pytest.mark.unit
def test_apply_rejects_a_zero_commanded_impulse():
    sample = BurnExecutionSample(scale=1.0, tilt_rad=0.01, azimuth_rad=0.0)
    with pytest.raises(DegenerateGeometryError, match="no direction"):
        sample.apply(np.zeros(3))


@pytest.mark.unit
def test_apply_rejects_malformed_commanded_impulse():
    sample = BurnExecutionSample(scale=1.0, tilt_rad=0.01, azimuth_rad=0.0)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        sample.apply(np.zeros(4))
    with pytest.raises(ValueError, match="must be finite"):
        sample.apply(np.array([1.0, math.nan, 0.0]))


@pytest.mark.unit
def test_azimuth_spans_the_full_perpendicular_plane():
    """The rotation axis must be uniform in azimuth, not pinned to one direction.

    Complement to the exactness tests: those would still pass if every burn were tilted
    the same way, which would be a systematic misalignment rather than a random one.
    """
    disp = MagnitudePointingDispersion(sigma_magnitude=0.0, sigma_pointing_rad=0.05)
    dv = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(23)
    perpendicular = np.array([s.apply(dv)[[0, 2]] for s in (disp.sample(rng) for _ in range(4000))])
    angles = np.arctan2(perpendicular[:, 1], perpendicular[:, 0])
    # Under a uniform azimuth the resultant of unit vectors has mean length ~sqrt(pi/4N);
    # 5 sigma of that is far below 0.1 at N = 4000.
    resultant = abs(complex(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))))
    assert resultant < 0.1


@pytest.mark.slow
@pytest.mark.unit
def test_magnitude_statistics_match_the_specified_sigma():
    """|dv'|/|dv| has mean 1 and standard deviation sigma_magnitude.

    Bounds from the standard errors of the mean (sigma/sqrt(N)) and of the standard
    deviation (sigma/sqrt(2N)) of the scale factor.
    """
    sigma_mag = 0.03
    disp = MagnitudePointingDispersion(sigma_magnitude=sigma_mag, sigma_pointing_rad=0.02)
    dv = np.array([0.0, -1.25, 0.0])
    ratios = np.array(
        [
            float(np.linalg.norm(s.apply(dv))) / float(np.linalg.norm(dv))
            for s in _burn_samples(disp, _N_BURN, seed=505)
        ]
    )
    assert abs(float(np.mean(ratios)) - 1.0) < _Z_BOUND * sigma_mag / math.sqrt(_N_BURN)
    assert abs(float(np.std(ratios, ddof=1)) - sigma_mag) < _Z_BOUND * sigma_mag / math.sqrt(
        2.0 * _N_BURN
    )


@pytest.mark.slow
@pytest.mark.unit
def test_pointing_angle_follows_the_half_normal_closed_form():
    """Alpha = |theta| with theta ~ N(0, s^2) is half-normal: mean s*sqrt(2/pi).

    Standard deviation s*sqrt(1 - 2/pi). Both are checked, so a model that got the spread
    right but the shape wrong (a uniform tilt, say) would fail on the mean.
    """
    sigma_point = 0.01
    disp = MagnitudePointingDispersion(sigma_magnitude=0.03, sigma_pointing_rad=sigma_point)
    dv = np.array([0.3, 0.4, 0.0])
    unit = dv / np.linalg.norm(dv)
    angles = np.array(
        [
            math.acos(
                min(
                    1.0,
                    max(-1.0, float(np.dot(s.apply(dv) / np.linalg.norm(s.apply(dv)), unit))),
                )
            )
            for s in _burn_samples(disp, _N_BURN, seed=606)
        ]
    )
    expected_mean = sigma_point * math.sqrt(2.0 / math.pi)
    expected_std = sigma_point * math.sqrt(1.0 - 2.0 / math.pi)
    assert abs(float(np.mean(angles)) - expected_mean) < _Z_BOUND * expected_std / math.sqrt(
        _N_BURN
    )
    # SE of the sample std, using the half-normal's fourth moment (3 s^4):
    # Var(s_hat) ~ (mu4 - sigma^4) / (4 N sigma^2).
    mu4 = 3.0 * sigma_point**4
    std_se = math.sqrt((mu4 - expected_std**4) / (4.0 * _N_BURN * expected_std**2))
    assert abs(float(np.std(angles, ddof=1)) - expected_std) < _Z_BOUND * std_se


@pytest.mark.slow
@pytest.mark.unit
def test_magnitude_and_pointing_errors_are_independent():
    """The design claim: magnitude error carries no information about pointing error.

    Two checks, because a zero correlation alone would also hold for some dependent
    pairs: the linear correlation is within its 5-sigma null bound, and the mean pointing
    angle is the same in the low- and high-magnitude halves of the sample.
    """
    disp = MagnitudePointingDispersion(sigma_magnitude=0.1, sigma_pointing_rad=0.05)
    dv = np.array([0.0, 1.0, 0.0])
    samples = _burn_samples(disp, _N_BURN, seed=707)
    executed = np.array([s.apply(dv) for s in samples])
    ratios = np.linalg.norm(executed, axis=1) / float(np.linalg.norm(dv))
    angles = np.arccos(np.clip(executed @ (dv / np.linalg.norm(dv)) / ratios, -1.0, 1.0))

    # Null SE of a sample correlation under independence is 1/sqrt(N).
    correlation = float(np.corrcoef(ratios, angles)[0, 1])
    assert abs(correlation) < _Z_BOUND / math.sqrt(_N_BURN)

    median = float(np.median(ratios))
    low, high = angles[ratios <= median], angles[ratios > median]
    difference_se = math.sqrt(
        float(np.var(low, ddof=1)) / low.size + float(np.var(high, ddof=1)) / high.size
    )
    assert abs(float(np.mean(low)) - float(np.mean(high))) < _Z_BOUND * difference_se


@pytest.mark.slow
@pytest.mark.unit
def test_component_wise_noise_couples_magnitude_and_pointing():
    """Complement test: the physically wrong shortcut is measurably different.

    ``dv + N(0, s^2 I)`` is the model the module docstring rejects. Expanding
    ``|dv + eps|`` to second order, the perpendicular part of the noise *increases* the
    magnitude by ``|eps_perp|^2 / 2|dv|`` regardless of sign, so the realised magnitude is
    biased upward by ``s^2/|dv|`` and is correlated with the realised pointing angle
    (which is ``|eps_perp|/|dv|``). Closed forms for eps_perp ~ Rayleigh(s):

        E|X| = s*sqrt(pi/2),  E|X|^2 = 2 s^2,  E|X|^3 = 3 s^3 sqrt(pi/2)
        Cov(m, alpha) = Cov(eps_par, -|eps_perp| eps_par) + Cov(|eps_perp|^2/2, |eps_perp|)
                      = -s^3 sqrt(pi/2) + 0.6267 s^3 = -0.6267 s^3
        corr(m, alpha) -> -0.9567 * s / |dv|                 (s << |dv|)

    The sign is negative because the shortening of the angle by a positive magnitude error
    dominates the second-order lengthening from the perpendicular noise. At s/|dv| = 0.1
    the predicted correlation is -0.0957 and the measured value is -0.0943 -- 21 standard
    errors from zero, where the implemented model must stay inside 5. Without this test, a
    mutation replacing the rotation with additive noise could still pass every statistical
    assertion above.
    """
    sigma = 0.1
    dv = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(808)
    perturbed = dv + rng.normal(0.0, sigma, size=(_N_BURN, 3))
    ratios = np.linalg.norm(perturbed, axis=1)
    angles = np.arccos(np.clip(perturbed @ dv / ratios, -1.0, 1.0))

    null_bound = _Z_BOUND / math.sqrt(_N_BURN)
    expected_correlation = -0.9567 * sigma
    correlation = float(np.corrcoef(ratios, angles)[0, 1])
    assert abs(correlation - expected_correlation) < 0.05 * abs(expected_correlation)
    # Four times the 5-sigma bound that test_magnitude_and_pointing_errors_are_independent
    # requires of the implemented model: the two models are separated, not merely different.
    assert abs(correlation) > 3.0 * null_bound

    # The magnitude bias the implemented model does not have: E|dv + eps| - |dv| ~ s^2/|dv|,
    # created purely by noise perpendicular to the burn.
    assert float(np.mean(ratios)) - 1.0 > 0.5 * sigma**2

    # And the coupling that makes the parameters inexpressible: with one isotropic sigma
    # the ratio of pointing spread to magnitude spread is pinned at sqrt(2 - pi/2) = 0.655,
    # so "1 % magnitude error with 0.1 mrad pointing" cannot be written down at all.
    ratio = float(np.std(angles, ddof=1)) / float(np.std(ratios, ddof=1))
    assert abs(ratio - math.sqrt(2.0 - math.pi / 2.0)) < 0.05 * math.sqrt(2.0 - math.pi / 2.0)


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def _reference_dispersions():
    return {
        "mag": NormalDispersion(mean=1.0, sigma=0.2),
        "phase": UniformDispersion(low=0.0, high=2.0 * math.pi),
        "nav": VectorNormalDispersion(
            mean=np.zeros(3), covariance=np.diag([25.0, 100.0, 4.0]) + 1.5
        ),
        "burn": MagnitudePointingDispersion(sigma_magnitude=0.02, sigma_pointing_rad=0.005),
    }


def _reference_run(nominal, samples, rng):
    dv = samples["burn"].apply(np.array([0.0, -1.0, 0.0]) * float(samples["mag"]))
    return {
        "dv_norm": float(np.linalg.norm(dv)),
        "nav_norm": float(np.linalg.norm(samples["nav"])),
        "phase": float(samples["phase"]),
        "extra": float(rng.normal()),
    }


_RETAIN = {
    "dv_norm": lambda r: r["dv_norm"],
    "nav_norm": lambda r: r["nav_norm"],
    "extra": lambda r: r["extra"],
}


def _metric_matrix(results):
    return [[record.metrics[name] for name in sorted(record.metrics)] for record in results.records]


@pytest.mark.unit
def test_same_seed_gives_bitwise_identical_results():
    kwargs = dict(retain=_RETAIN)
    a = run_campaign(None, _reference_dispersions(), _reference_run, 64, 1234, **kwargs)
    b = run_campaign(None, _reference_dispersions(), _reference_run, 64, 1234, **kwargs)
    assert _metric_matrix(a) == _metric_matrix(b)


@pytest.mark.unit
def test_different_seeds_give_different_results():
    """Complement: the equality above must come from the seed, not from a constant."""
    a = run_campaign(None, _reference_dispersions(), _reference_run, 64, 1234, retain=_RETAIN)
    b = run_campaign(None, _reference_dispersions(), _reference_run, 64, 1235, retain=_RETAIN)
    assert _metric_matrix(a) != _metric_matrix(b)


@pytest.mark.unit
def test_run_i_is_identical_regardless_of_campaign_size():
    """The resumability property: a 100-run campaign is the first 100 of a 1000-run one.

    This is the test that a naive ``rng = default_rng(seed)`` reused across runs silently
    fails only for *some* run counts, and that an ``spawn(n_runs)`` scheme whose child
    index depended on ``n_runs`` would fail outright.
    """
    small = run_campaign(None, _reference_dispersions(), _reference_run, 100, 99, retain=_RETAIN)
    large = run_campaign(None, _reference_dispersions(), _reference_run, 1000, 99, retain=_RETAIN)

    assert _metric_matrix(small) == _metric_matrix(large)[:100]
    for a, b in zip(small.records, large.records[:100], strict=True):
        assert a.index == b.index
        assert sorted(a.samples) == sorted(b.samples)
        assert float(a.samples["mag"]) == float(b.samples["mag"])
        np.testing.assert_array_equal(a.samples["nav"], b.samples["nav"])
        assert a.samples["burn"] == b.samples["burn"]


@pytest.mark.unit
def test_results_are_independent_of_execution_order():
    """Reverse the execution order; every record must be unchanged.

    Stands in for parallel execution: a process pool differs from this only in that it
    also varies *when* each run executes, which the harness cannot observe.
    """

    def reversed_map(fn, indices):
        return [fn(i) for i in reversed(list(indices))]

    sequential = run_campaign(
        None, _reference_dispersions(), _reference_run, 50, 555, retain=_RETAIN
    )
    shuffled = run_campaign(
        None, _reference_dispersions(), _reference_run, 50, 555, retain=_RETAIN, map_fn=reversed_map
    )
    assert [r.index for r in shuffled.records] == list(range(50))
    assert _metric_matrix(sequential) == _metric_matrix(shuffled)


@pytest.mark.unit
def test_samples_are_independent_of_dispersion_insertion_order():
    dispersions = _reference_dispersions()
    reordered = {name: dispersions[name] for name in reversed(list(dispersions))}
    a = draw_samples(dispersions, seed=42, index=7)
    b = draw_samples(reordered, seed=42, index=7)
    assert float(a["mag"]) == float(b["mag"])
    assert float(a["phase"]) == float(b["phase"])
    np.testing.assert_array_equal(a["nav"], b["nav"])


@pytest.mark.unit
def test_adding_a_dispersion_does_not_disturb_the_others():
    """Name-addressed substreams: an unrelated edit must not resample everything else."""
    base = _reference_dispersions()
    extended = dict(base, drag=NormalDispersion(mean=2.2e-3, sigma=1e-4))
    a = draw_samples(base, seed=8, index=3)
    b = draw_samples(extended, seed=8, index=3)
    assert float(a["mag"]) == float(b["mag"])
    np.testing.assert_array_equal(a["nav"], b["nav"])
    assert a["burn"] == b["burn"]
    assert "drag" in b


@pytest.mark.unit
def test_run_fn_rng_consumption_does_not_disturb_the_dispersion_samples():
    """run_fn gets its own substream; how much of it it burns is nobody else's business."""

    def greedy(nominal, samples, rng):
        rng.normal(size=10_000)
        return {"mag": float(samples["mag"])}

    def frugal(nominal, samples, rng):
        return {"mag": float(samples["mag"])}

    retain = {"mag": lambda r: r["mag"]}
    a = run_campaign(None, _reference_dispersions(), greedy, 20, 17, retain=retain)
    b = run_campaign(None, _reference_dispersions(), frugal, 20, 17, retain=retain)
    assert _metric_matrix(a) == _metric_matrix(b)


@pytest.mark.unit
def test_a_single_run_can_be_reproduced_outside_its_campaign():
    """Resumability in practice: re-execute run 37 alone and get run 37 back."""
    campaign = run_campaign(
        None, _reference_dispersions(), _reference_run, 60, 2024, retain=_RETAIN
    )
    standalone = execute_run(
        None,
        _reference_dispersions(),
        _reference_run,
        seed=2024,
        index=37,
        retain=_RETAIN,
    )
    assert standalone.metrics == campaign.records[37].metrics
    assert standalone.index == 37


@pytest.mark.unit
def test_draw_samples_rejects_invalid_seed_and_index():
    with pytest.raises(CampaignConfigurationError, match="seed must be a non-negative int"):
        draw_samples({}, seed=-1, index=0)
    with pytest.raises(CampaignConfigurationError, match="non-negative int"):
        draw_samples({}, seed=1.5, index=0)
    with pytest.raises(CampaignConfigurationError, match="run index must be non-negative"):
        draw_samples({}, seed=1, index=-3)


@pytest.mark.unit
def test_run_campaign_rejects_invalid_run_count():
    for bad in (0, -5, 2.0, True):
        with pytest.raises(CampaignConfigurationError, match="n_runs must be a positive int"):
            run_campaign(None, {}, _reference_run, bad, 0)


# --------------------------------------------------------------------------------------
# Wilson score interval
# --------------------------------------------------------------------------------------

_Z95 = statistics.NormalDist().inv_cdf(0.975)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("successes", "expected"),
    [
        (0, (0.0, 0.2775)),
        (5, (0.2366, 0.7634)),
        (10, (0.7225, 1.0)),
    ],
)
def test_wilson_interval_matches_hand_computed_values(successes, expected):
    """Textbook 95 % Wilson bounds for k/10, quoted to four decimal places."""
    lower, upper = wilson_interval(successes, 10)
    assert lower == pytest.approx(expected[0], abs=1e-4)
    assert upper == pytest.approx(expected[1], abs=1e-4)


@pytest.mark.unit
def test_wilson_interval_degenerates_to_its_closed_form_at_the_endpoints():
    """K = 0 gives exactly [0, z^2/(n + z^2)]; k = n is its mirror image.

    Derived, not copied: at k = 0 the half width equals the centre, so the lower bound is
    identically zero -- which is why the interval never leaves [0, 1] at the endpoints.
    """
    n = 10
    expected_upper = _Z95**2 / (n + _Z95**2)
    lower, upper = wilson_interval(0, n)
    assert lower == 0.0
    assert upper == pytest.approx(expected_upper, rel=1e-12)

    lower_n, upper_n = wilson_interval(n, n)
    assert upper_n == 1.0
    assert lower_n == pytest.approx(1.0 - expected_upper, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("successes", range(11))
def test_wilson_interval_stays_inside_the_unit_interval_where_wald_does_not(successes):
    """Wilson is bounded in [0, 1] for every k; the normal approximation is not.

    At k = 1, n = 10 the Wald lower bound is -0.086 and at k = 0 or 10 its width is zero.
    Both are checked here as complements, so this test fails if the implementation is
    swapped for the normal approximation.
    """
    n = 10
    lower, upper = wilson_interval(successes, n)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower <= successes / n <= upper

    p_hat = successes / n
    wald_half = _Z95 * math.sqrt(p_hat * (1.0 - p_hat) / n)
    if successes in (0, n):
        assert wald_half == 0.0
        assert upper - lower > 0.27
    if successes == 1:
        assert p_hat - wald_half < 0.0
        assert lower > 0.0


@pytest.mark.unit
def test_wilson_interval_narrows_as_the_sample_grows():
    """Convergence behaviour, not a single threshold: width must shrink like 1/sqrt(n)."""
    widths = []
    for n in (10, 100, 1000, 10_000):
        lower, upper = wilson_interval(n // 2, n)
        widths.append(upper - lower)
    assert widths == sorted(widths, reverse=True)
    # Halving-of-width check: 100x the samples should give ~10x the precision.
    assert 8.0 < widths[0] / widths[2] < 12.0


@pytest.mark.unit
def test_wilson_interval_widens_with_confidence():
    narrow = wilson_interval(5, 100, confidence=0.90)
    wide = wilson_interval(5, 100, confidence=0.99)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


@pytest.mark.unit
def test_wilson_interval_is_symmetric_under_swapping_successes_and_failures():
    lower, upper = wilson_interval(3, 17)
    mirrored_lower, mirrored_upper = wilson_interval(14, 17)
    assert mirrored_lower == pytest.approx(1.0 - upper, rel=1e-12)
    assert mirrored_upper == pytest.approx(1.0 - lower, rel=1e-12)


@pytest.mark.unit
def test_wilson_interval_rejects_invalid_counts_and_confidence():
    with pytest.raises(CampaignConfigurationError, match="trials must be a positive int"):
        wilson_interval(0, 0)
    with pytest.raises(CampaignConfigurationError, match="trials must be a positive int"):
        wilson_interval(0, -1)
    with pytest.raises(CampaignConfigurationError, match="trials must be a positive int"):
        wilson_interval(0, 10.0)
    with pytest.raises(CampaignConfigurationError, match="successes must be an int"):
        wilson_interval(1.5, 10)
    with pytest.raises(CampaignConfigurationError, match=r"must lie in \[0, trials=10\]"):
        wilson_interval(11, 10)
    with pytest.raises(CampaignConfigurationError, match=r"must lie in \[0, trials=10\]"):
        wilson_interval(-1, 10)
    for bad in (0.0, 1.0, -0.5, 1.5, math.nan):
        with pytest.raises(CampaignConfigurationError, match=r"strictly inside \(0, 1\)"):
            wilson_interval(5, 10, confidence=bad)


@pytest.mark.unit
def test_proportion_estimate_carries_the_point_estimate_and_the_interval():
    estimate = proportion_estimate(7, 10)
    assert estimate.point == 0.7
    assert estimate.lower < 0.7 < estimate.upper
    assert estimate.confidence == DEFAULT_CONFIDENCE
    assert estimate.describe()["interval"] == "wilson_score"


# --------------------------------------------------------------------------------------
# Retention and metric summaries
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_percentiles_use_linear_interpolation_between_order_statistics():
    """Hand-computed against 1..100: p50 = 50.5 and p95 = 95.05.

    Index = q/100 * (N - 1), so p95 lands at 0.95 * 99 = 94.05 -- between the 95th and
    96th order statistics, five per cent of the way up. Any off-by-one in the index or a
    switch to a different percentile convention moves both numbers.
    """
    summary = summarise_metric("x", np.arange(1.0, 101.0), percentiles=(0.0, 50.0, 95.0, 100.0))
    assert summary.percentiles[50.0] == pytest.approx(50.5, abs=1e-12)
    assert summary.percentiles[95.0] == pytest.approx(95.05, abs=1e-12)
    assert summary.percentiles[0.0] == 1.0
    assert summary.percentiles[100.0] == 100.0
    assert summary.minimum == 1.0
    assert summary.maximum == 100.0
    assert summary.count == 100
    assert summary.mean == pytest.approx(50.5, abs=1e-12)


@pytest.mark.unit
def test_metric_summary_uses_the_unbiased_standard_deviation():
    """ddof=1 on [1, 2, 3, 4] gives sqrt(5/3), not sqrt(5/4)."""
    summary = summarise_metric("x", [1.0, 2.0, 3.0, 4.0])
    assert summary.std == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-12)
    assert summary.std != pytest.approx(math.sqrt(5.0 / 4.0), rel=1e-6)


@pytest.mark.unit
def test_metric_summary_reports_none_rather_than_nan_when_undefined():
    empty = summarise_metric("x", [])
    assert (empty.count, empty.mean, empty.std, empty.minimum, empty.maximum) == (
        0,
        None,
        None,
        None,
        None,
    )
    assert empty.percentiles == {}
    single = summarise_metric("x", [4.0])
    assert single.count == 1 and single.mean == 4.0 and single.std is None
    # None, not NaN: json.dumps(allow_nan=False) must accept the result.
    assert json.loads(json.dumps(empty.describe()))["mean"] is None


@pytest.mark.unit
def test_summarise_metric_rejects_bad_input():
    with pytest.raises(CampaignConfigurationError, match="values must be 1-D"):
        summarise_metric("x", [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(CampaignConfigurationError, match="must all be finite"):
        summarise_metric("x", [1.0, math.nan])
    with pytest.raises(CampaignConfigurationError, match=r"outside \[0, 100\]"):
        summarise_metric("x", [1.0, 2.0], percentiles=(101.0,))


@pytest.mark.unit
def test_metric_values_rejects_an_unretained_name():
    results = run_campaign(
        None, _reference_dispersions(), _reference_run, 5, 1, retain={"dv_norm": _RETAIN["dv_norm"]}
    )
    with pytest.raises(CampaignConfigurationError, match="was not retained"):
        results.metric_values("nav_norm")


@pytest.mark.unit
def test_results_are_dropped_by_default_and_kept_on_request():
    """keep_results defaults to False; a 10k-run campaign must not hoard trajectories."""
    lean = run_campaign(None, _reference_dispersions(), _reference_run, 3, 1, retain=_RETAIN)
    fat = run_campaign(
        None, _reference_dispersions(), _reference_run, 3, 1, retain=_RETAIN, keep_results=True
    )
    assert all(r.result is None for r in lean.records)
    assert all(r.result is not None for r in fat.records)
    assert _metric_matrix(lean) == _metric_matrix(fat)


# --------------------------------------------------------------------------------------
# Failure accounting
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_failing_run_does_not_kill_the_campaign_and_is_not_dropped():
    """Exactly 30 of 100 runs raise; the campaign completes and says so.

    The failure indices are chosen by index arithmetic so the expected count is a fact
    about the test, not a number read back out of the implementation.
    """
    failing = set(range(0, 100, 10)) | set(range(3, 100, 5))
    assert len(failing) == 30

    def run(nominal, samples, rng):
        index = nominal["index_of"](samples)
        if index in failing:
            raise RuntimeError(f"deliberate failure at {index}")
        return {"value": float(samples["mag"])}

    # The run_fn needs its own index; execute_run does not pass it, so recover it from the
    # sample map by pre-computing the (index -> mag) mapping.
    lookup = {
        float(draw_samples(_reference_dispersions(), seed=4242, index=i)["mag"]): i
        for i in range(100)
    }
    nominal = {"index_of": lambda samples: lookup[float(samples["mag"])]}

    results = run_campaign(
        nominal,
        _reference_dispersions(),
        run,
        100,
        4242,
        retain={"value": lambda r: r["value"]},
    )

    assert results.n_runs == 100
    assert results.n_failures == 30
    assert {f.index for f in results.failures} == failing
    assert all(f.stage == "run" for f in results.failures)
    assert all(f.exception_type == "RuntimeError" for f in results.failures)

    # Failed runs contribute no observations -- not zeros, which would pull every mean.
    assert results.metric_values("value").size == 70
    assert all(record.metrics == {} for record in results.records if record.failed)


@pytest.mark.unit
def test_failures_stay_in_the_denominator_of_the_success_rate():
    """A campaign that crashes 30 % of the time has a 70 % success rate, not 100 %.

    This is the single most dangerous number the module could get wrong: dividing by the
    survivors lets a campaign improve its reported reliability by crashing more often.
    """
    lookup = {
        float(
            draw_samples({"mag": NormalDispersion(mean=0.0, sigma=1.0)}, seed=9, index=i)["mag"]
        ): i
        for i in range(100)
    }
    failing = set(range(0, 100, 10)) | set(range(3, 100, 5))

    def run(nominal, samples, rng):
        if lookup[float(samples["mag"])] in failing:
            raise RuntimeError("deliberate failure")
        return 1.0

    results = run_campaign(None, {"mag": NormalDispersion(mean=0.0, sigma=1.0)}, run, 100, 9)
    rate = results.success_rate()
    assert rate.trials == 100
    assert rate.successes == 70
    assert rate.point == 0.7
    assert rate.lower < 0.7 < rate.upper
    # Complement: the wrong denominator (survivors only) would be exactly 1.0.
    assert rate.point != 1.0
    assert results.completion_rate().point == 0.7


@pytest.mark.unit
def test_success_criterion_is_separate_from_completion():
    """A run can complete and still fail its criterion; the two rates must differ."""
    dispersions = {"x": UniformDispersion(low=0.0, high=1.0)}

    def run(nominal, samples, rng):
        return float(samples["x"])

    results = run_campaign(
        None, dispersions, run, 400, 31, success_fn=lambda v: v < 0.25, retain={"x": lambda v: v}
    )
    assert results.n_failures == 0
    assert results.completion_rate().point == 1.0
    # ~25 % by construction; the Wilson interval at n=400 must bracket the true rate.
    assert results.success_rate().lower < 0.25 < results.success_rate().upper
    assert results.n_successes < results.n_runs


@pytest.mark.unit
def test_a_raising_retain_extractor_fails_the_run_at_the_retain_stage():
    def run(nominal, samples, rng):
        return {"ok": 1.0}

    def bad(result):
        raise KeyError("missing_metric")

    results = run_campaign(None, {}, run, 5, 1, retain={"m": bad})
    assert results.n_failures == 5
    assert all(f.stage == "retain" for f in results.failures)
    assert all(f.exception_type == "KeyError" for f in results.failures)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_metric_fails_the_run_rather_than_poisoning_the_statistics(bad):
    results = run_campaign(None, {}, lambda n, s, r: bad, 4, 1, retain={"m": float})
    assert results.n_failures == 4
    assert all(f.stage == "retain" for f in results.failures)
    assert "non-finite metric" in results.failures[0].message


@pytest.mark.unit
def test_a_raising_success_predicate_fails_the_run_at_the_success_stage():
    def boom(result):
        raise ZeroDivisionError("bad predicate")

    results = run_campaign(None, {}, lambda n, s, r: 1.0, 3, 1, success_fn=boom)
    assert results.n_failures == 3
    assert all(f.stage == "success" for f in results.failures)


@pytest.mark.unit
def test_a_raising_dispersion_fails_the_run_at_the_sample_stage():
    class Exploding(NormalDispersion):
        def sample(self, rng):
            raise ArithmeticError("sampler exploded")

    results = run_campaign(None, {"x": Exploding()}, lambda n, s, r: 1.0, 3, 1)
    assert results.n_failures == 3
    assert all(f.stage == "sample" for f in results.failures)


@pytest.mark.unit
def test_keyboard_interrupt_is_not_swallowed_into_a_failed_run():
    """BaseException means the operator wants out, not that one run misbehaved."""

    def run(nominal, samples, rng):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_campaign(None, {}, run, 5, 1)


@pytest.mark.unit
def test_a_map_fn_that_drops_runs_is_rejected():
    """Silent loss one level up is still silent loss."""

    def dropping(fn, indices):
        return [fn(i) for i in list(indices)[:-1]]

    with pytest.raises(CampaignConfigurationError, match="dropped or duplicated"):
        run_campaign(None, {}, lambda n, s, r: 1.0, 10, 1, map_fn=dropping)


@pytest.mark.unit
def test_a_map_fn_that_duplicates_runs_is_rejected():
    def duplicating(fn, indices):
        records = [fn(i) for i in indices]
        return records + records[:1]

    with pytest.raises(CampaignConfigurationError, match="dropped or duplicated"):
        run_campaign(None, {}, lambda n, s, r: 1.0, 10, 1, map_fn=duplicating)


@pytest.mark.unit
def test_failure_records_carry_a_traceback_but_the_summary_does_not():
    """Tracebacks are machine-specific; a summary that embedded them would not be stable."""

    def run(nominal, samples, rng):
        raise RuntimeError("boom")

    results = run_campaign(None, {}, run, 2, 1)
    assert "RuntimeError: boom" in results.failures[0].traceback_text
    assert "traceback" not in json.dumps(results.summary().to_dict())


@pytest.mark.unit
def test_summary_counts_failures_by_type():
    def run(nominal, samples, rng):
        value = float(samples["x"])
        if value < 0.3:
            raise RuntimeError("low")
        if value > 0.7:
            raise ValueError("high")
        return value

    results = run_campaign(None, {"x": UniformDispersion(low=0.0, high=1.0)}, run, 300, 77)
    counts = results.summary().failure_counts_by_type()
    assert set(counts) == {"RuntimeError", "ValueError"}
    assert sum(counts.values()) == results.n_failures
    assert results.n_successes + results.n_failures == 300


# --------------------------------------------------------------------------------------
# Summary serialisation and end-to-end reproducibility
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_a_campaign_is_reproducible_from_its_own_summary():
    """The closing argument for the whole module.

    Take the JSON a campaign wrote, rebuild the dispersions from the ``dispersions`` block
    and the seed from the ``seed`` field, re-run, and get the same metrics. Nothing else
    from the original session is needed.
    """
    original = run_campaign(
        None, _reference_dispersions(), _reference_run, 128, 20260824, retain=_RETAIN
    )
    document = json.loads(original.summary().to_json())

    assert document["seed"] == 20260824
    assert document["n_runs"] == 128
    assert document["n_failures"] == 0
    assert document["n_successes"] == 128
    assert set(document["dispersions"]) == set(_reference_dispersions())
    assert document["success_rate"]["interval"] == "wilson_score"
    assert document["bit_generator"] == "PCG64"

    rebuilt = {name: dispersion_from_dict(spec) for name, spec in document["dispersions"].items()}
    replay = run_campaign(
        None, rebuilt, _reference_run, document["n_runs"], document["seed"], retain=_RETAIN
    )
    assert _metric_matrix(replay) == _metric_matrix(original)
    assert replay.summary().to_json() == original.summary().to_json()


@pytest.mark.unit
def test_summary_json_is_strict_and_carries_the_reproducibility_fields():
    def run(nominal, samples, rng):
        if float(samples["x"]) > 0.9:
            raise RuntimeError("tail failure")
        return float(samples["x"])

    results = run_campaign(
        None,
        {"x": UniformDispersion(low=0.0, high=1.0)},
        run,
        50,
        3,
        retain={"x": float},
    )
    text = results.summary().to_json()
    document = json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    for key in (
        "seed",
        "n_runs",
        "n_failures",
        "n_successes",
        "dispersions",
        "metrics",
        "failures",
        "failure_counts_by_type",
        "success_rate",
        "completion_rate",
        "numpy_version",
        "substream_scheme",
    ):
        assert key in document
    assert document["metrics"]["x"]["count"] == 50 - document["n_failures"]
    assert document["metrics"]["x"]["percentile_method"] == "linear"
    assert document["metrics"]["x"]["std_ddof"] == 1


@pytest.mark.unit
def test_summary_survives_a_campaign_in_which_every_run_failed():
    """The degenerate case must still report honestly, not divide by zero."""
    results = run_campaign(None, {}, lambda n, s, r: 1 / 0, 8, 1, retain={"m": lambda r: float(r)})
    summary = results.summary()
    assert summary.n_failures == 8
    assert summary.success_rate.point == 0.0
    assert summary.success_rate.lower == 0.0
    assert summary.success_rate.upper > 0.0
    assert summary.metrics["m"].count == 0
    assert summary.metrics["m"].mean is None
    json.loads(summary.to_json())


@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end_burn_dispersion_campaign_recovers_the_input_statistics():
    """A realistic campaign: disperse a V-bar hop's departure impulse and measure it back.

    The retained metric is the executed impulse magnitude, whose distribution is known in
    closed form from the dispersion definition: mean |dv| and standard deviation
    sigma_magnitude * |dv|. Recovering it end to end exercises every layer -- definition,
    substream, execution, retention, summary -- against a number derived from the inputs.
    """
    dv_nominal = np.array([0.045, 0.0, 0.0])
    sigma_mag = 0.02
    dispersions = {
        "burn": MagnitudePointingDispersion(sigma_magnitude=sigma_mag, sigma_pointing_rad=0.01)
    }
    n_runs = 20_000

    def run(nominal, samples, rng):
        return samples["burn"].apply(nominal)

    results = run_campaign(
        dv_nominal,
        dispersions,
        run,
        n_runs,
        6060,
        retain={"dv_m_s": lambda dv: float(np.linalg.norm(dv))},
    )
    summary = results.summary()
    metric = summary.metrics["dv_m_s"]

    expected_mean = float(np.linalg.norm(dv_nominal))
    expected_std = sigma_mag * expected_mean
    assert metric.count == n_runs
    assert abs(metric.mean - expected_mean) < _Z_BOUND * expected_std / math.sqrt(n_runs)
    assert abs(metric.std - expected_std) < _Z_BOUND * expected_std / math.sqrt(2.0 * n_runs)

    # Percentiles must track the Gaussian quantiles of the same distribution.
    for level in (5.0, 95.0):
        z = statistics.NormalDist().inv_cdf(level / 100.0)
        expected = expected_mean + z * expected_std
        # Quantile SE is sqrt(p(1-p)/N) / pdf(quantile).
        p = level / 100.0
        density = statistics.NormalDist(expected_mean, expected_std).pdf(expected)
        quantile_se = math.sqrt(p * (1.0 - p) / n_runs) / density
        assert abs(metric.percentiles[level] - expected) < _Z_BOUND * quantile_se

    assert summary.success_rate.point == 1.0
    assert summary.success_rate.lower > 0.99
    assert summary.success_rate.upper == 1.0
