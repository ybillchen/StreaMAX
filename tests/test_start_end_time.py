"""
Tests for the start_time / end_time API and Composite potential in generate_stream().

Run tests:  pytest tests/test_start_end_time.py -v
Run plot:   python tests/test_start_end_time.py
"""

import pytest
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import StreaMAX

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TYPE_HOST = 'NFW'
PARAMS_HOST = {
    'logM': 12.0, 'Rs': 25.0,
    'a': 1.0, 'b': 1.0, 'c': 1.0,
    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
}
TYPE_SAT = 'Plummer'
PARAMS_SAT = {'logM': 8.0, 'Rs': 1.0,
              'x_origin': 100.0, 'y_origin': 0.0, 'z_origin': 0.0}
XV_F = jnp.array([100.0, 0.0, 0.0, 0.0, 200.0, 0.0])

START_TIME  = 8.0
N_PARTICLES = 1000   # small for fast unit tests
N_STEPS     = 99     # n_steps+1 must divide n_particles

# Composite: params_host is a list ordered to match the type string.
# Duplicate types are allowed — each list entry is an independent component.

# NFW halo + Plummer bulge
TYPE_HOST_COMP = 'Composite:NFW,Plummer'
PARAMS_HOST_COMP = [
    {   # NFW halo
        'logM': 12.0, 'Rs': 25.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    },
    {   # Plummer bulge
        'logM': 10.0, 'Rs': 2.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    },
]

# Two NFW components (same type, different parameters)
TYPE_HOST_COMP_DUP = 'Composite:NFW,NFW'
PARAMS_HOST_COMP_DUP = [
    {   # outer NFW halo
        'logM': 12.0, 'Rs': 25.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    },
    {   # inner NFW (e.g. bulge-scale cusp)
        'logM': 10.0, 'Rs': 1.0,
        'a': 1.0, 'b': 1.0, 'c': 1.0,
        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0,
    },
]


def _run(end_time=0.0, type_host=TYPE_HOST, params_host=PARAMS_HOST,
         n_particles=N_PARTICLES):
    return StreaMAX.generate_stream(
        XV_F,
        type_host, params_host,
        TYPE_SAT, PARAMS_SAT,
        START_TIME, 1.0, N_STEPS,
        n_particles, False,
        end_time=end_time,
        m_f_sat=0.0,
    )


# ---------------------------------------------------------------------------
# end_time tests
# ---------------------------------------------------------------------------

def test_output_shapes():
    t_sat, xv_sat, xv_stream, xhi_stream = _run(end_time=0.0)
    assert t_sat.shape     == (N_STEPS + 1,)
    assert xv_sat.shape    == (N_STEPS + 1, 6)
    assert xv_stream.shape == (N_PARTICLES, 6)
    assert xhi_stream.shape == (N_PARTICLES,)


def test_end_time_zero_finite():
    # end_time=0 reproduces old behaviour; output must be finite
    _, _, xv_stream, _ = _run(end_time=0.0)
    assert jnp.isfinite(xv_stream).all()


@pytest.mark.parametrize("end_time", [2.0, 4.0, 6.0])
def test_positive_end_time_finite(end_time):
    _, _, xv_stream, _ = _run(end_time=end_time)
    assert jnp.isfinite(xv_stream).all(), \
        f"Non-finite values with end_time={end_time}"


def test_progenitor_orbit_unchanged():
    """Progenitor orbit (host only) must be identical regardless of end_time."""
    _, xv_sat_0, _, _ = _run(end_time=0.0)
    _, xv_sat_4, _, _ = _run(end_time=4.0)
    assert jnp.allclose(xv_sat_0, xv_sat_4, atol=1e-5)


# ---------------------------------------------------------------------------
# Composite potential tests
# ---------------------------------------------------------------------------

def test_composite_runs():
    """Composite:NFW,Plummer must produce finite output."""
    _, _, xv_stream, _ = _run(type_host=TYPE_HOST_COMP,
                               params_host=PARAMS_HOST_COMP)
    assert jnp.isfinite(xv_stream).all()


def test_composite_output_shapes():
    t_sat, xv_sat, xv_stream, xhi_stream = _run(type_host=TYPE_HOST_COMP,
                                                  params_host=PARAMS_HOST_COMP)
    assert t_sat.shape     == (N_STEPS + 1,)
    assert xv_sat.shape    == (N_STEPS + 1, 6)
    assert xv_stream.shape == (N_PARTICLES, 6)
    assert xhi_stream.shape == (N_PARTICLES,)


def test_composite_differs_from_single():
    """Adding a bulge component must change the stream morphology."""
    _, _, xv_nfw,  _ = _run()
    _, _, xv_comp, _ = _run(type_host=TYPE_HOST_COMP,
                             params_host=PARAMS_HOST_COMP)
    assert not jnp.allclose(xv_nfw, xv_comp, atol=1e-3)


def test_composite_duplicate_types():
    """Two components of the same type (NFW+NFW) must run and produce finite output."""
    _, _, xv_stream, _ = _run(type_host=TYPE_HOST_COMP_DUP,
                               params_host=PARAMS_HOST_COMP_DUP)
    assert jnp.isfinite(xv_stream).all()


def test_composite_duplicate_differs_from_single():
    """NFW+NFW composite must differ from a single NFW with the same outer params."""
    _, _, xv_single, _ = _run()
    _, _, xv_dup,    _ = _run(type_host=TYPE_HOST_COMP_DUP,
                               params_host=PARAMS_HOST_COMP_DUP)
    assert not jnp.allclose(xv_single, xv_dup, atol=1e-3)


# ---------------------------------------------------------------------------
# Comparison plot  (python tests/test_start_end_time.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    end_times = [0.0, 2.0, 4.0, 6.0]
    n_plot    = 10_000

    fig, axes = plt.subplots(1, len(end_times), figsize=(20, 5),
                             sharex=True, sharey=True)

    for ax, end_time in zip(axes, end_times):
        t_sat, xv_sat, xv_stream, xhi_stream = _run(end_time=end_time,
                                                     n_particles=n_plot)
        vmax = float(jnp.abs(xhi_stream).max())
        ax.scatter(xv_stream[:, 0], xv_stream[:, 1],
                   c=xhi_stream, cmap='seismic',
                   vmin=-vmax, vmax=vmax, s=1, alpha=0.8)
        ax.scatter(xv_sat[-1, 0], xv_sat[-1, 1], c='black', s=50, zorder=5)
        ax.set_title(f'end_time = {end_time} Gyr')
        ax.set_xlabel('X [kpc]')
        ax.set_aspect('equal')

    axes[0].set_ylabel('Y [kpc]')
    fig.suptitle(f'start_time = {START_TIME} Gyr, varying end_time', y=1.02)
    plt.tight_layout()

    out = 'tests/end_time_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
