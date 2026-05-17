import jax
import jax.numpy as jnp
from functools import partial

from .potentials import *
from .utils import get_rj_vj_R, prepare_params
from .methods import create_ic_particle_spray_Fardal2015, create_ic_particle_spray_Chen2025
from .integrants import integrate_leapfrog_final, integrate_leapfrog_traj, combined_integrate_leapfrog_final, precompute_prog_trajectories

@partial(jax.jit, static_argnames=('type_host', 'type_sat', 'n_particles', 'n_steps', 'unroll',
                                    'type_method'))
def generate_stream(xv_f,
                    type_host, params_host,
                    type_sat, params_sat,
                    start_time, alpha, n_steps,
                    n_particles,
                    unroll,
                    end_time=0,
                    type_method='Fardal2015',
                    m_f_sat=0,
                    tail=0, seed=111):

    # Lookup table for single-component potentials (used by both host dispatch and Composite)
    _SINGLE_ACC = {
        'PointMass':     PointMass_acceleration,
        'Isochrone':     Isochrone_acceleration,
        'Plummer':       Plummer_acceleration,
        'NFW':           NFW_acceleration,
        'MiyamotoNagai': MiyamotoNagai_acceleration,
        'Logarithmic':   Logarithmic_acceleration,
        'Bar_potential': Bar_acceleration,
        'Hernquist':     Hernquist_acceleration,
        'ExpDisk':       ExpDisk_acceleration,
        'DiskNFW':       NFW_MiyamotoNagai_acceleration,
    }

    # Host acceleration and param preparation.
    # Composite syntax: type_host = 'Composite:TYPE1,TYPE2,...'
    #   params_host = list of component param dicts, in the same order as the type string.
    #   Duplicate types are allowed (e.g. 'Composite:NFW,NFW' with two NFW dicts).
    if type_host in _SINGLE_ACC:
        acc_host    = _SINGLE_ACC[type_host]
        params_host = prepare_params(params_host)
    elif type_host.startswith('Composite:'):
        _comp_types = tuple(type_host.split(':')[1].split(','))
        _comp_fns   = tuple(_SINGLE_ACC[t] for t in _comp_types)
        def acc_host(x, y, z, params):
            return sum(fn(x, y, z, params[i]) for i, fn in enumerate(_comp_fns))
        params_host = [prepare_params(p) for p in params_host]
    else:
        raise NotImplementedError(f"Host potential '{type_host}' not implemented.")

    # Satellite acceleration and param preparation
    _SAT_ACC = {
        'PointMass': PointMass_acceleration,
        'Isochrone': Isochrone_acceleration,
        'Plummer':   Plummer_acceleration,
        'NFW':       NFW_acceleration,
    }
    if type_sat in _SAT_ACC:
        acc_sat    = _SAT_ACC[type_sat]
        params_sat = prepare_params(params_sat)
    else:
        raise NotImplementedError(f"Satellite potential '{type_sat}' not implemented.")

    # Ensure float32 throughout (consistent on both CPU and GPU)
    xv_f = jnp.asarray(xv_f, dtype=jnp.float32)

    # Define time step (dt) from total lookback time and number of steps
    dt = start_time/n_steps

    # Get initial position of the prog by integrating backwards
    _, xv_i = integrate_leapfrog_final(xv_f, params_host, acc_host, n_steps, dt=-dt, unroll=unroll)

    # Get the orbit of the prog by integrating forwards dt*alpha
    t_sat, xv_sat = integrate_leapfrog_traj(xv_i, params_host, acc_host, n_steps, dt = dt*alpha, unroll=unroll)

    # Compute radial Hessian component d²Φ/dr² via HVP — 3x faster than the full 3x3 Hessian.
    # acc_host = -grad(Phi), so grad(Phi) = -acc_host.
    # HVP: ∇²Φ @ pos = d/dε[-acc_host(pos + ε·pos)]|_{ε=0}  (forward-over-reverse AD)
    def _d2phi_radial(pos):
        _, hvp = jax.jvp(lambda p: -acc_host(p[0], p[1], p[2], params_host), (pos,), (pos,))
        return jnp.dot(pos, hvp) / (jnp.dot(pos, pos) + EPSILON)

    d2phi_dr2 = jax.vmap(_d2phi_radial)(xv_sat[:, :3])

    # Satellite mass evolution: decreases linearly from logM at lookback start_time
    # to m_f_sat at lookback end_time, then stays constant (end_time=0 recovers old behaviour).
    # t_sat runs from 0 → start_time*alpha; physical time elapsed = t_sat/alpha.
    t_loss_window = start_time - end_time
    loss_frac = jnp.clip(t_sat / (alpha * t_loss_window), 0.0, 1.0)
    m_sat = params_sat['logM'] + loss_frac * (m_f_sat - params_sat['logM'])
    dm    = (m_sat - m_f_sat) / n_steps  # zero automatically after end_time

    # Create initial conditions for the particle spray
    if type_method == 'Fardal2015':
        # Get the RJ and VJ matrices along the orbit
        rj, vj, R = get_rj_vj_R(d2phi_dr2, xv_sat, m_sat)

        # Get the particle spray initial conditions
        ic_particle_spray = create_ic_particle_spray_Fardal2015(xv_sat, rj, vj, R,
                                                    n_particles=n_particles, n_steps=len(xv_sat), tail=tail, seed=seed)
    elif type_method == 'Chen2025':
        # Get the Jacobi radius and velocity matrices along the orbit
        rj, vj, R = get_rj_vj_R(d2phi_dr2, xv_sat, m_sat)

        # Get the particle spray initial conditions
        ic_particle_spray = create_ic_particle_spray_Chen2025(xv_sat, rj, vj, R, m_sat,
                                                    n_particles=n_particles, n_steps=len(xv_sat), tail=tail, seed=seed)
    else:
        raise NotImplementedError(f"Method {type_method} not implemented.")

    # Precompute progenitor trajectories once for each unique orbit release step,
    # then look them up per particle instead of re-integrating n_particles times.
    dt_arr = (start_time - t_sat) * alpha / n_steps
    prog_pos, prog_M = precompute_prog_trajectories(
        xv_sat, params_host, acc_host, n_steps, dt_arr, m_sat, dm, unroll)

    # Integrate the particle spray
    index = jnp.repeat(jnp.arange(0, n_steps+1, 1), n_particles // (n_steps+1))
    xv_stream, xhi_stream = jax.vmap(combined_integrate_leapfrog_final,
                                    in_axes=(0, 0, None, None, None, None, None, None, None, None, None)) \
                                    (index, ic_particle_spray, params_host,
                                    prog_pos, prog_M,
                                    params_sat,
                                    acc_host, acc_sat,
                                    n_steps,
                                    dt_arr,
                                    unroll)

    return t_sat, xv_sat, xv_stream, xhi_stream