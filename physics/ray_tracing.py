import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize_scalar

Rs    = 1.447       # kpc
rho_s = 3.11e-3     # kpc^-2
r_obs = 50.0        # kpc


def Phi(r):
    return -(4.0 * np.pi * rho_s * Rs**3 * (np.log(r + Rs) - np.log(Rs))) / r


def b_shape(r, r0):
    return (r0
            + rho_s * Rs**3 * (Rs / (r + Rs) + np.log(r + Rs))
            - rho_s * Rs**3 * (Rs / (Rs + r0) + np.log(Rs + r0)))


def omega(r, J):
    return 2.0 * J / r**3


def find_photon_sphere(r0, J=0.0, sign=0):
    r_arr   = np.linspace(r0 + 0.001, 10.0, 8000)
    b_tilde = r_arr * np.exp(-Phi(r_arr))
    r_init  = r_arr[np.argmin(b_tilde)]

    res = minimize_scalar(
        lambda r: r * np.exp(-Phi(r)),
        bounds=(max(r0 + 0.001, r_init - 0.3), min(10.0, r_init + 0.3)),
        method='bounded'
    )
    r_ph  = res.x
    b_ph0 = r_ph * np.exp(-Phi(r_ph))

    if J == 0.0 or sign == 0:
        return r_ph, b_ph0

    correction = omega(r_ph, J) * r_ph * np.exp(-Phi(r_ph))
    return r_ph, float(b_ph0 * (1.0 + sign * correction))


def _geodesic_rhs(lam, y, b_impact, J, r0):
    r, phi = y
    if r <= r0 + 1e-7:
        return [0.0, 0.0]
    om   = omega(r, J)
    bval = b_shape(r, r0)
    Ph   = Phi(r)
    arg  = np.exp(-2.0 * Ph) - ((b_impact - om * r**2) / r**2)**2
    drdl = -np.sqrt(max(0.0, (1.0 - bval / r) * arg))
    return [drdl, (b_impact - om * r**2) / r**2]


def compute_intensity(b_impact, J, r0, b_ph_cut):
    if abs(b_impact) < b_ph_cut * 0.98:
        return 0.0

    def rhs(lam, y):
        return _geodesic_rhs(lam, y, b_impact, J, r0)

    def ev_throat(lam, y):
        return y[0] - r0 - 1e-7
    ev_throat.terminal = True

    def ev_turning(lam, y):
        r  = y[0]
        om = omega(r, J)
        Ph = Phi(r)
        return np.exp(-2.0 * Ph) - ((b_impact - om * r**2) / r**2)**2
    ev_turning.terminal  = True
    ev_turning.direction = -1

    sol = solve_ivp(rhs, [0, 400], [r_obs, 0.0],
                    events=[ev_throat, ev_turning],
                    rtol=1e-5, atol=1e-8, method='RK45')

    if len(sol.t) < 3:
        return 0.0

    r, phi = sol.y[0], sol.y[1]
    u      = 1.0 / r
    bv     = np.array([b_shape(ri, r0) for ri in r])
    Ph     = Phi(r)

    f_u   = (1.0 - bv * u) * (np.exp(-2.0 * Ph) / max(b_impact**2, 1e-12) - u**2)
    up    = np.sqrt(np.maximum(0.0, f_u))
    up_s  = np.maximum(up, 1e-14)
    emiss = 1.0 / r**2

    intgd = (np.exp(3.0 * Ph)
             * np.sqrt(1.0 / np.maximum(1.0 - bv * u, 1e-12) + u**2 / up_s**2)
             * emiss)

    mult = 2.0 if (len(sol.t_events) > 1 and len(sol.t_events[1]) > 0) else 1.0
    return abs(trapezoid(intgd * up_s, phi)) * mult if len(phi) > 1 else 0.0


def compute_intensity_profile(r0, J, b_grid):
    _, b_ph = find_photon_sphere(r0, J, sign=0)
    I = np.array([compute_intensity(b, J, r0, b_ph) for b in b_grid])
    mx = I.max()
    return (I / mx if mx > 0 else I), b_ph
