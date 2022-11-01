from typing import List, Dict, Tuple
import numpy as np
from numpy import heaviside

from dataclasses import dataclass, field

# from functools import partial
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('dark_background')

@dataclass
class ImpulseDict:
    """Simple class to record time spans that a range of given impulse channels will be active.
    This class is used both for impulses in the velocity array of the protocerebral bridge and the direct impulses to directional neurons.
    Args:
      N (int) number of channels
      id (dict) the impulse dict.  Each key (channel) has an array of time pairs: ((time0_on, time0_off), (time1_on, time1_off),...)
    """
    
    # number of channels:
    N: int = 10
    # impulse dict:
    id: Dict[int, Tuple[Tuple[float]]] = field(default_factory=dict)

    def get_impulse(self, t: float) -> int:
        """find impulse pair containing 't' and return its key"""
        for k, v in self.id.items():
            for vpair in v:
                if vpair[0] <= t and t < vpair[1]:
                    return k

# velocity drive history from protocerebral bridge:       
vels = ImpulseDict(
    # 2 node protocerebral bridge:
    #N = 2,
    #id = {0: ((5.0, 15.),), 1: ((19., 29.),)}

    # 4 node protocerebral bridge:
    N = 4,
    id = {1: ((5., 11.),), 0: ((11., 15.),), 2: ((19.,25.),), 3: ((25., 29.),)}
    )

# Number of angular elements to the ellipsoid body:
N_theta = 10 # 16

# Direct impulse
dir_impl = ImpulseDict(
    N = N_theta,
    id = {8: ((1., 2.),), 4: ((32.,34.),), 1: ((36., 38.),), 3: ((41., 43.),)}
    )

# simulation domain:
ang_ary = [0.0] * N_theta
ang_ary[3] = 1.0
tm_range = [0, 48]


def w_p(t: float, y: List[float]) -> List[float]:
    """weight value ode"""

    w = np.array(y)

    # This w_mx actuall makes the swap a lot smoother:
    w_mx = max(w.max(), 1.0)
    wdiff = np.subtract.outer(w, w)
    fac = w * (w_mx - w)
    fac_mat = np.outer(fac, fac)

    # inhibition_mat_ij = (w_i - w_j) * w_i * (1 - w_i) * w_j * (1 - w_j):
    inhibition_mat = np.multiply(wdiff, fac_mat)
    inhibition_term = inhibition_mat.sum(axis=0)

    # proximal velocity stimulus:
    w0 = np.power(np.abs(np.array(y)), 1.0)
    w_1 = np.roll(w0, -1)
    w1 = np.roll(w0, 1)

    v = np.zeros(vels.N)
    n2 = vels.N/2.
    if (idx := vels.get_impulse(t)) is not None:
        val = float(idx) - n2
        # add one if in upper half:
        if idx >= int(n2):
            val += 1
        v[idx] = abs(val)

    # probably can eliminate this sum:
    vm = v[:int(n2)].sum()
    vp = v[int(n2):].sum()
    
    vel_term = -(vp + vm) * w0 + vm * w_1 + vp * w1

    # Direct impulse stimulus:
    if (idx := dir_impl.get_impulse(t)) is not None:        
        impulse_term = -np.copy(w)
        impulse_term[idx] = (1.-w[idx])
    else:
        impulse_term = np.zeros_like(w)
        

    return 1.0 * vel_term - 20.0 * inhibition_term + impulse_term


sol = solve_ivp(
    w_p,
    tm_range,
    ang_ary,
    dense_output=True,
)

# Make figure:

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={"height_ratios": [1, 3]}
)

fig.suptitle("Ring Attractor")

theta = np.linspace(0.0, 360.0, sol.y.T.shape[-1], endpoint=False)
r = np.ones_like(theta)


# fig = plt.figure()
n = vels.N
ax1 = plt.subplot(211)
ax1.grid(False)
ax1.set_xlim([-n/2., n/2.])
ax1.set_title("Protocerebral bridge")
# ax1.set_ylim([-2, 2])
ax1.set_xticks([])
ax1.set_yticks([])
#vel_cols = np.array([vm(sol.t[0]), vp(sol.t[0])])
vel_cols = np.zeros(vels.N)
if (idx := vels.get_impulse(sol.t[0])) is not None:
    vel_cols[idx] = 1.0
# Need to set vmin/vmax range to include full possible range, in case initial values don't span this:
cols = list(np.arange(vels.N) + 0.5 - vels.N/2.)
rows = [0] * len(cols)
vel_row = ax1.scatter(
    cols, rows, c=vel_cols, s=2_000, cmap="magma", edgecolors="b", vmin=0.0, vmax=1.0
)


ax2 = plt.subplot(212, projection="polar")
ax2.set_title("Ellipsoid body")
ax2.set_theta_zero_location("N")  # theta = 0 is up
ax2.set_theta_direction(-1)  # clockwise
ax2.set_rlim(0,2.5)
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])


scat_ring = ax2.scatter(
    np.radians(theta),
    r,
    c=sol.y[:, 0],
    edgecolors="r",
    s=500,
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
)

impulse_cols = np.zeros_like(theta)
if (idx := dir_impl.get_impulse(sol.t[0])) is not None:        
    impulse_cols[idx] = 1.0

outer_scat_ring = ax2.scatter(
    np.radians(theta),
    1.6*r,
    c=impulse_cols,
    edgecolors="b",
    s=200,
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
)

def update(i):
    # Set colors..
    vel_cols = np.zeros(vels.N)
    if (idx := vels.get_impulse(sol.t[i])) is not None:
        vel_cols[idx] = 1.0
    vel_row.set_array(vel_cols)
    #vel_row.set_array(np.array([vm(sol.t[i]), vp(sol.t[i])]))

    # Set ellipsoid body colors:
    scat_ring.set_array(sol.y[:, i])

    # Set impulse ring colors:
    impulse_cols = np.zeros_like(theta)
    if (idx := dir_impl.get_impulse(sol.t[i])) is not None:        
        impulse_cols[idx] = 1.0
    outer_scat_ring.set_array(impulse_cols)

# Shave off last few frames to make cycle close.
a = animation.FuncAnimation(
    fig, update, frames=sol.y.T.shape[0], interval=35, repeat=True
)

make_html = False
if make_html:
    # Make HTML output: -------------------
    from matplotlib.animation import HTMLWriter
    import matplotlib

    # Increase size limit for html file:

    matplotlib.rcParams["animation.embed_limit"] = 2**32  # 128
    a.save("cann_ellipsoid_body.html", writer=HTMLWriter(embed_frames=True))

    # To open file in web browser:
    # > xdg-open cann_cycle.html
    # --------------------------------------

    fig2 = plt.figure(1)
    plt.clf()
    plt.plot(sol.t, sol.y.T)
    plt.xlabel("time")
    plt.ylabel("fire rate")
    plt.savefig("fire_rates_per_time.png")

else:
    plt.show()

    fig2 = plt.figure(1)
    plt.plot(sol.t, sol.y.T)
    plt.xlabel("time")
    plt.ylabel("fire rate")
    plt.show()
