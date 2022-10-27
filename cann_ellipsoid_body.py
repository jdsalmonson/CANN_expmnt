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
    
vels = ImpulseDict(
    #N = 2,
    #id = {0: ((5.0, 15.),), 1: ((19., 29.),)}
    N = 4,
    id = {1: ((5., 10.),), 0: ((10., 15.),), 2: ((19.,24.),), 3: ((24., 29.),)}
    )


tm0 = 5.0
on_tm = 10.0  # 17.5
mid_tm = 4.0
end_tm = 10.0  # 4.0

tm_range = [0, tm0 + 2 * on_tm + mid_tm + end_tm]
ang_ary = [0.0] * 10 # 16
ang_ary[3] = 1.0


def vp(t, t0=tm0, t1=tm0 + on_tm) -> float:
    # v+ velocity input signal
    return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)


def vm(t, t0=tm0 + on_tm + mid_tm, t1=tm0 + 2 * on_tm + mid_tm) -> float:
    # v- velocity input signal
    return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)


def direct_impulse(t: float, t0: float, t1: float) -> float:
    # apply direct impulse to neuron over time range [t0:t1]
    return heaviside(t - t0, 0.5) * heaviside(t1 - t, 0.5)


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
    
    #vel_term = -(vp(t) + vm(t)) * w0 + vm(t) * w_1 + vp(t) * w1
    vel_term = -(vp + vm) * w0 + vm * w_1 + vp * w1

    #impulse_term = np.zeros_like(w)
    #impulse_term[8] += direct_impulse(t, 1., 5.) #28.0, 32)  # 28.0, 30.0)

    di = direct_impulse(t, 1., 2.) #5.)
    impulse_term = -np.copy(w) * di
    impulse_term[8] = di * (1.-w[8])
    #if impulse_term[8] > 0.0:
    #    print(t, impulse_term, inhibition_term, w_mx, fac)

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

outer_scat_ring = ax2.scatter(
    np.radians(theta),
    1.6*r,
    c=0.*r,
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
    
    scat_ring.set_array(sol.y[:, i])


# Shave off last few frames to make cycle close.
a = animation.FuncAnimation(
    fig, update, frames=sol.y.T.shape[0], interval=25, repeat=True
)

make_html = False
if make_html:
    # Make HTML output: -------------------
    from matplotlib.animation import HTMLWriter
    import matplotlib

    # Increase size limit for html file:

    matplotlib.rcParams["animation.embed_limit"] = 2**32  # 128
    a.save("cann_cycle.html", writer=HTMLWriter(embed_frames=True))

    # To open file in web browser:
    # > xdg-open cann_cycle.html
    # --------------------------------------
else:
    # pass
    plt.show()

fig2 = plt.figure(1)
plt.plot(sol.t, sol.y.T)
plt.show()
