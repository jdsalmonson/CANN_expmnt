from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp

from impulsedict import ImpulseDict


class CANNKinematic:
    """Kinematic Continuous Attractor Neural Network (CANN) class

    Note: remember to initially set v = 0 index

    Args:
        Nacc (int): number of acceleration neurons in the network
        Nvel (int): number of velocity neurons in the network
        Nx (int): number of position neurons in the network
        acc_impl (ImpulseDict): direct acceleration impulses
        vel_impl (ImpulseDict): direct velocity impulses
        dir_impl (ImpulseDict): direct position impulses
    """

    def __init__(
        self,
        Nacc: int = 4,
        Nvel: int = 5,
        Nx: int = 10,
        acc_impl: ImpulseDict = None,
        vel_impl: ImpulseDict = None,
        dir_impl: ImpulseDict = None,
    ):
        self.Nacc = Nacc
        self.Nvel = Nvel
        self.Nx = Nx
        self.acc_impl = acc_impl
        self.vel_impl = vel_impl
        self.dir_impl = dir_impl

        assert self.Nacc % 2 == 0, "Nacc must be even (no zero acceleration neuron)"
        assert self.Nvel % 2 == 1, "Nvel must be odd (centered around zero)"

        # eliminate zero from even array (e.g. if Nacc = 4, a_stencil = [-2, -1, 1, 2]):
        self.n2a = self.Nacc // 2
        self.a_stencil = np.arange(self.Nacc)
        self.a_stencil[self.n2a :] += 1
        self.a_stencil -= self.n2a

        self.n2v = (self.Nvel - 1) // 2
        self.v_stencil = np.arange(self.Nvel) - self.n2v

    def w_p(self, t: float, w: np.ndarray) -> np.ndarray:
        """The differential equation for the network.

        Args:
            t (float): time
            w (np.ndarray): the network state

        Returns:
            np.ndarray: the derivative of the network state
        """

        wx = w[: self.Nx]
        wv = w[self.Nx : self.Nx + self.Nvel]
        wa = w[self.Nx + self.Nvel :]

        # velocity coupling to x: ----------------------------------------
        # This w_mx actually makes the swap a lot smoother:
        wx_mx = max(wx.max(), 1.0)
        wxdiff = np.subtract.outer(wx, wx)
        fac = wx * (wx_mx - wx)
        fac_mat = np.outer(fac, fac)

        # inhibition_mat_ij = (w_i - w_j) * w_i * (1 - w_i) * w_j * (1 - w_j):
        inhibition_mat = np.multiply(wxdiff, fac_mat)
        x_inhibition_term = inhibition_mat.sum(axis=0)

        # Velocity term:
        wx_1 = np.roll(wx, -1)
        wx1 = np.roll(wx, 1)

        # multiply velocity state by velocity stencil:
        v = abs(np.multiply(self.v_stencil, w[self.Nx : self.Nx + self.Nvel]))

        vm = v[: self.n2v].sum()
        vp = v[self.n2v + 1 :].sum()

        vel_term = -(vp + vm) * wx + vm * wx_1 + vp * wx1

        # acceleration coupling to v: ---------------------------------------------
        wv_mx = max(wv.max(), 1.0)
        wvdiff = np.subtract.outer(wv, wv)
        fac = wv * (wv_mx - wv)
        fac_mat = np.outer(fac, fac)

        # inhibition_mat_ij = (w_i - w_j) * w_i * (1 - w_i) * w_j * (1 - w_j):
        inhibition_mat = np.multiply(wvdiff, fac_mat)
        v_inhibition_term = inhibition_mat.sum(axis=0)

        # Acceleration term:
        # Add wrapped boundary value back to new boundary value, then set wrapped value to zero.
        # This turns off periodicity and terminates velocity at max values:
        wv_1 = np.roll(wv, -1)
        wv_1[0] += wv_1[-1]
        wv_1[-1] = 0
        wv1 = np.roll(wv, 1)
        wv1[-1] += wv1[0]
        wv1[0] = 0

        # multiply acceleration state by acceleration stencil:
        a = abs(np.multiply(self.a_stencil, w[self.Nx + self.Nvel :]))

        am = a[: self.n2a].sum()
        ap = a[self.n2a :].sum()

        acc_term = -(ap + am) * wv + am * wv_1 + ap * wv1

        # ---------------------------------------------

        # Direct impulse stimuli applied to each kinematic variable:
        # Sets impulse terms to 1.0, and the rest are inhibited by being set to -w.
        if self.dir_impl and (idx := self.dir_impl.get_impulse(t)) is not None:
            x_impulse_term = -np.copy(wx)
            x_impulse_term[idx] = 1.0 - wx[idx]
        else:
            x_impulse_term = np.zeros_like(wx)

        if self.vel_impl and (idx := self.vel_impl.get_impulse(t)) is not None:
            vel_impulse_term = -np.copy(wv)
            vel_impulse_term[idx] = 1.0 - wv[idx]
        else:
            vel_impulse_term = np.zeros_like(wv)

        if self.acc_impl and (idx := self.acc_impl.get_impulse(t)) is not None:
            acc_impulse_term = -np.copy(wa)
            acc_impulse_term[idx] = 1.0 - wa[idx]
        else:
            # acc_impulse_term = np.zeros_like(wa)
            # If no acceleration impulse, then decay the acceleration state to zero:
            acc_impulse_term = -np.copy(wa)

        w_prime = np.zeros_like(w)

        w_prime[: self.Nx] += 1.0 * vel_term - 20.0 * x_inhibition_term + x_impulse_term
        w_prime[self.Nx : self.Nx + self.Nvel] += (
            1.0 * acc_term - 20.0 * v_inhibition_term + vel_impulse_term
        )
        w_prime[self.Nx + self.Nvel :] += acc_impulse_term

        return w_prime

    def solve(self, t: np.ndarray, w0: np.ndarray) -> np.ndarray:
        """Solve the differential equation for the network.

        Args:
            t (np.ndarray): time range
            w0 (np.ndarray): the initial network state

        Returns:
            np.ndarray: the network solution
        """
        return solve_ivp(self.w_p, (t[0], t[-1]), w0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Nacc = 6
    Nvel = 5
    Nx = 10

    n0a = Nx + Nvel  # base of acceleration neurons
    # direct impulses:
    acc_impl = ImpulseDict(
        N=Nacc,
        id={
            3: ((8.0, 12.5),),
            2: ((19.0, 25.0),),
        },
    )
    vel_impl = ImpulseDict(N=Nvel, id={2: ((42.0, 45.0),)})
    dir_impl = ImpulseDict(
        N=Nx,
        id={
            # 1: ((5.0, 11.0),),
            # 4: ((11.0, 15.0),),
            # 8: ((19.0, 25.0),),
            # 3: ((25.0, 29.0),),
        },
    )

    # simulation domain:
    init_ary = [0.0] * (Nx + Nvel + Nacc)
    init_ary[4] = 1.0  # initial position
    init_ary[Nx + 2] = 1.0  # initial velocity
    tm_range = [0, 48]

    cann = CANNKinematic(
        Nacc, Nvel, Nx, acc_impl=acc_impl, vel_impl=vel_impl, dir_impl=dir_impl
    )

    sol = cann.solve(tm_range, init_ary)

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    # spread curves out in y:
    asol = sol.y[Nx + Nvel :]
    for i in range(asol.shape[0]):
        asol[i, :] += i * 0.04
    axs[0].plot(sol.t, asol.T)  # sol.y[Nx + Nvel :].T)
    axs[0].set_ylabel("acc")

    vsol = sol.y[Nx : Nx + Nvel]
    for i in range(vsol.shape[0]):
        vsol[i, :] += i * 0.02
    axs[1].plot(sol.t, vsol.T)  # sol.y[Nx : Nx + Nvel].T)
    axs[1].set_ylabel("vel")

    xsol = sol.y[:Nx]
    for i in range(xsol.shape[0]):
        xsol[i, :] += i * 0.02
    axs[2].plot(sol.t, xsol.T)  # sol.y[:Nx].T)
    axs[2].set_ylabel("x")

    # plt.plot(sol.t, sol.y.T[:, Nx + Nvel :])
    # plt.xlabel("time")
    # plt.ylabel("fire rate")
    plt.show()
