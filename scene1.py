"""
This scenario demonstrates the transient performance of
a feedback controller linearly interpolating two controllers,
which are both desgined to stabilize the morphing aircraft
at each trim points.
"""
import os
import shutil
import sys
import numpy as np

import fym.logging as logging
from fym.utils.linearization import jacob_analytic
from fym.agents.LQR import clqr
from fym.core import BaseEnv, BaseSystem
from fym.models import aircraft

INITIAL_PERTURB = np.array([[2, np.deg2rad(0), 0, np.deg2rad(-3)]]).T
ETA1 = (1, 0)  # Loiter
ETA2 = (0, 1)  # Dash
LQR_Q1 = np.diag([0.1, 1, 1, 1, 10, 1])
LQR_R1 = np.diag([1, 100])
LQR_Q2 = np.diag([1, 1, 1, 1, 10, 1])
LQR_R2 = np.diag([1, 100])

# Morphing actuator
TAU_MORPHING = 2  # sec
SATURATION_MORPHING = np.array([
    [0, 0],  # Lower limits
    [1, 1]   # Upper limits
])[..., None]
RATE_LIMIT_MORPHING = np.array([
    [-0.3, -0.2],
    [0.3, 0.2]
])[..., None]


class Env(BaseEnv):
    """
    Plane:
        state (x)   : 4x1 vector of (V, alpha, q, gamma)
        control (u) : 2x1 vector of (delt, dele)
        morph (eta) : 2x1 vector of (eta1, eta2)
    """
    def __init__(self):
        super().__init__(dt=0.01, max_t=40)
        self.plane = aircraft.MorphingLon(INITIAL_PERTURB)
        self.thrust_actuator = Actuator(
            np.zeros((2, 1)),
            A=np.array([[0, 1], [0, 0]]),
            B=np.array([[0], [1]]),
            saturation=(
                np.array([[self.plane.control_limits["delt"][0], -np.inf]]).T,
                np.array([[self.plane.control_limits["delt"][1], np.inf]]).T,
            )
        )
        self.morphing_actuator = Actuator(
            np.zeros((2, 1)),
            A=np.array(-1 / TAU_MORPHING),
            B=np.array(1 / TAU_MORPHING),
            saturation=SATURATION_MORPHING,
            rate_limit=RATE_LIMIT_MORPHING,
        )

        # Calculate trim points (#1, #2)
        # Trim 1: V: 20 m/s, h: 300 m/s, eta1: 1, eta2: 0
        # Trim 2: V: 20 m/s, h: 300 m/s, eta1: 0, eta2: 1
        eta1 = ETA1
        eta2 = ETA2
        fixed1 = dict(V=20, h=300, eta=eta1)  # Loiter
        fixed2 = dict(V=20, h=300, eta=eta2)  # Dash
        options = dict(disp=False)
        trim1 = self.plane.get_trim(fixed=fixed1, options=options, verbose=True)
        trimx1, trimu1, trimeta1 = trim1
        trim2 = self.plane.get_trim(fixed=fixed2, options=options, verbose=True)
        trimx2, trimu2, trimeta2 = trim2

        # Augment the state with the thrust state
        # Thrust actuator state (trimmed) : delt_trim, delt_dot_trim = 0
        thrustx1 = np.vstack((trimu1[0], 0))
        thrustx2 = np.vstack((trimu2[0], 0))

        # Augmented x (xa) : V, alpha, Q, gamma, delt, delt_dot
        trimax1 = np.vstack((trimx1, thrustx1))
        trimax2 = np.vstack((trimx2, thrustx2))

        # Augmented u (ua) : deltc (acc) = 0, dele
        trimau1 = np.vstack((0, trimu1[1]))
        trimau2 = np.vstack((0, trimu2[1]))

        # Linearize the system at each trim point
        linf = jacob_analytic(self.deriv, i=0)
        ling = jacob_analytic(self.deriv, i=1)
        trim1 = (trimax1, trimau1, trimeta1)
        trim2 = (trimax2, trimau2, trimeta2)
        A1, A2 = map(linf, *zip(trim1, trim2))
        B1, B2 = map(ling, *zip(trim1, trim2))

        # Get LQR controller for each trim point
        K1, *_ = clqr(A1, B1, LQR_Q1, LQR_R1)
        K2, *_ = clqr(A2, B2, LQR_Q2, LQR_R2)

        eigs1 = np.linalg.eigvals(A1 - B1.dot(K1))
        eigs2 = np.linalg.eigvals(A2 - B2.dot(K2))
        print("Eigenvalues")
        print("===========")
        print(f"Trim1: {eigs1}")
        print(f"Trim2: {eigs2}")

        # Set initial states perturbed from the trim points
        self.plane.initial_state = trimx1 + INITIAL_PERTURB
        self.thrust_actuator.initial_state = thrustx1
        self.morphing_actuator.initial_state = trimeta1

        self.trimx1, self.trimthrustx1, self.trimax1 = trimx1, thrustx1, trimax1
        self.trimu1, self.trimau1, self.trimeta1 = trimau1, trimau1, trimeta1
        self.trimx2, self.trimthrustx2, self.trimax2 = trimx2, thrustx2, trimax2
        self.trimu2, self.trimau2, self.trimeta2 = trimu2, trimau2, trimeta2
        self.K1, self.K2 = K1, K2

        # Misc
        self.logger_callback = self._logger_callback_fn

    def _logger_callback_fn(self, i, t, y, *args):
        states = self.observe_dict(y)
        x, tx = states["plane"], states["thrust_actuator"]
        eta = states["morphing_actuator"]

        # Control
        u, deltc = self.ctrl(x, tx, eta)
        cu, ceta = self.clipping(u, eta)

        # Morping command
        etac = self.morphing_ctrl(t)

        return dict(
            time=t,
            state=x,
            control=cu,
            eta=ceta,
            etac=etac,
            deltc=deltc,
        )

    def set_dot(self, t):
        x = self.plane.state
        tx = self.thrust_actuator.state
        eta = self.morphing_actuator.state

        u, deltc = self.ctrl(x, tx, eta)
        cu, ceta = self.clipping(u, eta)

        # Morping command
        etac = self.morphing_ctrl(t)

        self.plane.dot = self.plane.deriv(x, cu, ceta)
        self.thrust_actuator.set_dot(deltc)
        self.morphing_actuator.set_dot(etac)

    def clipping(self, u, eta):
        tlim = self.plane.control_limits["delt"]
        elim = self.plane.control_limits["dele"]
        eta1lim = self.plane.control_limits["eta1"]
        eta2lim = self.plane.control_limits["eta2"]
        ulim = np.vstack((tlim, elim)).T[..., None]
        etalim = np.vstack((eta1lim, eta2lim)).T[..., None]

        cu = np.clip(u, *ulim)
        ceta = np.clip(eta, *etalim)
        return cu, ceta

    def deriv(self, ax, au, eta):
        x, tx = ax[:4], ax[4:]

        # Real control input
        u = np.vstack((tx[0], au[1]))
        deltc = np.atleast_2d(au[0])

        d1 = self.plane.deriv(x, u, eta)
        d2 = self.thrust_actuator.deriv(tx, deltc)
        return np.vstack((d1, d2))

    def step(self):
        *_, done = self.update()
        done = done or self.clock.time_over()
        return done

    def set_ctrl(self, ctrl):
        """
        input: x, tx, eta
        output: u, deltc
        """
        self.ctrl = ctrl.get
        self.ctrl_name = ctrl.name

    def set_morphing_ctrl(self, ctrl):
        """
        input: time
        output: etac
        """
        self.morphing_ctrl = ctrl.get
        self.morphing_ctrl_name = ctrl.name

    def get_expname(self):
        return "-".join([self.name, self.ctrl_name, self.morphing_ctrl_name])


class Actuator(BaseSystem):
    """
    Linear system with rate limits and saturations.

    Parameters:
        rate_limit: 2x(state_size)
        saturation: 2x(state_size)
    """
    def __init__(self, initial_state, A, B,
                 saturation=(-np.inf, np.inf), rate_limit=(-np.inf, np.inf)):
        self.rl = rate_limit
        self.sat = saturation
        super().__init__(initial_state)
        self.A = np.asarray(A)
        self.B = np.asarray(B)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = np.clip(state, *self.sat)

    def set_dot(self, c):
        x = self.state
        self.dot = self.deriv(x, c)

    def deriv(self, x, c):
        deriv = self.A.dot(x) + self.B.dot(c)
        return np.clip(deriv, *self.rl)


class BaseCtrl:
    def __init__(self):
        self.name = self.__class__.__name__

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


class ConstantGain(BaseCtrl):
    def __init__(self, trimax1, trimau1, K1):
        super().__init__()
        self.trimax1 = trimax1
        self.trimau1 = trimau1
        self.K1 = K1

    def get(self, x, tx, eta):
        trimax = self.trimax1
        trimau = self.trimau1
        K = self.K1

        # Augmeted state vector
        ax = np.vstack((x, tx))

        # Get delta_x
        dax = ax - trimax
        dau = -K.dot(dax)
        au = trimau + dau

        # Real control input
        u = np.vstack((tx[0], au[1]))
        deltc = np.atleast_2d(au[0])

        return u, deltc


class InterpGain(BaseCtrl):
    def __init__(self, env):
        super().__init__()
        self.trimeta1 = env.trimeta1
        self.trimeta2 = env.trimeta2
        self.trimax1 = env.trimax1
        self.trimax2 = env.trimax2
        self.trimau1 = env.trimau1
        self.trimau2 = env.trimau2
        self.K1 = env.K1
        self.K2 = env.K2

    def get(self, x, tx, eta):
        # Get trim approximations
        trim_pctg = (
            np.linalg.norm(eta - self.trimeta1)
            / np.linalg.norm(self.trimeta2 - self.trimeta1)
        )
        trimax = self.trimax1 + (self.trimax2 - self.trimax1) * trim_pctg
        trimau = self.trimau1 + (self.trimau2 - self.trimau1) * trim_pctg
        K = self.K1 + (self.K2 - self.K1) * trim_pctg

        # Augmeted state vector
        ax = np.vstack((x, tx))

        # Get delta_x
        dax = ax - trimax
        dau = -K.dot(dax)
        au = trimau + dau

        # Real control input
        u = np.vstack((tx[0], au[1]))
        deltc = np.atleast_2d(au[0])

        return u, deltc


class Switching(BaseCtrl):
    def __init__(self, env, seq):
        super().__init__()
        self.trimeta1 = env.trimeta1
        self.trimeta2 = env.trimeta2
        self.seq = np.asarray(seq)

    def get(self, t):
        fi = np.where(self.seq > t)[0]
        if fi.size == 0:
            fi = [len(self.seq)]

        if fi[0] % 2 == 0:
            return self.trimeta1
        else:
            return self.trimeta2


if __name__ == "__main__":
    env = Env()

    ctrls = {
        "fixed_xu1_K1": ConstantGain(env.trimax1, env.trimau1, env.K1),
        "fixed_xu1_K2": ConstantGain(env.trimax1, env.trimau1, env.K2),
        "fixed_xu2_K1": ConstantGain(env.trimax2, env.trimau2, env.K1),
        "fixed_xu2_K2": ConstantGain(env.trimax2, env.trimau2, env.K2),
        "interp": InterpGain(env)
    }

    if os.path.exists("data"):
        if input(
                f"Delete \"data\"? [Y/n]: ") in ["", "Y", "y"]:
            shutil.rmtree("data")
        else:
            sys.exit()

    morphing_ctrl = Switching(env, seq=[0, 20])
    env.set_morphing_ctrl(morphing_ctrl)

    for cname, ctrl in ctrls.items():
        ctrl.name = cname
        env.set_ctrl(ctrl)

        expname = env.get_expname()
        path = os.path.join("data", "scene1", expname + ".h5")
        env.logger = logging.Logger(path=path)
        env.reset()
        while True:
            env.render()
            done = env.step()
            if done:
                break
        env.close()

    import matplotlib.pyplot as plt
    import os
    from glob import glob

    scenepath = os.path.join("data", "scene1")
    pathlist = glob(os.path.join(scenepath, "*.h5"))
    # cnamelist = [
    #     "interp",
    #     "fixed_xu1_K1",
    #     "fixed_xu1_K2",
    #     "fixed_xu2_K1",
    #     "fixed_xu2_K2",
    # ]
    options = dict(
        fixed_xu1_K1=dict(
            color="k",
            ls="--",
            label="Fixed trim 1, gain 1",
        ),
        fixed_xu1_K2=dict(
            color="g",
            ls="--",
            label="Fixed trim 1, gain 2",
        ),
        fixed_xu2_K1=dict(
            color="k",
            ls="-.",
            label="Fixed trim 2, gain 1",
        ),
        fixed_xu2_K2=dict(
            color="g",
            ls="-.",
            label="Fixed trim 2, gain 2",
        ),
        interp=dict(
            color="r",
            label="Interpolated",
        ),
    )
    cnamelist = options.keys()

    datalist = {}
    for p in pathlist:
        for c in cnamelist:
            if c in p:
                datalist[c] = logging.load(p)

    # Figure 1
    fig, axes = plt.subplots(2, 2)

    ylabels = [
        r"$V_T$ [m/s]", r"$\alpha$ [deg]", r"$Q$ [deg/s]", r"$\gamma$ [deg]"]
    factor = [1] + [np.rad2deg(1)] * 3

    for i, ax in enumerate(axes.flat):
        ln = []
        for c in cnamelist:
            data = datalist[c]
            option = options[c]

            t = data["time"]
            state = data["state"].squeeze()

            ln += ax.plot(t, state[:, i] * factor[i], **option)
            ax.set_ylabel(ylabels[i])
            ax.set_xlabel(r"Time [sec]")

    fig.legend(*ax.get_legend_handles_labels())
    fig.tight_layout()

    # Figure 2
    fig, axes = plt.subplots(2, 2)

    ylabels = [r"$\delta_t$", r"$\delta_e$"]
    factor = [1, np.rad2deg(1)]

    for i, ax in enumerate(axes[0].flat):
        for c in cnamelist:
            data = datalist[c]
            option = options[c]

            t = data["time"]
            control = data["control"].squeeze()

            ax.plot(t, control[:, i] * factor[i], **option)
            ax.set_ylabel(ylabels[i])
            ax.set_xlabel(r"Time [sec]")

    ylabels = [r"$\eta_1$", r"$\eta_2$"]
    for i, ax in enumerate(axes[1].flat):
        for c in cnamelist:
            data = datalist[c]
            option = options[c]

            t = data["time"]
            eta = data["eta"].squeeze()

            ax.plot(t, eta[:, i], **option)
            ax.set_ylabel(ylabels[i])
            ax.set_xlabel(r"Time [sec]")

    fig.tight_layout()

    plt.show()
