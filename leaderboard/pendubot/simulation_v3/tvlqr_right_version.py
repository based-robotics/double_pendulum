from time import sleep
import os
from collections.abc import Callable

import numpy as np
import sympy as sp
import yaml
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr import iterative_riccati, lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.utils.csv_trajectory import load_trajectory, save_trajectory
from double_pendulum.utils.pcw_polynomial import InterpolateMatrix, InterpolateVector
from double_pendulum.utils.wrap_angles import wrap_angles_diff


def create_tv_lqr_control_generator(
    disc_lin_sys_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    inv_dyn_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dt: float,
    horizon: int,
    nu: int,
    Q: np.ndarray,
    R: np.ndarray,
    Q_final: np.ndarray,
    goal: np.ndarray,
    Kp_goal: np.ndarray = 10.0,
    Kd_goal: np.ndarray = 6.0,
    v_max: float = 5.0,
    a_max: float = 3.0,
):
    # TODO: is this correct?

    def control_generator(
        x0: np.ndarray,
        xf: np.ndarray,
    ) -> np.ndarray:
        # Generate cubic interpolation trajectory
        T = horizon * dt
        ts_values = np.linspace(0, T, horizon)
        q_parts = []
        v_parts = []
        a_parts = []

        # ==== Option 1: Trapezoidal velocity profile ====
        # # Total duration from ts_values (assumed evenly spaced from 0 to T)
        # T = ts_values[-1]
        # for i in range(x0[:2].shape[0]):
        #     # Extract initial and final positions and velocities
        #     q0_i = x0[:2][i]
        #     v0_i = x0[2:][i]
        #     qf_i = xf[:2][i]
        #     vf_i = xf[2:][i]

        #     # Compute overall displacement and get its sign
        #     d = qf_i - q0_i
        #     s = 1.0 if d >= 0 else -1.0
        #     d_abs = abs(d)
        #     # Work with nonnegative velocities along the movement direction
        #     v0_abs = v0_i * s
        #     vf_abs = vf_i * s
        #     v_peak = v_max
        #     # Calculate time and distance for the acceleration phase
        #     t_acc = max(0.0, (v_peak - v0_abs) / a_max)
        #     d_acc = v0_abs * t_acc + 0.5 * a_max * t_acc**2
        #     # print(f"t_acc: {t_acc}, d_acc: {d_acc}")

        #     # Calculate time and distance for the deceleration phase
        #     t_dec = max(0.0, (v_peak - vf_abs) / a_max)
        #     d_dec = v_peak * t_dec - 0.5 * a_max * t_dec**2
        #     # print(f"t_dec: {t_dec}, d_dec: {d_dec}")

        #     # Check if a full trapezoid is feasible; otherwise, use a triangular profile
        #     if d_acc + d_dec > d_abs:
        #         # Compute peak velocity for triangular profile
        #         t_acc = t_dec = (-v0_abs + np.sqrt(v0_abs**2 + a_max * d_abs)) / a_max
        #         d_acc = q0_i + v0_abs * t_acc + 0.5 * a_max * t_acc**2
        #         v_peak = v0_abs + a_max * t_acc
        #         t_const = 0.0
        #     else:
        #         t_const = (d_abs - d_acc - d_dec) / v_peak

        #     profile_time = t_acc + t_const + t_dec

        #     # Prepare arrays for the current dimension
        #     q_traj = np.zeros_like(ts_values)
        #     v_traj = np.zeros_like(ts_values)
        #     a_traj = np.zeros_like(ts_values)
        #     # print(f"Joint {i}:")
        #     for j, t_val in enumerate(ts_values):
        #         if t_val < t_acc:
        #             # Acceleration phase
        #             q_phase = q0_i + v0_abs * t_val + 0.5 * a_max * t_val**2
        #             v_phase = v0_abs + a_max * t_val
        #             a_phase = a_max
        #             # print(f"Accelerating with: {q_phase}, {v_phase}, {a_phase}")
        #         elif t_val < t_acc + t_const:
        #             # Constant velocity phase
        #             q_phase = q0_i + d_acc + v_peak * (t_val - t_acc)
        #             v_phase = v_peak
        #             a_phase = 0.0
        #             # print(f"Steady motion with: {q_phase}, {v_phase}, {a_phase}")
        #         elif t_val <= profile_time:
        #             # Deceleration phase
        #             t_dec_elapsed = t_val - (t_acc + t_const)
        #             q_phase = q0_i + d_acc + v_peak * t_const + v_peak * t_dec_elapsed - 0.5 * a_max * t_dec_elapsed**2
        #             v_phase = v_peak - a_max * t_dec_elapsed
        #             a_phase = -a_max
        #             # print(f"Decelerating with: {q_phase}, {v_phase}, {a_phase}")
        #         else:
        #             # After the profile time, hold the final state
        #             q_phase = qf_i
        #             v_phase = vf_abs
        #             a_phase = 0.0
        #             # print(f"Holding with: {q_phase}, {v_phase}, {a_phase}")
        #         # Apply the sign to return to the original motion direction
        #         q_traj[j] = q0_i + s * (q_phase - q0_i)
        #         v_traj[j] = s * v_phase
        #         a_traj[j] = s * a_phase
        #     q_parts.append(q_traj)
        #     v_parts.append(v_traj)
        #     a_parts.append(a_traj)
        # traj = np.vstack(q_parts + v_parts).T
        # acc = np.vstack(a_parts).T
        # === Option 2: Cubic interpolation ===
        A_least_squares = np.array(
            [
                [1, 0, 0, 0],
                [1, T, T**2, T**3],
                [0, 1, 0, 0],
                [0, 1, 2 * T, 3 * T**2],
            ]
        )
        q_parts = []
        v_parts = []
        a_parts = []
        num = x0[:2].shape[0]
        for i in range(num):
            q0_i = x0[:2][i]
            v0_i = x0[2:][i]
            q_des_i = xf[:2][i]
            v_des_i = xf[2:][i]
            b_least_squares = np.array([q0_i, q_des_i, v0_i, v_des_i])
            a = np.linalg.solve(A_least_squares, b_least_squares)
            q_i = a[0] + a[1] * ts_values + a[2] * ts_values**2 + a[3] * ts_values**3
            v_i = a[1] + 2 * a[2] * ts_values + 3 * a[3] * ts_values**2
            a_i = 2 * a[2] + 6 * a[3] * ts_values
            q_parts.append(q_i)
            v_parts.append(v_i)
            a_parts.append(a_i)
        traj = np.vstack(q_parts + v_parts).T
        acc = np.vstack(a_parts).T

        # Linear System Discretization
        # Version 1: inverse dynamics + goal PD
        # tau = np.array([inv_dyn_fn(traj[i], acc[i]) for i in range(horizon)])
        # baseline_u = tau - Kp_goal * (traj[:, :2] - goal[:2]) - Kd_goal * (traj[:, 2:] - goal[2:])
        # Version 2: goal PD
        # baseline_u = -Kp_goal * (traj[:, :2] - goal[:2]) - Kd_goal * (traj[:, 2:] - goal[2:])
        # Version 3: zero control
        baseline_u = np.zeros((horizon, nu))
        A_d, B_d = [], []
        for i in range(len(traj) - 1):
            A, B = disc_lin_sys_fn(
                traj[i],
                baseline_u[i],
                dt,
            )
            A_d.append(A)
            B_d.append(B)
        A_d = np.array(A_d)
        B_d = np.array(B_d)

        def dlqr_ltv(P, A, B):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
            return P, K

        # Compute LQR gain
        P = Q_final
        K = []
        for i in range(len(A_d) - 1, -1, -1):
            P, K_i = dlqr_ltv(P, A_d[i], B_d[i])
            K.append(K_i)
        K = K[::-1]

        # FIXME: should we skip the first control?
        return K, traj, baseline_u

    return control_generator


class TVLQRController(AbstractController):
    """TVLQRController
    Controller to stabilize a trajectory with TVLQR

    Parameters
    ----------
    mass : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.6]
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : array_like, optional
        shape=(2,), dtype=float, default=[0.3, 0.2]
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : array_like, optional
        shape=(2,), dtype=float, default=[0.3, 0.2]
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : array_like, optional
        shape=(2,), dtype=float, default=[0.1, 0.1]
        damping coefficients of the double pendulum actuators
        [b1, b2], units=[kg*m/s]
    gravity : float, optional
        default=9.81
        gravity acceleration (pointing downwards),
        units=[m/s²]
    coulomb_fric : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 0.0]
        coulomb friction coefficients for the double pendulum actuators
        [cf1, cf2], units=[Nm]
    inertia : array_like, optional
        shape=(2,), dtype=float, default=[None, None]
        inertia of the double pendulum links
        [I1, I2], units=[kg*m²]
        if entry is None defaults to point mass m*l² inertia for the entry
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    model_pars : model_parameters object, optional
        object of the model_parameters class, default=None
        Can be used to set all model parameters above
        If provided, the model_pars parameters overwrite
        the other provided parameters
        (Default value=None)
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
        (Default value="")
    num_break : int
        number of break points used for interpolation
        (Default value = 40)
        (Default value=100)
    """

    def __init__(
        self,
        Q=None,
        R=None,
        Qf=None,
        mass=[0.5, 0.6],
        length=[0.3, 0.2],
        com=[0.3, 0.2],
        damping=[0.1, 0.1],
        coulomb_fric=[0.0, 0.0],
        gravity=9.81,
        inertia=[None, None],
        torque_limit=[0.0, 1.0],
        model_pars=None,
        csv_path="",
        num_break=40,
        dt=2e-3,
        tf=2.0,
    ):
        super().__init__()

        # model parameters
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            self.Ir = model_pars.Ir
            self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        self.splant = SymbolicDoublePendulum(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.cfric,
            inertia=self.inertia,
            torque_limit=self.torque_limit,
        )

        self.num_break = int(tf / dt)

        self.dt = dt
        self.t0 = 0
        self.tf = tf
        self.ts = np.arange(0, self.tf, self.dt)

        # default parameters
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.goal = np.array([np.pi, 0.0, 0.0, 0.0])

        # initializations
        self.K = []
        self.traj = []
        self.inv_dyn_tau = []
        # self.k = []

        # Previous system state to check disturbance
        _, x0, _ = load_trajectory(csv_path=csv_path, with_tau=True)
        self._x_prev = x0[0]
        self._u_prev = np.zeros(2)
        self._t_prev = 0

        self.Kp = 50.0
        self.Kd = 0.4
        self.control_seq_fn = create_tv_lqr_control_generator(
            self.splant.linear_matrices_discrete,
            self.splant.inverse_dynamics,
            self.dt,
            self.num_break,
            nu=2,
            Q=self.Q,
            R=self.R,
            Q_final=self.Qf,
            goal=self.goal,
            Kp_goal=self.Kp,
            Kd_goal=self.Kd,
        )

        self.t_fall = 0.5
        self.t_fall_init = 0.0
        self.is_falling = False

    def init_(self, t=0):
        # print("INITIALIZATION!!!!!!")
        self._t_traj_init = t
        self._t_prev = t
        self.K, self.traj, self.inv_dyn_tau = self.control_seq_fn(self._x_prev, self.goal)

    def control_idx(self, t):
        t_traj = t - self._t_traj_init
        index = int((t_traj / self.tf) * (self.num_break - 1))
        return index

    def check_disturbance(self, controller_dt):
        return controller_dt > self.dt * 2

    def set_cost_parameters(self, Q, R, Qf):
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.init_()

    def get_control_output_(self, x, t):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        index = self.control_idx(t)
        if self.check_disturbance(t - self._t_prev) or index > self.num_break - 2:
            if not self.is_falling:
                self.t_fall_init = t
                self.is_falling = True

            if t - self.t_fall_init > self.t_fall:
                self._x_prev = x
                self.init_(t)
                index = self.control_idx(t)
                self.is_falling = False
            else:
                Kp_fall = 10.0
                Kd_fall = 5.0
                tau = -Kp_fall * (x[:2]) - Kd_fall * (x[2:])
                tau[0] = np.clip(tau[0], -self.torque_limit[0], self.torque_limit[0])
                tau[1] = np.clip(tau[0], -self.torque_limit[1], self.torque_limit[1])
                sleep(0.01)
                return tau

        # Rollout the dynamics for the horizon
        tau = (
            # self.splant.inverse_dynamics(x, self.splant.forward_dynamics(x, self._u_prev))
            -self.K[index] @ (np.array(x) - self.traj[index])
            - self.Kp * (x[:2] - self.goal[:2])
            - self.Kd * (x[2:] - self.goal[2:])
        )

        tau[0] = np.clip(tau[0], -self.torque_limit[0], self.torque_limit[0])
        tau[1] = np.clip(tau[0], -self.torque_limit[1], self.torque_limit[1])

        self._t_prev = t
        self._x_prev = x
        self._u_prev = tau
        sleep(0.01)
        return tau
