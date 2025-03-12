import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

# from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from sim_parameters import (
    design,
    dt,
    goal,
    integrator,
    model,
    mpar,
    robot,
    t_final,
    x0,
)

from pendulomovich.config import Config
from pendulomovich.controller import MPPIController

name = "vimmpi"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "IMPPI",
    "short_description": "Stabilization of iLQR trajectory with time-varying LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "adk",
}


# model parameters
# design = "design_A.0"
# model = "model_1.0"
# robot = "double_pendulum"
torque_limit = [6.0, 0.0]

# model_par_path = (
#     "../third_party/double_pendulum/data/system_identification/identified_parameters/"
#     + design
#     + "/"
#     + model
#     + "/model_parameters.yml"
# )
# mpar = model_parameters()
# mpar.load_yaml(model_par_path)
# mpar.set_torque_limit(torque_limit)

# simulation parameters
# dt = 0.001
# t_final = 70.0
# integrator = "runge_kutta"
# x0 = [-np.pi / 6 - 0.0, -np.pi / 2 - 0.0, 0.0, 0.0]
# goal = [np.pi, 0.0, 0.0, 0.0]

# setup savedir
# timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
# save_dir = os.path.join("data", design, model, robot, "mppi", timestamp)
# os.makedirs(save_dir)

cfg = Config(
    key=jax.random.PRNGKey(0),
    horizon=20,
    samples=4096,
    exploration=0,
    lambda_=50.0,
    alpha=1.0,  # balance between low energy and smoothness
    # Disturbance detection parameters
    deviation_type="time",
    dx_delta_max=1e-1,
    dt_delta_max=0.02,
    # Baseline control parameters
    baseline_control_type="zero",
    model_path="../../../data/policies/design_C.1/model_1.1/pendubot/AR_EAPO/model.zip",
    robot="pendubot",
    lqr_dt=0.005,
    sigma=jnp.diag(jnp.array([0.2, 0.2])),
    state_dim=4,
    act_dim=2,
    act_min=-jnp.array(torque_limit),
    act_max=jnp.array(torque_limit),
    Qdiag=jnp.array([10.0, 1.0, 0.1, 0.1]),
    Rdiag=jnp.array([0.1, 0.1]),
    Pdiag=jnp.array([5.0, 5.0, 2.0, 2.0]),
    terminal_coeff=1e6,
    mppi_dt=0.02,
    mpar=mpar,
    mppi_integrator="implicit",
)

controller = MPPIController(config=cfg)
controller.set_goal(goal)
controller.init()
