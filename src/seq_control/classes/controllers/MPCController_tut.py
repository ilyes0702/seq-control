from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from seq_control.utils.plotting_utils import *

import matplotlib.pyplot as plt
plt.style.use("src/seq_control/style.mplstyle")


# Import your ChemostatPlant class
from seq_control.classes.plants.ChemostatPlant import ChemostatPlant


# ==========================================
# 1. UNIVERSAL PLANT INTERFACE & WRAPPER
# ==========================================
class BasePlant(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, u: torch.Tensor, t: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class ChemostatWrapper(BasePlant):
    def __init__(self, chemostat_plant_instance):
        self.plant = chemostat_plant_instance

    @property
    def state_dim(self) -> int:
        return 2  # [Biomass x, Substrate s]

    @property
    def control_dim(self) -> int:
        return 1  # [Dilution rate u]

    @property
    def output_dim(self) -> int:
        return 1  # [Growth rate y]

    @property
    def device(self):
        return self.plant.device

    def step(self, state, u, t, dt):
        # Pass time step 't' through to the plant model
        return self.plant.step(state, u, t=t, dt=dt)


# ==========================================
# 2. AGNOSTIC MPC CONTROLLER
# ==========================================
class AgnosticMPCController(nn.Module):
    def __init__(
        self,
        plant: BasePlant,
        horizon: int = 15,
        lr: float = 0.05,
        num_sim_steps: int = 25,
        u_min: torch.Tensor | float = -float("inf"),
        u_max: torch.Tensor | float = float("inf"),
        w_y: torch.Tensor | float = 10.0,
        w_u: torch.Tensor | float = 0.01,
        w_du: torch.Tensor | float = 1.0,
        w_terminal: torch.Tensor | float = 5.0,
    ):
        super().__init__()
        self.plant = plant
        self.horizon = horizon
        self.lr = lr
        self.num_sim_steps = num_sim_steps

        self.u_dim = plant.control_dim
        self.y_dim = plant.output_dim
        
        device = getattr(plant, "device", torch.device("cpu"))
        self.device = device

        self.u_min = self._format_bounds(u_min, self.u_dim, device)
        self.u_max = self._format_bounds(u_max, self.u_dim, device)

        self.w_y = self._format_weight(w_y, self.y_dim, device)
        self.w_u = self._format_weight(w_u, self.u_dim, device)
        self.w_du = self._format_weight(w_du, self.u_dim, device)
        self.w_terminal = self._format_weight(w_terminal, self.y_dim, device)

        self.u_plan = None

    def _format_bounds(self, val, dim, device):
        if isinstance(val, (int, float)):
            return torch.full((dim,), float(val), device=device)
        return val.to(device)

    def _format_weight(self, val, dim, device):
        if isinstance(val, (int, float)):
            return torch.full((dim,), float(val), device=device)
        return val.to(device)

    def reset_plan(self, batch_size: int = 1):
        init_u = torch.clamp((self.u_min + self.u_max) / 2.0, -1.0, 1.0)
        self.u_plan = init_u.unsqueeze(0).unsqueeze(0).repeat(self.horizon, batch_size, 1).to(self.device)
        self.u_plan.requires_grad_(True)

    def _compute_cost(
        self,
        initial_state: torch.Tensor,
        y_ref_horizon: torch.Tensor,
        dt: float,
        u_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        state = initial_state
        total_cost = 0.0

        u_clamped = torch.clamp(self.u_plan, self.u_min, self.u_max)
        last_u = u_prev if u_prev is not None else u_clamped[0]

        for h in range(self.horizon):
            u_h = u_clamped[h]

            # Generic forward step on target plant
            state, y_pred = self.plant.step(state, u_h, t=0.0, dt=dt)

            ref_h = y_ref_horizon[h] if y_ref_horizon.dim() == 3 else y_ref_horizon

            # 1. Output tracking loss
            w_tracking = self.w_terminal if h == (self.horizon - 1) else self.w_y
            y_err = (y_pred - ref_h) ** 2
            total_cost = total_cost + torch.mean(y_err * w_tracking)

            # 2. Control input magnitude penalty
            total_cost = total_cost + torch.mean((u_h ** 2) * self.w_u)

            # 3. Slew-rate penalty (du / dt)
            du = u_h - last_u
            total_cost = total_cost + torch.mean((du ** 2) * self.w_du)

            last_u = u_h

        return total_cost

    def get_action(
        self,
        current_state: torch.Tensor,
        y_ref_horizon: torch.Tensor,
        dt: float,
        last_action: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Guarantee device alignment across inputs
        current_state = current_state.to(self.device)
        y_ref_horizon = y_ref_horizon.to(self.device)
        if last_action is not None:
            last_action = last_action.to(self.device)

        batch_size = current_state.size(0)

        if self.u_plan is None or self.u_plan.size(1) != batch_size:
            self.reset_plan(batch_size)

        optimizer = optim.Adam([self.u_plan], lr=self.lr)

        for _ in range(self.num_sim_steps):
            optimizer.zero_grad()
            cost = self._compute_cost(current_state, y_ref_horizon, dt, u_prev=last_action)
            cost.backward()
            optimizer.step()

        with torch.no_grad():
            u_apply = torch.clamp(self.u_plan[0], self.u_min, self.u_max)
            u_next = torch.cat([self.u_plan[1:], self.u_plan[-1:]], dim=0)
            self.u_plan = u_next.detach().clone().requires_grad_(True)

        return u_apply


# ==========================================
# 3. CLOSED-LOOP EXPERIMENT SIMULATION
# ==========================================

# Mock/actual hyperparameter config setup
hyperparam_config_ChemostatPlant = {
    "train": {"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
    "training_data_cfg": {"dt": 0.1},
    "plant": {
        "mu-max": 0.5,
        "Ks": 0.2,
        "Y": 0.4,
        "sR": 10.0
    }
}

# 1. Instantiate concrete plant and wrap it
chemostat = ChemostatPlant(hyperparam_config_ChemostatPlant)
plant_adapter = ChemostatWrapper(chemostat)

# 2. Instantiate MPC controller
mpc = AgnosticMPCController(
    plant=plant_adapter,
    horizon=12,
    lr=0.03,
    num_sim_steps=20,
    u_min=0.01,
    u_max=0.8,
    w_y=15.0,     # Strong tracking weight
    w_du=2.5      # Smooth action updates
)

# 3. Define Simulation Setup
sim_steps = 100
dt = hyperparam_config_ChemostatPlant["training_data_cfg"]["dt"]
batch_size = 1

current_state = chemostat.get_initial_state(batch_size=batch_size)

# Create sinusoidal target growth rate trajectory (\mu)
target_time_steps = torch.linspace(0, 3.14 * 2, sim_steps, device=plant_adapter.device)
target_mu_trajectory = torch.sin(target_time_steps) * 0.1 + 0.25  # Target output signal [sim_steps]

# Closed-loop tracking storage lists
history_states, history_u, history_y = [], [], []
last_action = None

# 4. Simulation Loop
for step in range(sim_steps):
    # Construct reference horizon slice [horizon, batch_size, output_dim]
    if step + mpc.horizon <= sim_steps:
        y_ref_horizon = target_mu_trajectory[step : step + mpc.horizon].view(mpc.horizon, batch_size, 1)
    else:
        # Pad tail near the end of trajectory
        remainder = sim_steps - step
        pad_len = mpc.horizon - remainder
        tail = target_mu_trajectory[step:]
        pad = target_mu_trajectory[-1].expand(pad_len)
        y_ref_horizon = torch.cat([tail, pad]).view(mpc.horizon, batch_size, 1)

    # Solve MPC action step
    u_cmd = mpc.get_action(current_state, y_ref_horizon, dt=dt, last_action=last_action)

    # Step physical plant forward
    next_state, y_actual = plant_adapter.step(current_state, u_cmd, t=step * dt, dt=dt)

    # Log metrics
    history_states.append(current_state.detach().cpu())
    history_u.append(u_cmd.detach().cpu())
    history_y.append(y_actual.detach().cpu())

    last_action = u_cmd
    current_state = next_state


# ==========================================
# 4. PLOTTING WITH PLOT_STACKED
# ==========================================
t = np.arange(sim_steps) * dt

# Extract 1D NumPy arrays
y_actual = np.array([y[0, 0].item() for y in history_y])
y_ref = target_mu_trajectory.cpu().numpy()
u_actual = np.array([u[0, 0].item() for u in history_u])

signals = [
    [y_ref, y_actual],  # Subplot 1: Output tracking
    [u_actual]          # Subplot 2: Control input
]

labels = [
    [r"$\mu_{\mathrm{ref}}$ (Target)", r"$\mu_{\mathrm{actual}}$ (Plant)"],
    [r"$D$ (Dilution Rate)"]
]

ylabels = [
    r"Growth Rate $\mu$ [$\mathrm{h}^{-1}$]",
    r"Control Input $D$ [$\mathrm{h}^{-1}$]"
]

img = plot_stacked(
    t=t,
    signals=signals,
    labels=labels,
    ylabel=ylabels,
    xlabel="Time [h]",
    title="MPC Tracking \& Control Input Evolution",
    asp=0.3,
    show=True
)