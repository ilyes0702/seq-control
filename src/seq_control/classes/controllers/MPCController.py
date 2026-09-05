from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.seq_control.classes.plants import BasePlant
from seq_control.utils.plotting_utils import *

import matplotlib.pyplot as plt
plt.style.use("src/seq_control/style.mplstyle")


# Import your ChemostatPlant class
from seq_control.classes.plants.ChemostatPlant import ChemostatPlant



# ==========================================
# 2. AGNOSTIC MPC CONTROLLER
# ==========================================
class MPCController(nn.Module):
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


