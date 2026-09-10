import torch


class TrophophasePlant:
    """Trophophase bioprocess plant simulation model.

    Models a 2-state fed-batch bioprocess system representing the trophophase 
    (growth phase) of fermentation with a time-dependent reactor volume :math:`V(t)` 
    and Monod growth kinetics. Integrated via a 5th-order Dormand-Prince (RK45) 
    numerical solver scheme.

    :param hyperparam_config: Configuration dictionary containing device ('train'),
        time step ('training_data_cfg'), and kinetic/bioprocess parameters ('plant').
    :type hyperparam_config: dict
    """

    def __init__(self, hyperparam_config):
        """Initialize plant physical parameters, kinetic constants, and device placement.

        :param hyperparam_config: Nested configuration dictionary containing training 
            and plant parameters.
        :type hyperparam_config: dict
        """
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["training_data_cfg"]["dt"]
        self.plant_cfg = hyperparam_config["plant"]

        # Biological and physical parameters from Table 1
        self.mu_max = self.plant_cfg["mu_max"]      # [1/h]
        self.Ks = self.plant_cfg["Ks"]              # [mg S/l]
        self.m_S = self.plant_cfg["m_S"]            # [mg S/(g TS h)]
        self.p1 = self.plant_cfg["p1"]              # [g TS/(mg S)]
        self.p2 = self.plant_cfg["p2"]              # [mg S/l]

        self.hyperparam_config = hyperparam_config

    def get_volume(self, t):
        """Calculate the time-dependent reactor volume :math:`V(t)`.

        Computes volume according to a piecewise ramp function with linear filling 
        and holding intervals:
        
        .. math::
            V(t) = 150 + 2(t - 5)\\sigma(t - 5) - 2(t - 15)\\sigma(t - 15)

        :param t: Current simulation time or timestamp tensor.
        :type t: float or int or torch.Tensor
        :returns: Computed reactor volume :math:`V(t)` matching the target execution device.
        :rtype: torch.Tensor
        """
        # Ensure t is a tensor for element-wise operations
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
            
        ramp1 = torch.clamp(t - 5.0, min=0.0)
        ramp2 = torch.clamp(t - 15.0, min=0.0)
        
        return 150.0 + 2.0 * ramp1 - 2.0 * ramp2

    def get_initial_state(self, batch_size, randomize=True):
        """Generate initial state vectors for a batch of simulation trajectories.

        State vector components layout:
        ``[x1 (Biomass mass), x2 (Substrate mass)]``

        :param batch_size: Number of parallel batch instances to sample.
        :type batch_size: int
        :param randomize: If True, applies uniform random scaling within range ``[0.95, 1.05)`` 
            around nominal values. If False, fills strictly with nominal values. Defaults to True.
        :type randomize: bool, optional
        :returns: Initial state tensor of shape ``(batch_size, 2)``.
        :rtype: torch.Tensor
        """
        # Fetch nominal values from config
        x1_nominal = self.hyperparam_config["plant"]["x10"]
        x2_nominal = self.hyperparam_config["plant"]["x20"]
        
        if randomize:
            # Randomization formula: nominal * (0.95 + 0.1 * rand) -> [0.95*nominal, 1.05*nominal)
            x1_values = x1_nominal * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x2_values = x2_nominal * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
        else:
            # Create tensors filled entirely with the nominal values
            x1_values = torch.full((batch_size, 1), x1_nominal, device=self.device, dtype=torch.float32)
            x2_values = torch.full((batch_size, 1), x2_nominal, device=self.device, dtype=torch.float32)
            
        return torch.cat([x1_values, x2_values], dim=1)

    def get_y(self, state, t):
        """Compute the observable plant output (specific growth rate :math:`\\mu`).

        Calculates specific growth rate according to Monod growth kinetics based on 
        substrate concentration :math:`c_2 = x_2 / V(t)`:

        .. math::
            y = \\mu(c_2) = \\frac{\\mu_{\\mathrm{max}} c_2}{K_s + c_2}

        :param state: Plant state tensor of shape ``(batch_size, 2)`` containing ``[x1, x2]``.
        :type state: torch.Tensor
        :param t: Current simulation time step or timestamp.
        :type t: float or int or torch.Tensor
        :returns: Computed specific growth rate :math:`\\mu` of shape ``(batch_size, 1)``.
        :rtype: torch.Tensor
        """
        x2 = state[:, 1:2]
        V = self.get_volume(t)
        
        # Substrate concentration c2
        c2 = x2 / V
        
        # Monod growth kinetics
        mu = (self.mu_max * c2) / (self.Ks + c2)
        return mu

    def get_v_dot(self, t):
        """Calculate the time derivative of reactor volume :math:`\\mathrm{d}V/\\mathrm{d}t`.

        Evaluates rate of volume change using Heaviside step functions :math:`\\sigma(t)`:

        .. math::
            \\frac{\\mathrm{d}V}{\\mathrm{d}t} = 2\\sigma(t - 5) - 2\\sigma(t - 15)

        :param t: Current simulation time or timestamp tensor.
        :type t: float or int or torch.Tensor
        :returns: Time derivative of volume :math:`\\mathrm{d}V/\\mathrm{d}t`.
        :rtype: torch.Tensor
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
        
        # Heaviside step function: 1.0 if t >= threshold else 0.0
        sigma1 = (t >= 5.0).to(torch.float32)
        sigma2 = (t >= 15.0).to(torch.float32)
        
        return 2.0 * sigma1 - 2.0 * sigma2

    def get_y_dot(self, state, u1, t):
        """Calculate the first time derivative of specific growth rate :math:`\\mathrm{d}y/\\mathrm{d}t`.

        Computes the rate of change of observable output :math:`y = \\mu(c_2)` using the chain rule:

        .. math::
            \\frac{\\mathrm{d}y}{\\mathrm{d}t} = \\frac{\\mathrm{d}y}{\\mathrm{d}c_2} \\cdot \\frac{\\mathrm{d}c_2}{\\mathrm{d}t}

        :param state: Plant state tensor of shape ``(batch_size, 2)`` containing ``[x1, x2]``.
        :type state: torch.Tensor
        :param u1: Applied substrate feed control input tensor of shape ``(batch_size, 1)``.
        :type u1: torch.Tensor
        :param t: Current simulation time.
        :type t: float or int or torch.Tensor
        :returns: First derivative :math:`\\mathrm{d}y/\\mathrm{d}t` of shape ``(batch_size, 1)``.
        :rtype: torch.Tensor
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        V = self.get_volume(t)
        V_dot = self.get_v_dot(t)
        
        c2 = x2 / V
        y = (self.mu_max * c2) / (self.Ks + c2)
        
        # System dynamics: dx2/dt
        _, dx2dt = self.dynamics(x1, x2, u1, t)
        
        # dc2/dt
        c2_dot = (dx2dt - c2 * V_dot) / V
        
        # dy/dc2
        dy_dc2 = (self.mu_max * self.Ks) / ((self.Ks + c2) ** 2)
        
        y_dot = dy_dc2 * c2_dot
        return y_dot

    def get_y_ddot(self, state, u1, t, u1_dot=0.0):
        """Calculate the second time derivative of specific growth rate :math:`\\mathrm{d}^2y/\\mathrm{d}t^2`.

        Computes acceleration of observable output :math:`y = \\mu(c_2)` incorporating higher-order 
        concentration derivatives and state trajectories.

        :param state: Plant state tensor of shape ``(batch_size, 2)`` containing ``[x1, x2]``.
        :type state: torch.Tensor
        :param u1: Applied substrate feed control input tensor of shape ``(batch_size, 1)``.
        :type u1: torch.Tensor
        :param t: Current simulation time.
        :type t: float or int or torch.Tensor
        :param u1_dot: First time derivative of control input :math:`\\mathrm{d}u_1/\\mathrm{d}t`, defaults to 0.0.
        :type u1_dot: float or torch.Tensor or int, optional
        :returns: Second derivative :math:`\\mathrm{d}^2y/\\mathrm{d}t^2` of shape ``(batch_size, 1)``.
        :rtype: torch.Tensor
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        V = self.get_volume(t)
        V_dot = self.get_v_dot(t)
        # V_ddot is 0 almost everywhere
        V_ddot = 0.0 
        
        c2 = x2 / V
        y = (self.mu_max * c2) / (self.Ks + c2)
        
        # First derivatives
        dx1dt, dx2dt = self.dynamics(x1, x2, u1, t)
        c2_dot = (dx2dt - c2 * V_dot) / V
        
        dy_dc2 = (self.mu_max * self.Ks) / ((self.Ks + c2) ** 2)
        y_dot = dy_dc2 * c2_dot
        
        # Secondary derivatives for state equations
        dx2dt_dot = -(1.0 / self.p1) * (y_dot * x1 + y * dx1dt) - self.m_S * dx1dt + self.p2 * u1_dot
        
        # d^2c2/dt^2
        c2_ddot = (dx2dt_dot - 2.0 * c2_dot * V_dot - c2 * V_ddot) / V
        
        # d^2y/dc2^2
        d2y_dc22 = -2.0 * (self.mu_max * self.Ks) / ((self.Ks + c2) ** 3)
        
        # d^2y/dt^2 = (dy/dc2)*c2_ddot + (d2y/dc22)*(c2_dot^2)
        y_ddot = dy_dc2 * c2_ddot + d2y_dc22 * (c2_dot ** 2)
        
        return y_ddot

    def dynamics(self, x1, x2, u1, t):
        """Compute state derivative vector :math:`\\mathrm{d}x/\\mathrm{d}t` for the trophophase model.

        Evaluates continuous differential equations governing biomass mass (:math:`x_1`) 
        and substrate mass (:math:`x_2`):

        .. math::
            \\frac{\\mathrm{d}x_1}{\\mathrm{d}t} &= \\mu(c_2) x_1 \\\\
            \\frac{\\mathrm{d}x_2}{\\mathrm{d}t} &= -\\frac{1}{p_1} \\mu(c_2) x_1 - m_S x_1 + p_2 u_1

        :param x1: Biomass mass state tensor of shape ``(batch_size, 1)``.
        :type x1: torch.Tensor
        :param x2: Substrate mass state tensor of shape ``(batch_size, 1)``.
        :type x2: torch.Tensor
        :param u1: Substrate feed control input tensor of shape ``(batch_size, 1)``.
        :type u1: torch.Tensor
        :param t: Current simulation time.
        :type t: float or int or torch.Tensor
        :returns: Tuple ``(dx1dt, dx2dt)`` containing state continuous derivatives.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        V = self.get_volume(t)
        c2 = x2 / V
        mu = (self.mu_max * c2) / (self.Ks + c2)
        
        # dx1/dt
        dx1dt = mu * x1
        
        # dx2/dt
        dx2dt = -(1.0 / self.p1) * mu * x1 - self.m_S * x1 + self.p2 * u1
        return dx1dt, dx2dt

    def step(self, state, u, t, dt):
        """Perform dynamic state integration using the Dormand-Prince (RK45) scheme.

        Advances the system state forward across time step ``dt`` using a 5th-order accurate 
        embedded Runge-Kutta numerical solver step.

        :param state: Current state tensor of shape ``(batch_size, 2)``.
        :type state: torch.Tensor
        :param u: Applied control input (dilution/feed rate) tensor of shape ``(batch_size, 1)``.
        :type u: torch.Tensor
        :param t: Current simulation start time.
        :type t: float or int or torch.Tensor
        :param dt: Time integration step size.
        :type dt: float or int or torch.Tensor
        :returns: Tuple ``(state_next, y_next)`` containing updated state tensor of shape 
            ``(batch_size, 2)`` and observable output tensor of shape ``(batch_size, 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # k1 at t
        dx1_1, ds1_1 = self.dynamics(x1, x2, u, t)
        
        # k2 at t + (1/5)*dt
        x1_2 = x1 + dt * (1/5 * dx1_1)
        x2_2 = x2 + dt * (1/5 * ds1_1)
        dx1_2, ds1_2 = self.dynamics(x1_2, x2_2, u, t + 0.2 * dt)
        
        # k3 at t + (3/10)*dt
        x1_3 = x1 + dt * (3/40 * dx1_1 + 9/40 * dx1_2)
        x2_3 = x2 + dt * (3/40 * ds1_1 + 9/40 * ds1_2)
        dx1_3, ds1_3 = self.dynamics(x1_3, x2_3, u, t + 0.3 * dt)
        
        # k4 at t + (4/5)*dt
        x1_4 = x1 + dt * (44/45 * dx1_1 - 56/15 * dx1_2 + 32/9 * dx1_3)
        x2_4 = x2 + dt * (44/45 * ds1_1 - 56/15 * ds1_2 + 32/9 * ds1_3)
        dx1_4, ds1_4 = self.dynamics(x1_4, x2_4, u, t + 0.8 * dt)
        
        # k5 at t + (8/9)*dt
        x1_5 = x1 + dt * (19372/6561 * dx1_1 - 25360/2187 * dx1_2 + 64448/6561 * dx1_3 - 212/729 * dx1_4)
        x2_5 = x2 + dt * (19372/6561 * ds1_1 - 25360/2187 * ds1_2 + 64448/6561 * ds1_3 - 212/729 * ds1_4)
        dx1_5, ds1_5 = self.dynamics(x1_5, x2_5, u, t + (8/9) * dt)
        
        # k6 at t + dt
        x1_6 = x1 + dt * (9017/3168 * dx1_1 - 355/33 * dx1_2 + 46732/5247 * dx1_3 + 49/176 * dx1_4 - 5103/18656 * dx1_5)
        x2_6 = x2 + dt * (9017/3168 * ds1_1 - 355/33 * ds1_2 + 46732/5247 * ds1_3 + 49/176 * ds1_4 - 5103/18656 * ds1_5)
        dx1_6, ds1_6 = self.dynamics(x1_6, x2_6, u, t + dt)

        # 5th-order accurate state update
        x1_next = x1 + dt * (35/384 * dx1_1 + 500/1113 * dx1_3 + 125/192 * dx1_4 - 2187/6784 * dx1_5 + 11/84 * dx1_6)
        x2_next = x2 + dt * (35/384 * ds1_1 + 500/1113 * ds1_3 + 125/192 * ds1_4 - 2187/6784 * ds1_5 + 11/84 * ds1_6)

        state_next = torch.cat([x1_next, x2_next], dim=1)
        
        # Return next state and tracking output evaluated at t + dt
        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        """Configure matplotlib visualization metadata and LaTeX labels for simulation plots.

        :returns: List of plotting configuration dictionaries specifying columns, labels, and axis titles.
        :rtype: list[dict]
        """
        return [
            {
                "cols": ["t"],
                "labels": [r"$t$ [$\mathrm{h}$]"],
                "xlabel": [r"$t$ [$\mathrm{h}$]"]
            },
            {
                "cols": ["x_1", "x_2"],
                "labels": [r"$x_1$ [$\mathrm{g}$]", r"$x_2$ [$\mathrm{mg}$]"],
                "ylabel": [r"$x_1$ [$\mathrm{g}$]", r"$x_2$ [$\mathrm{mg}$]"]
            },
            {
                "cols": ["y"],
                "labels": [r"$y$ [$\mathrm{h}^{-1}$]"],
                "ylabel": r"$y$ [$\mathrm{h}^{-1}$]"
            },
            {
                "cols": ["u"],
                "labels": [r"$u$ [$\mathrm{h}^{-1}$]"],
                "ylabel": r"$u$ [$\mathrm{h}^{-1}$]"
            }
        ]

# Default hyperparameter configuration
hyperparam_config_TrophophasePlant = {
    "plant" :{
        "mu_max": 0.12,
        "Ks": 50,
        "m_S": 23.0, 
        "p1": 0.00047,
        "p2": 200000.0,
        
        "u_1_D_center_min": 0.6,
        "u_1_D_center_max": 0.9,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min": 0,
        "x_1_hard_max": None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.12,

        "x10": 1500.0,   #wenn trainiert mit 1500 aber getesttet mit 1600, gute performnce
        "x20": 2000.0,

        "input_dim": 1,  # y
        "output_dim": 1  # u
    },
    "training_data_cfg" : {
        "batch_size": 100,
        "seq_len":    2001,
        "input_dim": 1,  # y
        "output_dim": 1,  # u
        "dt" : 0.01,
        "min_correlation_threshold": -1.1,
        "n_u": 2,
        "n_y": 2,

        "u_1_D_center_min": 0.6,
        "u_1_D_center_max": 0.9,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min": 0,
        "x_1_hard_max": None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.12,

        "u_1_p" : 0.5,
        "u_1_lambd" : 4
    },

    "train": {
        # Hyperparameters to be constant
        "device": "cuda", 
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
        "k_folds": 2,
        "lr": 1e-3,

        "mini_batch_size": 1,

        # Hyperparameters for tuning
        "epochs": 100,
        "min_correlation_threshold": -1.1,
        "constant_signal_probability": 0.0,
        "n_u": 2,
        "n_y": 2,
        
        "test_patience_epochs": 3,
        "test_min_delta": 0.0001,
        "lookback_offset": 20
    },

    "mamba": {
        "expand": 9,
        "d_state": 31,
        "d_conv": 9
    },

    "esn": {
            "units": 200,   
            "lr": 0.5,
            "sr": 0.9,
            "ridge": 1e-7,    # Regularization coefficient  
        },

    "mamba_param_space" : {
    "mamba.d_conv":  {"type": "int", "low": 1, "high": 10},
    "mamba.d_state": {"type": "int", "low": 1, "high": 64},
    "mamba.expand":  {"type": "int", "low": 1, "high": 10},
    },

    "transformer": {
        "nhead" : 2,
        "num_layers" : 6,
        "dim_feedforward" : 256,
        "max_seq_len" : 2000
    },

    "lstm":{
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.1

    },
    "transformer_param_space":  {
    "transformer.nhead":           {"type": "categorical", "choices": [1, 2, 3]}, # Must divide d_model
    "transformer.num_layers":      {"type": "int", "low": 1, "high": 4},
    "transformer.dim_feedforward": {"type": "categorical", "choices": [64, 128, 256]},
    },

    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,
    }
}