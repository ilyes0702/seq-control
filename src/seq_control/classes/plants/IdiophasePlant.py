import torch


class IdiophasePlant:
    """Idiophase bioprocess plant simulation model.

    Models the secondary metabolite production phase (idiophase) of a bioprocess
    system using a 4-state differential equation model integrated via a 5th-order
    Dormand-Prince (RK45) Runge-Kutta scheme.

    :param hyperparam_config: Configuration dictionary containing hardware ('train'),
        signal ('dt'), and kinetic/bioprocess parameters ('plant').
    :type hyperparam_config: dict
    """

    def __init__(self, hyperparam_config):
        """Initialize plant parameters, device placement, and constants from config.

        :param hyperparam_config: Nested dictionary with system configuration parameters.
        :type hyperparam_config: dict
        """
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Biological and Plant Parameters from Context
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu_max"], device=self.device) # e.g., 0.12
        self.Ks = torch.tensor(hyperparam_config["plant"]["Ks"], device=self.device)         # e.g., 50
        self.p1 = torch.tensor(hyperparam_config["plant"]["p1"], device=self.device)         # e.g., 0.00047
        self.p2 = torch.tensor(hyperparam_config["plant"]["p2"], device=self.device)         # e.g., 200000
        self.p5 = torch.tensor(hyperparam_config["plant"]["p5"], device=self.device)         # e.g., 0.9
        self.p6 = torch.tensor(hyperparam_config["plant"]["p6"], device=self.device)         # e.g., 100
        self.p7 = torch.tensor(hyperparam_config["plant"]["p7"], device=self.device)         # e.g., 0.04
        self.q = torch.tensor(hyperparam_config["plant"]["q"], device=self.device)           # e.g., 2000
        self.mu_Pen = torch.tensor(hyperparam_config["plant"]["mu_Pen"], device=self.device) # e.g., 3
        self.m_S = torch.tensor(hyperparam_config["plant"]["m_S"], device=self.device)

        self.hyperparam_config = hyperparam_config

        # Fixed volume for Idiophase if specified as constant
        self.V_const = torch.tensor(hyperparam_config["plant"].get("V_idiophase", 170.0), device=self.device)

    def get_V(self, t):
        """Retrieve reactor volume V(t) at simulation time ``t``.

        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :returns: Volume tensor matching input time horizon shape or scalar constant.
        :rtype: torch.Tensor
        """
        if torch.is_tensor(t):
            return torch.full_like(t, self.V_const, device=self.device)
        return self.V_const

    def get_initial_state(self, batch_size):
        """Construct initial state tensor for a batch of simulation instances.

        :param batch_size: Number of parallel simulation trajectories.
        :type batch_size: int
        :returns: Initial state tensor of shape ``(batch_size, 4)``.
        :rtype: torch.Tensor
        """
        x1_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x10"], device=self.device)
        x2_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x20"], device=self.device)
        x3_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x30"], device=self.device)
        x4_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x40"], device=self.device)

        return torch.cat([x1_init, x2_init, x3_init, x4_init], dim=1)

    def get_y(self, state, t):
        """Calculate 2-dimensional MIMO tracking output vector y = [y1, y2].

        y1 represents specific growth rate (mu) and y2 represents precursor concentration (c3 = x3 / V).

        :param state: Current state tensor of shape ``(batch_size, 4)``.
        :type state: torch.Tensor
        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :returns: Observable output tracking vector tensor of shape ``(batch_size, 2)``.
        :rtype: torch.Tensor
        """
        x2 = state[:, 1:2]
        x3 = state[:, 2:3]
        V = self.get_V(t)

        # Monod growth kinetics equation
        mu = (self.mu_max * x2) / (self.Ks * V + x2)

        # Precursor Concentration
        c3 = x3 / V

        return torch.cat([mu, c3], dim=1)

    def dynamics(self, x1, x2, x3, x4, u, t):
        """Compute state derivative vector dx/dt = f(x, u, t) for the idiophase system.

        :param x1: Biomass state tensor of shape ``(batch_size, 1)``.
        :type x1: torch.Tensor
        :param x2: Substrate state tensor of shape ``(batch_size, 1)``.
        :type x2: torch.Tensor
        :param x3: Precursor mass state tensor of shape ``(batch_size, 1)``.
        :type x3: torch.Tensor
        :param x4: Product mass state tensor of shape ``(batch_size, 1)``.
        :type x4: torch.Tensor
        :param u: Applied control inputs tensor [u1, u2] of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :returns: Tuple (dx1dt, dx2dt, dx3dt, dx4dt) containing continuous derivatives.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        V = self.get_V(t)

        # Monod growth kinetics
        mu = (self.mu_max * x2) / (self.Ks * V + x2)

        # System differential state equations
        dx1dt = mu * x1
        dx2dt = - (1.0 / self.p1) * mu * x1 - (1.0 / self.p5) * self.mu_Pen * x1 - self.m_S * x1 + self.p2 * u1
        dx3dt = - (1.0 / self.q) * self.mu_Pen * x1 + self.p6 * u2
        dx4dt = self.mu_Pen * x1 - self.p7 * x4

        return dx1dt, dx2dt, dx3dt, dx4dt

    def step(self, state, u, t, dt=None):
        """Perform dynamic state integration using the Dormand-Prince (RK45) scheme.

        Advances the 4-state idiophase system forward across time step ``dt`` using 
        a 5th-order explicit Dormand-Prince numerical integration step.

        :param state: Current state tensor of shape ``(batch_size, 4)``.
        :type state: torch.Tensor
        :param u: Applied control inputs tensor [u1, u2] of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :param dt: Time increment step size. If ``None``, defaults to ``self.dt``.
        :type dt: float or torch.Tensor, optional
        :returns: Tuple ``(state_next, y_next)`` containing updated state tensor of shape 
            ``(batch_size, 4)`` and observable output tensor of shape ``(batch_size, 2)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if dt is None:
            dt = self.dt

        # Unpack state tensor components
        x1, x2, x3, x4 = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]

        # Butcher tableau coefficients for Dormand-Prince
        # Stage 1 (k1)
        dx1_1, dx2_1, dx3_1, dx4_1 = self.dynamics(x1, x2, x3, x4, u, t)

        # Stage 2 (k2)
        x1_2 = x1 + dt * (1/5 * dx1_1)
        x2_2 = x2 + dt * (1/5 * dx2_1)
        x3_2 = x3 + dt * (1/5 * dx3_1)
        x4_2 = x4 + dt * (1/5 * dx4_1)
        dx1_2, dx2_2, dx3_2, dx4_2 = self.dynamics(x1_2, x2_2, x3_2, x4_2, u, t + 0.2 * dt)

        # Stage 3 (k3)
        x1_3 = x1 + dt * (3/40 * dx1_1 + 9/40 * dx1_2)
        x2_3 = x2 + dt * (3/40 * dx2_1 + 9/40 * dx2_2)
        x3_3 = x3 + dt * (3/40 * dx3_1 + 9/40 * dx3_2)
        x4_3 = x4 + dt * (3/40 * dx4_1 + 9/40 * dx4_2)
        dx1_3, dx2_3, dx3_3, dx4_3 = self.dynamics(x1_3, x2_3, x3_3, x4_3, u, t + 0.3 * dt)

        # Stage 4 (k4)
        x1_4 = x1 + dt * (44/45 * dx1_1 - 56/15 * dx1_2 + 32/9 * dx1_3)
        x2_4 = x2 + dt * (44/45 * dx2_1 - 56/15 * dx2_2 + 32/9 * dx2_3)
        x3_4 = x3 + dt * (44/45 * dx3_1 - 56/15 * dx3_2 + 32/9 * dx3_3)
        x4_4 = x4 + dt * (44/45 * dx4_1 - 56/15 * dx4_2 + 32/9 * dx4_3)
        dx1_4, dx2_4, dx3_4, dx4_4 = self.dynamics(x1_4, x2_4, x3_4, x4_4, u, t + 0.8 * dt)

        # Stage 5 (k5)
        x1_5 = x1 + dt * (19372/6561 * dx1_1 - 25360/2187 * dx1_2 + 64448/6561 * dx1_3 - 212/729 * dx1_4)
        x2_5 = x2 + dt * (19372/6561 * dx2_1 - 25360/2187 * dx2_2 + 64448/6561 * dx2_3 - 212/729 * dx2_4)
        x3_5 = x3 + dt * (19372/6561 * dx3_1 - 25360/2187 * dx3_2 + 64448/6561 * dx3_3 - 212/729 * dx3_4)
        x4_5 = x4 + dt * (19372/6561 * dx4_1 - 25360/2187 * dx4_2 + 64448/6561 * dx4_3 - 212/729 * dx4_4)
        dx1_5, dx2_5, dx3_5, dx4_5 = self.dynamics(x1_5, x2_5, x3_5, x4_5, u, t + (8/9) * dt)

        # Stage 6 (k6)
        x1_6 = x1 + dt * (9017/3168 * dx1_1 - 355/33 * dx1_2 + 46732/5247 * dx1_3 + 49/176 * dx1_4 - 5103/18656 * dx1_5)
        x2_6 = x2 + dt * (9017/3168 * dx2_1 - 355/33 * dx2_2 + 46732/5247 * dx2_3 + 49/176 * dx2_4 - 5103/18656 * dx2_5)
        x3_6 = x3 + dt * (9017/3168 * dx3_1 - 355/33 * dx3_2 + 46732/5247 * dx3_3 + 49/176 * dx3_4 - 5103/18656 * dx3_5)
        x4_6 = x4 + dt * (9017/3168 * dx4_1 - 355/33 * dx4_2 + 46732/5247 * dx4_3 + 49/176 * dx4_4 - 5103/18656 * dx4_5)
        dx1_6, dx2_6, dx3_6, dx4_6 = self.dynamics(x1_6, x2_6, x3_6, x4_6, u, t + dt)

        # 5th-order accurate state update
        x1_next = x1 + dt * (35/384 * dx1_1 + 500/1113 * dx1_3 + 125/192 * dx1_4 - 2187/6784 * dx1_5 + 11/84 * dx1_6)
        x2_next = x2 + dt * (35/384 * dx2_1 + 500/1113 * dx2_3 + 125/192 * dx2_4 - 2187/6784 * dx2_5 + 11/84 * dx2_6)
        x3_next = x3 + dt * (35/384 * dx3_1 + 500/1113 * dx3_3 + 125/192 * dx3_4 - 2187/6784 * dx3_5 + 11/84 * dx3_6)
        x4_next = x4 + dt * (35/384 * dx4_1 + 500/1113 * dx4_3 + 125/192 * dx4_4 - 2187/6784 * dx4_5 + 11/84 * dx4_6)

        state_next = torch.cat([x1_next, x2_next, x3_next, x4_next], dim=1)

        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        """Return plot formatting metadata for visualizing simulation trajectories.

        :returns: List of metadata dictionaries defining column names, labels, and LaTeX y-axis titles.
        :rtype: list[dict]
        """
        return [
            {
                "cols": ["x1", "x2", "x3", "x4"],
                "labels": [
                    r"$x_1$ [$\mathrm{mg}$]", 
                    r"$x_2$ [$\mathrm{g}$]", 
                    r"$x_3$ [$\mathrm{g}$]", 
                    r"$x_4$ [$\mathrm{g}$]"
                ],
                "ylabel": [
                    r"$x_1$ [$\mathrm{mg}$]", 
                    r"$x_2$ [$\mathrm{g}$]", 
                    r"$x_3$ [$\mathrm{g}$]", 
                    r"$x_4$ [$\mathrm{g}$]"
                ]
            },
            {
                "cols": ["y1", "y2"],
                "labels": [
                    r"$y_1$ [$\mathrm{h}^{-1}$]", 
                    r"$y_2$ [$\mathrm{g}\,\mathrm{L}^{-1}$]"
                ],
                "ylabel": [
                    r"$y_1$ [$\mathrm{h}^{-1}$]", 
                    r"$y_2$ [$\mathrm{g}\,\mathrm{L}^{-1}$]"
                ]
            },
            {
                "cols": ["u1", "u2"],
                "labels": [
                    r"$u_1$ [$\mathrm{L}\,\mathrm{h}^{-1}$]", 
                    r"$u_2$ [$\mathrm{L}\,\mathrm{h}^{-1}$]"
                ],
                "ylabel": [
                    r"$u_1$ [$\mathrm{L}\,\mathrm{h}^{-1}$]", 
                    r"$u_2$ [$\mathrm{L}\,\mathrm{h}^{-1}$]"
                ]
            }
        ]

# Default hyperparameter configuration
hyperparam_config_IdiophasePlant = {
        "signal": {
            "seq_len": 2001,
            "dt": 0.01,
            #/ 🕹️ Channel 1 Signal Parameters (e.g., highly dynamic)
            "u_1_lambd": 4,        
            "u_1_p": 0.5,            
            
            #// 🕹️ Channel 2 Signal Parameters (e.g., highly filtered, slow moving)
            "u_2_lambd": 4,        
            "u_2_p": 0.5,
        
        },
        "train": {
            "batch_size": 1000,
            "device": "cuda",
            "delay_steps": 1,
            "epochs": 50,
            "lr": 1e-3,
            "loss_function": "MSELoss()",
            "k_folds": 5,
            "lr_decay_rate":1,
            "min_correlation_threshold": -1.1,
            "n_y": 2,
            "n_u": 2,
            "lookback_offset": 100,
            "test_min_epochs": 3,
            "test_min_delta": 0.00001,
            "test_patience_epochs": 3,
            "mini_batch_size": 1
        },
        "plant": {
            "mu_max": 0.12,
            "Ks": 50.0,
            "p1": 0.00047,
            "p2": 200000.0,
            "p5": 0.9,
            "p6": 100.0,
            "p7": 0.04,
            "q": 2000.0,
            "mu_Pen": 3.0,
            "V_idiophase": 170.0,
            "m_S": 23,

            "x10": 1500,
            "x20": 2000,
            "x30": 25,
            "x40": 1600,

            "x_1_hard_min": 0.0,
            "x_1_hard_max": None,

            "x_2_hard_min": 0.0,
            "x_2_hard_max": None,

            "x_3_hard_min": 0.0,
            "x_3_hard_max": None,

            "x_4_hard_min": 0.0,
            "x_4_hard_max": None,
            
            "u_1_hard_min": 0.0,
            "u_1_hard_max": 1.0,

            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1.0,

            "y_1_hard_min": 0.0,
            "y_1_hard_max": 0.12,

            "y_2_hard_min": 0.0,
            "y_2_hard_max": None,

            "u_1_D_center_min": 0.6,
            "u_1_D_center_max": 0.9,

            "u_2_D_center_min": 0.0,
            "u_2_D_center_max": 0.5,

            
            "input_dim": 2,  # y1, y2
            "output_dim": 2  # u1, u2
        },
        "training_data_generation_config": {
            "batch_size": 2000,
            "seq_len":    2001,
            "dt" : 0.01,
            "input_dim": 2,  # y
            "output_dim": 2,  # u
            "min_correlation_threshold": -1.1,

            "u_1_D_center_min": 0.6,
            "u_1_D_center_max": 0.9,

            "u_1_hard_min": 0.0,
            "u_1_hard_max": 1,

            "u_2_D_center_min": 0.0,
            "u_2_D_center_max": 0.5,

            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1,

            "x_1_hard_min": 0,
            "x_1_hard_max": None,

            "y_1_hard_min": 0,
            "y_1_hard_max": 0.12,

            "u_1_p" : 0.5,
            "u_1_lambd" : 4,

            "u_2_p" : 0.5,
            "u_2_lambd" : 4,            
        },
        "training_data_cfg" : {
            "batch_size": 100,
            "seq_len": 2001,
            "dt": 0.01,
            "min_correlation_threshold": -1.1,
            "delay_steps": 1,
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

            "input_dim": 2,  # y
            "output_dim": 2,  # u

            "u_1_p" : 0.5,
            "u_1_lambd" : 4,

            "u_2_p" : 0.5,
            "u_2_lambd" : 4,

            "u_2_D_center_min": 0.0,
            "u_2_D_center_max": 0.5,

            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1,
        },
        "mamba": {
                "d_state": 1,
                "expand": 1,
                "d_conv" : 1
            },
        "simulate": {
            "batch_size": 10,
            "seq_len": 2001,
        }
    }