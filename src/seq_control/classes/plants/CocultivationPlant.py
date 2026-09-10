import torch

class CoCultivationPlant:
    """Two-strain microbial consortium simulation environment in a chemostat.

    Implements a PyTorch-based batch simulator for a two-strain microbial 
    community under optogenetic control. System dynamics incorporate 
    optogenetically-modulated Monod growth kinetics, enzyme synthesis driven by 
    light inputs, substrate depletion, and continuous dilution in a chemostat. 
    State integration is performed using 4th-order explicit Runge-Kutta (RK4).

    :param hyperparam_config: Configuration dictionary containing plant kinetics,
        bioprocess parameters, device selection, and simulation time step settings.
    :type hyperparam_config: dict
    """
    def __init__(self, hyperparam_config):
        """Initialize kinetic constants, bioprocess parameters, and execution device.

        :param hyperparam_config: Nested configuration dictionary structured as:

            * **train**: ``{"device": str or torch.device}``
            * **signal**: ``{"dt": float}``
            * **plant**: Kinetic parameters (``mu_max1``, ``mu_max2``, ``k_g_1``, 
              ``k_g_2``, ``f_c``, ``k_a_1``, ``k_a_2``, ``Y_g_b1``, ``Y_g_b2``, 
              ``q_a_max_1``, ``q_a_max_2``, ``n_1``, ``k_I_1``, ``n_2``, ``k_I_2``, 
              ``d_l``, ``S_in``, ``x10``, ``x20``, ``s0``, ``a10``, ``a20``).
        :type hyperparam_config: dict
        """
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]
        self.plant_cfg = hyperparam_config["plant"]

        # Kinetic and Bioprocess Parameters
        self.mu_max1 = self.plant_cfg["mu_max1"]
        self.mu_max2 = self.plant_cfg["mu_max2"]
        self.k_g_1 = self.plant_cfg["k_g_1"]
        self.k_g_2 = self.plant_cfg["k_g_2"]
        self.f_c = self.plant_cfg["f_c"]
        self.k_a_1 = self.plant_cfg["k_a_1"]
        self.k_a_2 = self.plant_cfg["k_a_2"]
        self.Y_g_b1 = self.plant_cfg["Y_g_b1"]
        self.Y_g_b2 = self.plant_cfg["Y_g_b2"]
        self.q_a_max_1 = self.plant_cfg["q_a_max_1"]
        self.q_a_max_2 = self.plant_cfg["q_a_max_2"]
        self.n_1 = self.plant_cfg["n_1"]
        self.k_I_1 = self.plant_cfg["k_I_1"]
        self.n_2 = self.plant_cfg["n_2"]
        self.k_I_2 = self.plant_cfg["k_I_2"]
        self.d_l = self.plant_cfg["d_l"]
        self.S_in = self.plant_cfg["S_in"]
        self.d_a_1 = 150
        self.d_a_2 = 150

        self.hyperparam_config = hyperparam_config

    def get_initial_state(self, batch_size, randomize=True):
        """Construct the initial state tensor for batched simulations.

        The 5-element state vector consists of:
        $[X_1, X_2, S, A_1, A_2]$ where $X_1, X_2$ are strain biomasses, 
        $S$ is substrate concentration, and $A_1, A_2$ are enzyme concentrations.

        :param batch_size: Number of parallel simulation trajectories in the batch.
        :type batch_size: int
        :param randomize: If ``True``, applies uniform random noise within a $\\pm 1\\%$ 
            to $\\pm 3\\%$ range (scaled by $[0.99, 1.01]$) to state initializations, 
            defaults to ``True``.
        :type randomize: bool, optional
        :returns: Initialized state tensor of shape ``(batch_size, 5)``.
        :rtype: torch.Tensor
        """
        # Fetch nominal values from config or defaults
        x1_nom = self.plant_cfg["x10"]
        x2_nom = self.plant_cfg["x20"]
        s_nom  = self.plant_cfg["s0"]
        a1_nom = self.plant_cfg["a10"]
        a2_nom = self.plant_cfg["a20"]

        nominal = torch.tensor([x1_nom, x2_nom, s_nom, a1_nom, a2_nom], 
                               device=self.device, dtype=torch.float32)
        
        # Broadcast across batch dimension
        states = nominal.repeat(batch_size, 1)

        if randomize:
            # Randomization within ±5% boundaries
            rand_scale = 0.99 + 0.02 * torch.rand((batch_size, 5), device=self.device)
            states = states * rand_scale
            
        return states

    def get_y(self, state, t=None):
        """Extract monitored output variables from the system state vector.

        :param state: Current state tensor of shape ``(batch_size, 5)``.
        :type state: torch.Tensor
        :param t: Current simulation time step, defaults to ``None``.
        :type t: float or torch.Tensor, optional
        :returns: Monitored tracking biomass outputs $[X_1, X_2]$ of shape ``(batch_size, 2)``.
        :rtype: torch.Tensor
        """
        return state[:, 0:2] # Returns [X1, X2]

    def dynamics(self, X1, X2, S, A1, A2, u, t):
        """Compute the continuous-time state derivatives of the co-cultivation system.

        Evaluates Monod growth kinetics modulated by enzyme concentrations, Hill-type 
        light-induced enzyme expression, substrate consumption, and chemostat inflow/outflow.

        :param X1: Strain 1 biomass tensor of shape ``(batch_size, 1)``.
        :type X1: torch.Tensor
        :param X2: Strain 2 biomass tensor of shape ``(batch_size, 1)``.
        :type X2: torch.Tensor
        :param S: Substrate concentration tensor of shape ``(batch_size, 1)``.
        :type S: torch.Tensor
        :param A1: Strain 1 optogenetic enzyme concentration tensor of shape ``(batch_size, 1)``.
        :type A1: torch.Tensor
        :param A2: Strain 2 optogenetic enzyme concentration tensor of shape ``(batch_size, 1)``.
        :type A2: torch.Tensor
        :param u: Light actuation inputs $[I_1, I_2]$ of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current evaluation time.
        :type t: float or torch.Tensor
        :returns: Tuple of continuous derivative tensors 
            $(dX_1/dt, dX_2/dt, dS/dt, dA_1/dt, dA_2/dt)$, each of shape ``(batch_size, 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # u expected shape: [batch_size, 2] -> [I_1, I_2]
        I_1 = u[:, 0:1]
        I_2 = u[:, 1:2]

        # Optogenetically-modulated Monod Kinetics
        mu1 = ((self.mu_max1 * S) / (self.k_g_1 + S)) * ((self.f_c * A1) / (self.f_c * A1 + self.k_a_1))
        mu2 = ((self.mu_max2 * S) / (self.k_g_2 + S)) * ((self.f_c * A2) / (self.f_c * A2 + self.k_a_2))
        
        q_g_1 = self.Y_g_b1 * mu1
        q_g_2 = self.Y_g_b2 * mu2

        # Enzyme synthesis rates via light inputs
        q_a_1 = self.q_a_max_1 * (I_1**self.n_1 / (I_1**self.n_1 + self.k_I_1**self.n_1))
        q_a_2 = self.q_a_max_2 * (I_2**self.n_2 / (I_2**self.n_2 + self.k_I_2**self.n_2))

        # Chemostat system differential math
        dX1_dt = (mu1 - self.d_l) * X1
        dX2_dt = (mu2 - self.d_l) * X2
        dS_dt  = self.d_l * (self.S_in - S) - q_g_1 * X1 - q_g_2 * X2
        dA1_dt = q_a_1 - (self.d_a_1 + mu1) * A1
        dA2_dt = q_a_2 - (self.d_a_2 + mu2) * A2

        return dX1_dt, dX2_dt, dS_dt, dA1_dt, dA2_dt

    def step(self, state, u, t, dt=None):
        """Perform dynamic state integration using the Dormand-Prince (RK45) scheme.

        Advances the 5-state co-cultivation system forward across time step ``dt`` using 
        a 5th-order explicit Dormand-Prince numerical integration step.

        :param state: Current state tensor of shape ``(batch_size, 5)``.
        :type state: torch.Tensor
        :param u: Applied optogenetic control input tensor $[I_1, I_2]$ of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :param dt: Time increment step size. If ``None``, defaults to ``self.dt``.
        :type dt: float or torch.Tensor, optional
        :returns: Tuple ``(state_next, y_next)`` containing updated state tensor of shape 
            ``(batch_size, 5)`` and monitored biomass output tensor of shape ``(batch_size, 2)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if dt is None:
            dt = self.dt

        # Unpack state tensor into individual component columns
        X1, X2 = state[:, 0:1], state[:, 1:2]
        S      = state[:, 2:3]
        A1, A2 = state[:, 3:4], state[:, 4:5]

        # Butcher tableau coefficients for Dormand-Prince
        # Stage 1 (k1)
        dX1_1, dX2_1, dS_1, dA1_1, dA2_1 = self.dynamics(X1, X2, S, A1, A2, u, t)

        # Stage 2 (k2)
        X1_2 = X1 + dt * (1/5 * dX1_1)
        X2_2 = X2 + dt * (1/5 * dX2_1)
        S_2  = S  + dt * (1/5 * dS_1)
        A1_2 = A1 + dt * (1/5 * dA1_1)
        A2_2 = A2 + dt * (1/5 * dA2_1)
        dX1_2, dX2_2, dS_2, dA1_2, dA2_2 = self.dynamics(X1_2, X2_2, S_2, A1_2, A2_2, u, t + 0.2 * dt)

        # Stage 3 (k3)
        X1_3 = X1 + dt * (3/40 * dX1_1 + 9/40 * dX1_2)
        X2_3 = X2 + dt * (3/40 * dX2_1 + 9/40 * dX2_2)
        S_3  = S  + dt * (3/40 * dS_1  + 9/40 * dS_2)
        A1_3 = A1 + dt * (3/40 * dA1_1 + 9/40 * dA1_2)
        A2_3 = A2 + dt * (3/40 * dA2_1 + 9/40 * dA2_2)
        dX1_3, dX2_3, dS_3, dA1_3, dA2_3 = self.dynamics(X1_3, X2_3, S_3, A1_3, A2_3, u, t + 0.3 * dt)

        # Stage 4 (k4)
        X1_4 = X1 + dt * (44/45 * dX1_1 - 56/15 * dX1_2 + 32/9 * dX1_3)
        X2_4 = X2 + dt * (44/45 * dX2_1 - 56/15 * dX2_2 + 32/9 * dX2_3)
        S_4  = S  + dt * (44/45 * dS_1  - 56/15 * dS_2  + 32/9 * dS_3)
        A1_4 = A1 + dt * (44/45 * dA1_1 - 56/15 * dA1_2 + 32/9 * dA1_3)
        A2_4 = A2 + dt * (44/45 * dA2_1 - 56/15 * dA2_2 + 32/9 * dA2_3)
        dX1_4, dX2_4, dS_4, dA1_4, dA2_4 = self.dynamics(X1_4, X2_4, S_4, A1_4, A2_4, u, t + 0.8 * dt)

        # Stage 5 (k5)
        X1_5 = X1 + dt * (19372/6561 * dX1_1 - 25360/2187 * dX1_2 + 64448/6561 * dX1_3 - 212/729 * dX1_4)
        X2_5 = X2 + dt * (19372/6561 * dX2_1 - 25360/2187 * dX2_2 + 64448/6561 * dX2_3 - 212/729 * dX2_4)
        S_5  = S  + dt * (19372/6561 * dS_1  - 25360/2187 * dS_2  + 64448/6561 * dS_3  - 212/729 * dS_4)
        A1_5 = A1 + dt * (19372/6561 * dA1_1 - 25360/2187 * dA1_2 + 64448/6561 * dA1_3 - 212/729 * dA1_4)
        A2_5 = A2 + dt * (19372/6561 * dA2_1 - 25360/2187 * dA2_2 + 64448/6561 * dA2_3 - 212/729 * dA2_4)
        dX1_5, dX2_5, dS_5, dA1_5, dA2_5 = self.dynamics(X1_5, X2_5, S_5, A1_5, A2_5, u, t + (8/9) * dt)

        # Stage 6 (k6)
        X1_6 = X1 + dt * (9017/3168 * dX1_1 - 355/33 * dX1_2 + 46732/5247 * dX1_3 + 49/176 * dX1_4 - 5103/18656 * dX1_5)
        X2_6 = X2 + dt * (9017/3168 * dX2_1 - 355/33 * dX2_2 + 46732/5247 * dX2_3 + 49/176 * dX2_4 - 5103/18656 * dX2_5)
        S_6  = S  + dt * (9017/3168 * dS_1  - 355/33 * dS_2  + 46732/5247 * dS_3  + 49/176 * dS_4  - 5103/18656 * dS_5)
        A1_6 = A1 + dt * (9017/3168 * dA1_1 - 355/33 * dA1_2 + 46732/5247 * dA1_3 + 49/176 * dA1_4 - 5103/18656 * dA1_5)
        A2_6 = A2 + dt * (9017/3168 * dA2_1 - 355/33 * dA2_2 + 46732/5247 * dA2_3 + 49/176 * dA2_4 - 5103/18656 * dA2_5)
        dX1_6, dX2_6, dS_6, dA1_6, dA2_6 = self.dynamics(X1_6, X2_6, S_6, A1_6, A2_6, u, t + dt)

        # 5th-order accurate state update
        X1_next = X1 + dt * (35/384 * dX1_1 + 500/1113 * dX1_3 + 125/192 * dX1_4 - 2187/6784 * dX1_5 + 11/84 * dX1_6)
        X2_next = X2 + dt * (35/384 * dX2_1 + 500/1113 * dX2_3 + 125/192 * dX2_4 - 2187/6784 * dX2_5 + 11/84 * dX2_6)
        S_next  = S  + dt * (35/384 * dS_1  + 500/1113 * dS_3  + 125/192 * dS_4  - 2187/6784 * dS_5  + 11/84 * dS_6)
        A1_next = A1 + dt * (35/384 * dA1_1 + 500/1113 * dA1_3 + 125/192 * dA1_4 - 2187/6784 * dA1_5 + 11/84 * dA1_6)
        A2_next = A2 + dt * (35/384 * dA2_1 + 500/1113 * dA2_3 + 125/192 * dA2_4 - 2187/6784 * dA2_5 + 11/84 * dA2_6)

        # Re-pack and apply non-negativity constraint
        state_next = torch.cat([X1_next, X2_next, S_next, A1_next, A2_next], dim=1)
        state_next = torch.clamp(state_next, min=0.0)

        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        """Return plotting configuration metadata for trajectory visualizers.

        Provides group specifications, signal identifiers, LaTeX axis labels, and 
        variable groupings for plotting states, tracked outputs, and control inputs.

        :returns: List of dictionary specifications defining plot panels and LaTeX formatting.
        :rtype: list[dict]
        """
        return [
            {
                "cols": ["x1", "x2", "s", "a1", "a2"],
                "labels": [r"$x_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_2 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_3 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_4 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_5 / \mathrm{g}\,\mathrm{L}^{-1}$"],
                "ylabel": [r"$x_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_2 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_3 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_4 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$x_5 / \mathrm{g}\,\mathrm{L}^{-1}$"]
            },
            {
                "cols": ["y1", "y2"],
                "labels": [r"$y_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$y_2 / \mathrm{g}\,\mathrm{L}^{-1}$"],
                "ylabel": [r"$y_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$y_2 / \mathrm{g}\,\mathrm{L}^{-1}$"]
            },
            {
                "cols": ["u1", "u2"],
                "labels": [r"$u_1 / \mathrm{W}\,\mathrm{m}^{-2}$", r"$u_2 / \mathrm{W}\,\mathrm{m}^{-2}$"],
                "ylabel": [r"$u_1 / \mathrm{W}\,\mathrm{m}^{-2}$", r"$u_2 / \mathrm{W}\,\mathrm{m}^{-2}$"]
            }
        ]    

# Default hyperparameter configuration
hyperparam_config_CoCultivationPlant = {
    "plant": {
        # Kinetic Parameters for Strain 1 & Strain 2
        "mu_max1": 0.982,       # Max growth rate strain 1 (1/h)
        "mu_max2": 0.982,       # Max growth rate strain 2 (1/h)
        "k_g_1": 2.964e-4,      # Substrate affinity constant strain 1
        "k_g_2": 2.964e-4,      # Substrate affinity constant strain 2
        "f_c": 1100.0,          # Conversion factor scaling enzyme concentration
        "k_a_1": 1.7,           # Activation constant for strain 1 growth
        "k_a_2": 0.182,         # Activation constant for strain 2 growth
        "Y_g_b1": 10.18,        # Yield coefficient factor for strain 1
        "Y_g_b2": 10.18,        # Yield coefficient factor for strain 2
        "q_a_max_1": 0.337,     # Max enzyme expression rate via light 1
        "q_a_max_2": 0.036,     # Max enzyme expression rate via light 2
        "n_1": 2.0,             # Hill coefficient for light input 1
        "k_I_1": 1.052,         # Light intensity constant for induction 1
        "n_2": 4.865,           # Hill coefficient for light input 2
        "k_I_2": 1.34,          # Light intensity constant for induction 2
        "d_l": 0.15,            # Dilution rate of the chemostat (1/h)
        "S_in": 200.0,          # Substrate concentration in the feed (g/L)
        "d_a_1": 0.15,           # Enzyme degradation rate 1 # unkown
        "d_a_2": 0.15,           # Enzyme degradation rate 2 #unknown
        
        # Signal generation center boundaries for Strain 1 (Channel 1)
        "u_1_D_center_min": 0.5,   # Adjust these values based on your light intensity needs
        "u_1_D_center_max": 2.0,   
        
        # Signal generation center boundaries for Strain 2 (Channel 2)
        "u_2_D_center_min": 0.5,   # Adjust these values based on your light intensity needs
        "u_2_D_center_max": 2.0,
        
        # Operational limits / bounds (Adjust boundaries based on your light units/caps)
        "u_1_hard_min": 0.0,
        "u_1_hard_max": 5.0,      # Max expected light intensity cap 
        
        "u_2_hard_min": 0.0,
        "u_2_hard_max": 5.0,      # Max expected light
        # Initial conditions (nominal state values)
        "x10": 0.005,           # Biomass X1 (g/L)
        "x20": 0.005,           # Biomass X2 (g/L)
        "s0": 1.0,              # Substrate S (g/L)
        "a10": 1.545e-2,        # Enzyme concentration A1
        "a20": 1.655e-3,        # Enzyme concentration A2

        # IO Dimensions: 2 controlled tracker variables (X1, X2), 2 actuators (I1, I2)
        "input_dim": 2,         # Tracker dimension (y)
        "output_dim": 2         # Actuator control dimension (u)
    },
    "signal": {
        "lambd": 4,
        "p": 0.5,
        "seq_len": 2001,
        "dt": 0.01               # Matching the dt=1 step time from your original code
    },
    "train": {
        "k_folds": 2,
        "epochs": 100,
        "batch_size": 1000,
        "lr": 1e-3,
        "device": "cuda",       # Automatically falls back to device selection patterns
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate": 1,
        "min_correlation_threshold": -1.1,
        "test_patience_epochs": 3,
        "test_min_delta": 0.0001,
        "n_y" : 2,
        "n_u" : 2,
        "mini_batch_size": 1
    },
    "training_data_cfg": {
        "batch_size": 100, 
        "seq_len": 2001,
        "input_dim": 2,        
        "output_dim": 2,         
        "dt" : 0.01,
        "min_correlation_threshold": -1.1,
        "n_u": 2,
        "n_y": 2,

        "u_1_D_center_min": 0.5,
        "u_1_D_center_max": 2.0,

        "u_2_D_center_min": 0.5,
        "u_2_D_center_max": 2.0,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 5.0,

        "u_2_hard_min": 0.0,
        "u_2_hard_max": 5.0,

        "x_1_hard_min": 0.0,
        "x_1_hard_max": None,

        "x_2_hard_min": 0.0,
        "x_2_hard_max": None,

        "y_1_hard_min": 0.0,
        "y_1_hard_max": None,

        "y_2_hard_min": 0.0,
        "y_2_hard_max": None,

        "u_1_p" : 0.5,
        "u_1_lambd" : 4,
        
        "u_2_p" : 0.5,
        "u_2_lambd" : 4,
        

        
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