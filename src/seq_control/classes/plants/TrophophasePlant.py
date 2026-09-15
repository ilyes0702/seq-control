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
        x1_init, x2_init = self.hyperparam_config["plant"]["initial_state"]
        
        if randomize:
            # Randomization formula: nominal * (0.95 + 0.1 * rand) -> [0.95*nominal, 1.05*nominal)
            x1_init = x1_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x2_init = x2_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
        else:
            # Create tensors filled entirely with the nominal values
            x1_init = torch.full((batch_size, 1), x1_init, device=self.device, dtype=torch.float32)
            x2_init = torch.full((batch_size, 1), x2_init, device=self.device, dtype=torch.float32)
            
        x_init = torch.cat([x1_init, x2_init], 
                        dim=1)

        # Ensure no negative values
        return torch.clamp(x_init, min=0.0)

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

    def step(self, state, u, t):
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
        dt = self.dt
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
        # Source of model parameters: Rothfuß, R. (1997). Anwendung der flachheitsbasierten Analyse und Regelung nichtlinearer Mehrgrößensysteme (Als Ms. gedr). VDI-Verl.
        "mu_max": 0.12,                 # Maximum growth rate [1/h]
        "Ks": 50,                       # Affinity constant [(mg S)/L]
        "m_S": 23.0,                    # Maintenance coefficient [(mg S)/(g TS h)]
        "p1": 0.00047,                  # Yield coefficient of substrate [(g TS)/(mg S)]
        "p2": 200000.0,                 # Feed concentration of substrate [(mg S)/L]
        
        "initial_state": [
                        1500.0, 
                        2000.0
                        ],
    },

    "training_data_cfg" : {
        "batch_size": 100,
        "dt" : 0.01,
        "seq_len":    2001,

        "input_dim": 1,                     # Number of plant control inputs
        "output_dim": 1,                    # Number of plant outputs
        
        "min_correlation_threshold": -1.1,  # Minimum value of Pearson's correlation coefficient between input and output
        "nu_u": 2,
        "nu_y": 2,

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
        "k_folds": 2,
        "epochs": 100,
        "lr": 1e-3,
        "device": "cuda", 
        "mini_batch_size": 1,

        "loss_function": "MSELoss",    
        "patience_epochs": 3,    
        "patience_min_improvement": 0.0001,
        "min_correlation_threshold": -1.1,

        "nu_u": 2,
        "nu_y": 2,
    },

    # Default hyperparameters for Mamba-based sequence models
    "mamba": {
            "d_state": 31,                      
            "expand": 9,
            "d_conv" : 9
        },

    # Hyperparameter space of the Mamba sequence model for hyperparameter tuning via Optuna
        "mamba_param_space" : {
            "mamba.d_conv":  {"type": "int", "low": 1, "high": 10},
            "mamba.d_state": {"type": "int", "low": 1, "high": 64},
            "mamba.expand":  {"type": "int", "low": 1, "high": 10},
            },

    # Default hyperparameters for LSTM-based sequence model
    "lstm": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
    # Hyperparameter space of the LSTM sequence model for hyperparameter tuning via Optuna
    "lstm_param_space" : {
        "lstm.hidden_size": {"type": "int", "low": 16, "high": 128},
        "lstm.num_layers": {"type": "int", "low": 1, "high": 4},
        "lstm.dropout": {"type": "float", "low": 0.0, "high": 0.5},
    },

    # Default hyperparameters for Transformer-based sequence model 
    "transformer": {
                    "nhead" : 2,
                    "num_layers" : 6,
                    "dim_feedforward" : 256,
                    "max_seq_len" : 2000
                },
    # Hyperparameter space of the Transformer sequence model for hyperparameter tuning via Optuna                
    "transformer_param_space":  {
        "transformer.nhead":           {"type": "categorical", "choices": [1, 2, 3]}, # Must divide d_model
        "transformer.num_layers":      {"type": "int", "low": 1, "high": 4},
        "transformer.dim_feedforward": {"type": "categorical", "choices": [64, 128, 256]},
        },

    # Default hyperparameters for ESN-based sequence model 
    "esn": {
        "units": 200,   
        "lr": 0.5,
        "sr": 0.9,
        "ridge": 1e-7,    # Regularization coefficient  
    },

    "validation_trajectories" : {
            "batch_size": 10,
            "seq_len"   : 2001,
            
            "set_point" : [0.015],
            "amplitude" : [0.004],
            "period"    : [20.0],
    
            "y_start"   : [0.12],
            "y_target"  : [0.015],
            "tau"       : [0.5]         
        }
}