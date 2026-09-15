import torch


class IndForProteinProductionPlant:
    """Inducible recombinant protein production bioprocess plant simulation model.

    Models a 7-state fed-batch fermentation process for recombinant protein 
    production upon chemical induction, integrated via a batch-parallel vectorized 
    adaptive-step Dormand-Prince (RK45) integration scheme.

    :param hyperparam_config: Configuration dictionary containing device ('train'),
        time step ('signal'), and kinetic/bioprocess parameters ('plant').
    :type hyperparam_config: dict
    """

    def __init__(self, hyperparam_config):
        """Initialize plant constants, kinetic parameters, and device placement.

        :param hyperparam_config: Nested configuration dictionary.
        :type hyperparam_config: dict
        """
        self.hyperparam_config = hyperparam_config
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["training_data_cfg"]["dt"]

        # Constants and parameters from plant model definition
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu_max"], device=self.device)
        self.K_CI = torch.tensor(hyperparam_config["plant"]["K_CI"], device=self.device)
        self.k_22 = torch.tensor(hyperparam_config["plant"]["k_22"], device=self.device)
        
        self.K_s = torch.tensor(hyperparam_config["plant"]["K_s"], device=self.device)
        self.f_I_0 = torch.tensor(hyperparam_config["plant"]["f_I_0"], device=self.device)
        self.C_n_f = torch.tensor(hyperparam_config["plant"]["C_n_f"], device=self.device)
        self.Y = torch.tensor(hyperparam_config["plant"]["Y"], device=self.device)
        self.K_CN = torch.tensor(hyperparam_config["plant"]["K_CN"], device=self.device)
        self.k_11 = torch.tensor(hyperparam_config["plant"]["k_11"], device=self.device)
        self.K_IX = torch.tensor(hyperparam_config["plant"]["K_IX"], device=self.device)
        self.f_max = torch.tensor(hyperparam_config["plant"]["f_max"], device=self.device)
        
        self.K_I = torch.tensor(hyperparam_config["plant"]["K_I"], device=self.device)
        self.C_i_f = torch.tensor(hyperparam_config["plant"]["C_i_f"], device=self.device)

    def get_initial_state(self, batch_size, randomize=True):
        """Construct initial state tensor across a batch of simulation trajectories.

        State components vector layout:
        ``[x1 (Volume), x2 (Biomass), x3 (Glucose), x4 (Protein), x5 (Inducer), x6 (Shock), x7 (Recovery)]``

        :param batch_size: Number of parallel simulation trajectories in batch.
        :type batch_size: int
        :returns: Initial state tensor of shape ``(batch_size, 7)``.
        :rtype: torch.Tensor
        """

        x1_init, x2_init, x3_init, x4_init, x5_init, x6_init, x7_init = self.hyperparam_config["plant"]["initial_state"]    


        if randomize:
            # Randomization formula: nominal * (0.95 + 0.1 * rand) -> [0.95*nominal, 1.05*nominal)
            x1_init = x1_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x2_init = x2_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x3_init = x3_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x4_init = x4_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x5_init = x5_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))  
            x6_init = x6_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))  
            x7_init = x7_init * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))              
        else:
            # Create tensors filled entirely with the nominal values
            x1_init = torch.full((batch_size, 1), x1_init, device=self.device, dtype=torch.float32)
            x2_init = torch.full((batch_size, 1), x2_init, device=self.device, dtype=torch.float32)
            x3_init = torch.full((batch_size, 1), x3_init, device=self.device, dtype=torch.float32)
            x4_init = torch.full((batch_size, 1), x4_init, device=self.device, dtype=torch.float32)
            x5_init = torch.full((batch_size, 1), x5_init, device=self.device)
            x6_init = torch.full((batch_size, 1), x6_init, device=self.device)
            x7_init = torch.full((batch_size, 1), x7_init, device=self.device)
            
            
        x_init = torch.cat([x1_init, x2_init, x3_init, x4_init, x5_init, x6_init, x7_init], 
                        dim=1)

        # Ensure no negative values
        return torch.clamp(x_init, min=0.0)
        

    def get_y(self, state, t=None):
        """Extract monitored 3-dimensional controlled output vector y = [x1, x2, x4].

        Outputs correspond to reactor volume (x1), cell density/biomass (x2), and protein concentration (x4).

        :param state: Current state tensor of shape ``(batch_size, 7)``.
        :type state: torch.Tensor
        :param t: Current simulation time, defaults to ``None``.
        :type t: float or torch.Tensor, optional
        :returns: Monitored tracking vector tensor of shape ``(batch_size, 3)``.
        :rtype: torch.Tensor
        """
        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        x4 = state[:, 3:4]
        return torch.cat([x1, x2, x4], dim=1)

    def dynamics(self, x1, x2, x3, x4, x5, x6, x7, u, t=None):
        """Compute state derivative vector dx/dt for the protein production plant model.

        :param x1: Reactor volume tensor of shape ``(batch_size, 1)``.
        :type x1: torch.Tensor
        :param x2: Biomass cell density tensor of shape ``(batch_size, 1)``.
        :type x2: torch.Tensor
        :param x3: Glucose substrate concentration tensor of shape ``(batch_size, 1)``.
        :type x3: torch.Tensor
        :param x4: Recombinant protein concentration tensor of shape ``(batch_size, 1)``.
        :type x4: torch.Tensor
        :param x5: Inducer concentration tensor of shape ``(batch_size, 1)``.
        :type x5: torch.Tensor
        :param x6: Physiological shock factor tensor of shape ``(batch_size, 1)``.
        :type x6: torch.Tensor
        :param x7: Physiological recovery factor tensor of shape ``(batch_size, 1)``.
        :type x7: torch.Tensor
        :param u: Applied control feed inputs tensor [u1 (glucose), u2 (inducer)] of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current simulation time, defaults to ``None``.
        :type t: float or torch.Tensor, optional
        :returns: Tuple (dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt) containing continuous derivatives.
        :rtype: tuple[torch.Tensor, ...]
        """
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        
        R_R = self.K_CI / (self.K_CI + x5)
        
        # Intermediate kinetics (Eq 2, 3, 4)
        mu = (self.mu_max * x3 / (self.K_CN + x3 * (1.0 + x3 / self.K_s))) * \
             (x6 + x7 * R_R)
             
        R_f_p = (self.f_max * x3 / (self.K_CN + x3 * (1.0 + x3 / self.K_s))) * \
                ((self.f_I_0 + x5) / (self.K_I + x5))
            
        k_1 = self.k_11 * x5 / (self.K_IX + x5)
        k_2 = self.k_22 * x5 / (self.K_IX + x5)

        # Main Differential State Vector Mapping (Eq 1)
        dx1dt = u1 + u2
        dx2dt = x2 * mu - ((u1 + u2) / x1) * x2
        dx3dt = (u1 * self.C_n_f / x1) - ((u1 + u2) / x1) * x3 - (mu / self.Y) * x2
        dx4dt = x2 * R_f_p - ((u1 + u2) / x1) * x4
        dx5dt = (u2 * self.C_i_f / x1) - ((u1 + u2) / x1) * x5
        dx6dt = -k_1 * x6
        dx7dt = k_2 * (1.0 - x7)

        return dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt

    def step(self, state, u, t):
        """Advance simulation state across time horizon ``dt`` via vectorized adaptive RK45 integration.

        Integrates the 7-state continuous dynamic system over ``[t, t + dt]`` using a batch-parallel 
        adaptive step size Dormand-Prince method with local error estimation and adaptive step scaling.

        :param state: Current state tensor of shape ``(batch_size, 7)``.
        :type state: torch.Tensor
        :param u: Applied control feed inputs tensor [u1, u2] of shape ``(batch_size, 2)``.
        :type u: torch.Tensor
        :param t: Current simulation start time.
        :type t: float or torch.Tensor
        :param dt: Time integration horizon step size.
        :type dt: float or torch.Tensor
        :returns: Tuple ``(state_next, y_next)`` containing updated state tensor of shape 
            ``(batch_size, 7)`` and monitored output tensor of shape ``(batch_size, 3)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        dt = self.dt
        batch_size = state.shape[0]
        device = state.device
        
        t_start = t.item() if torch.is_tensor(t) else t
        t_end = t_start + dt
        
        # Dormand-Prince Tableau Matrix constants
        c2, a21 = 1/5, 1/5
        c3, a31, a32 = 3/10, 3/40, 9/40
        c4, a41, a42, a43 = 4/5, 44/45, -56/15, 32/9
        c5, a51, a52, a53, a54 = 8/9, 19372/6561, -25360/2187, 64448/6561, -212/729
        c6, a61, a62, a63, a64, a65 = 1.0, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        b1, b2, b3, b4, b5, b6, b7 = 35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0
        b1_star, b2_star, b3_star, b4_star, b5_star, b6_star, b7_star = (
            5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        )

        rtol, atol = 1e-4, 1e-6
        current_t = torch.full((batch_size, 1), t_start, device=device)
        y = state.clone()
        h = torch.full((batch_size, 1), dt / 10.0, device=device)
        
        max_steps, step_count = 100, 0
        
        while torch.any(current_t < t_end) and step_count < max_steps:
            h = torch.clamp(h, max=t_end - current_t)
            
            def unpack(s):
                return (s[:, 0:1], s[:, 1:2], s[:, 2:3], s[:, 3:4], 
                        s[:, 4:5], s[:, 5:6], s[:, 6:7])
            
            k1 = torch.cat(self.dynamics(*unpack(y), u, current_t), dim=1)
            y2 = y + h * (a21 * k1)
            k2 = torch.cat(self.dynamics(*unpack(y2), u, current_t + c2 * h), dim=1)
            y3 = y + h * (a31 * k1 + a32 * k2)
            k3 = torch.cat(self.dynamics(*unpack(y3), u, current_t + c3 * h), dim=1)
            y4 = y + h * (a41 * k1 + a42 * k2 + a43 * k3)
            k4 = torch.cat(self.dynamics(*unpack(y4), u, current_t + c4 * h), dim=1)
            y5 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
            k5 = torch.cat(self.dynamics(*unpack(y5), u, current_t + c5 * h), dim=1)
            y6 = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
            k6 = torch.cat(self.dynamics(*unpack(y6), u, current_t + c6 * h), dim=1)
            
            y_next = y + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
            k7 = torch.cat(self.dynamics(*unpack(y_next), u, current_t + h), dim=1)
            y_next_star = y + h * (b1_star*k1 + b2_star*k2 + b3_star*k3 + b4_star*k4 + b5_star*k5 + b6_star*k6 + b7_star*k7)
            
            error = torch.abs(y_next - y_next_star)
            scale = atol + rtol * torch.max(torch.abs(y), torch.abs(y_next))
            norm_error = torch.sqrt(torch.mean((error / scale) ** 2, dim=1, keepdim=True))
            
            step_accepted = norm_error <= 1.0
            y = torch.where(step_accepted, y_next, y)
            current_t = torch.where(step_accepted, current_t + h, current_t)
            
            scale_factor = torch.clamp(0.9 * (norm_error ** -0.2), min=0.2, max=5.0)
            h = h * scale_factor
            h = torch.clamp(h, min=1e-5)
            step_count += 1

        return y, self.get_y(y, t_end)

    def get_plot_config(self):
        """Configure matplotlib visualization metadata and LaTeX labels for simulation plots.

        :returns: List of plotting configuration dictionaries specifying columns, titles, and unit labels.
        :rtype: list[dict]
        """
        return [
            {
                "cols": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
                "labels": [
                    r"$x_1 \; [\mathrm{L}]$", 
                    r"$x_2 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_3 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_4 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_5 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_6 \; [\mathrm{dimensionless}]$",
                    r"$x_7 \; [\mathrm{dimensionless}]$"
                ],
                "ylabel": [
                    r"$x_1 \; [\mathrm{L}]$", 
                    r"$x_2 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_3 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_4 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_5 \; [\mathrm{g \cdot L^{-1}}]$",
                    r"$x_6 \; [\mathrm{-}]$",
                    r"$x_7 \; [\mathrm{-}]$"
                ]
            },
            {
                "cols": ["u1", "u2"],
                "labels": [
                    r"$u_1 \; [\mathrm{L \cdot h^{-1}}]$", 
                    r"$u_2 \; [\mathrm{L \cdot h^{-1}}]$"
                ],
                "ylabel": [
                    r"$u_1 \; [\mathrm{L \cdot h^{-1}}]$", 
                    r"$u_2 \; [\mathrm{L \cdot h^{-1}}]$"
                ]
            },
            {
                "cols": ["y1", "y2", "y3"],
                "labels": [
                    r"$y_1 \; [\mathrm{L}]$", 
                    r"$y_2 \; [\mathrm{g \cdot L^{-1}}]$", 
                    r"$y_3 \; [\mathrm{g \cdot L^{-1}}]$"
                ],
                "ylabel": [
                    r"$y_1 \; [\mathrm{L}]$", 
                    r"$y_2 \; [\mathrm{g \cdot L^{-1}}]$", 
                    r"$y_3 \; [\mathrm{g \cdot L^{-1}}]$"
                ]
            }
        ]

# Default hyperparameter configuration
hyperparam_config_IndForProteinProductionPlant = {
    "plant": {
        # Source of model parameters: Lee, J., & Ramirez, W. F. (1994). Optimal fed‐batch control of induced foreign protein production by recombinant bacteria. AIChE Journal, 40(5), 899–907. https://doi.org/10.1002/aic.690400516
        "mu_max": 0.407,            # Maximum specific growth rate [1/h]
        "K_CI": 0.22,               # Inducer inhibition/shock structural constant [g/L]
        "k_22": 0.09,               # Deactivation rate coefficient for protein shock [1/h]
        "K_s": 14814.8,             # Substrate inhibition constant multiplier [g/L]
        "f_I_0": 0.0005,
        "C_n_f": 100,               # Nutrient level of feed [g/L]
        "Y": 0.51,                  # Yield coefficient of substrate [(g cell mass)/(g nutrient)]

        "K_CN": 0.108,              # Nitrogen/Nutrient saturation constant [g/L]
        "k_11": 0.09,               # Deactivation rate coefficient for growth shock [1/h]
        "K_IX": 0.034,              # Cell density impact factor on deactivation [g/L]
        
        "f_max": 0.095,             # Max specific foreign protein production rate [1/h]
        "K_IX": 0.034,              # Inducer activation affinity constant [g/L]   

        "f_max": 0.095,             # Maximum protein production rate              
        "K_I": 0.022,               # Constant    
        "C_i_f": 4,                 # Inducer level of feed [g/L]

        "initial_state": [1,        # Initial reactor volume [L] 
                          0.1,      # Inital concentration of biomass [g/L]
                          40,       # Initial nutrient mass concentration [g/L]
                          0,        # Initial foreign protein mass concentration [g/L]
                          0,        # Initial inducer mass concentration [g/L]
                          1,        # Initial shock rate effect [-]
                          0         # Initial inducer recovery factor on cell growth rate [-]
                          ],
    },

    # Training data generation
    "training_data_cfg": {        
        "batch_size": 100,          
        "dt": 0.01,
        "seq_len": 1501,
        
        "input_dim": 2,            
        "output_dim": 3,

        "min_correlation_threshold": -1.1,
        "nu_u": 1,
        "nu_y": 1,

        "u_1_D_center_min": 0.05,
        "u_1_D_center_max": 0.80,

        "u_2_D_center_min": 0.00,
        "u_2_D_center_max": 0.25,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1.5,

        "u_2_hard_min": 0.0,
        "u_2_hard_max": 0.5,

        "x_1_hard_min": 0.0,
        "x_1_hard_max": None,

        "x_2_hard_min": 0.0,

        "x_3_hard_min": 0.0,

        "x_4_hard_min": 0.0,

        "x_5_hard_min": 0.0,   
        
        "x_6_hard_min": 0.0,
        "x_6_hard_max": None,

        "x_7_hard_min": 0.0,
        "x_7_hard_max": None, 

        "u_1_lambd": 4,
        "u_1_p": 0.5,
        
        "u_2_lambd": 4,
        "u_2_p": 0.5,
                
    },

    "train": {
        "k_folds": 5,                               # Number of cross validation splits
        "epochs": 100,                              # Maximum number of epochs
        "lr": 1e-3,                                 # Learning rate of the Adam optimizer
        "device": "cuda",                           # Core processing target execution context
        "mini_batch_size": 1,                       # Number of training sequences per epoch

        "loss_function": "MSELoss()",               # Loss function of the training process
        "patience_epochs": 3,                       # Number of patience epochs for the stopping criterion
        "patience_min_improvement": 0.0001,         # Minimum loss function improvement on the test set for stopping criterion

        "nu_y": 1,                                  # Lookback window for outputs
        "nu_u": 1,                                  # Lookback window for inputs
    },
    
    # Default hyperparameters for Mamba-based sequence models
    "mamba": {
        "d_state": 31,
        "expand": 9,
        "d_conv": 9
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
        "ridge": 1e-7,      
    },

    "validation_trajectories" : {
            "batch_size": 10,
            "seq_len"   : 401,
            "set_point" : 0.25,
            "amplitude" : 0.04,
            "period"    : 20.0,
    
            "y_start"   : 0.5,
            "y_target"  : 0.2,
            "tau"       : 0.1         
        }
}

