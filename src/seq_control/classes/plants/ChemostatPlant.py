import torch

class ChemostatPlant:
    """Simulates a continuous chemostat bioreactor plant in PyTorch.

    Models the biological growth process of biomass and substrate consumption 
    governed by Monod kinetics using explicit Dormand-Prince (RK45) numerical integration.

    .. math::

        \\mu(s) = \\frac{\\mu_{max} \\cdot s}{K_s + s}

        \\frac{dx}{dt} = (\\mu - u) x

        \\frac{ds}{dt} = u (s_R - s) - \\frac{\\mu x}{Y}

    Attributes:
        VARIABLE_UNITS (dict): Mapping of state and control variable names to physical units.
        device (torch.device or str): Execution device for tensor operations (CPU or CUDA).
        dt (float): Simulation step size in hours.
        mu_max (torch.Tensor): Maximum specific growth rate parameter (:math:`h^{-1}`).
        Ks (torch.Tensor): Half-saturation affinity constant parameter (:math:`g\\,L^{-1}`).
        Y (torch.Tensor): Biomass yield coefficient parameter on substrate (:math:`g\\,g^{-1}`).
        sR (torch.Tensor): Feed substrate concentration parameter (:math:`g\\,L^{-1}`).
        hyperparam_config (dict): Full hyperparameter configuration dictionary.
    """
    VARIABLE_UNITS = {
        "x": "g L⁻¹",      # Biomass concentration
        "biomass": "g L⁻¹",
        "s": "g L⁻¹",      # Substrate concentration
        "substrate": "g L⁻¹",
        "y": "h⁻¹",        # Growth rate (mu)
        "mu": "h⁻¹",
        "u": "h⁻¹",        # Dilution rate (D)
    }
    def __init__(self, hyperparam_config):
        """Initialize the Chemostat plant model with configuration settings.

        :param hyperparam_config: Dictionary containing plant physical parameters and setup options.
            Expects keys ``['train']['device']``, ``['training_data_cfg']['dt']``, and
            ``['plant']`` with keys ``['mu-max']``, ``['Ks']``, ``['Y']``, and ``['sR']``.
        :type hyperparam_config: dict
        """
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["training_data_cfg"]["dt"]

        # Biological Parameters from Config
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu-max"], device=self.device)
        self.Ks = torch.tensor(hyperparam_config["plant"]["Ks"], device=self.device)
        self.Y = torch.tensor(hyperparam_config["plant"]["Y"], device=self.device)
        self.sR = torch.tensor(hyperparam_config["plant"]["sR"], device=self.device)
        self.hyperparam_config = hyperparam_config

    def get_initial_state(self, batch_size):
        """Generate randomized initial state vectors for training batch initialization.

        :param batch_size: Number of parallel batch instances to sample.
        :type batch_size: int
        :returns: Tensor of shape ``(batch_size, 2)`` representing initial states ``[x, s]``.
        :rtype: torch.Tensor
        """
        x_init = torch.rand((batch_size, 1), device=self.device) * 0.2 + 0.2 # 0.1 to 0.6
        s_init = torch.rand((batch_size, 1), device=self.device) * 0.2 + 0.1 # 0.1 to 0.6
        return torch.cat([x_init, s_init], dim=1)

    def get_y(self, state, t=None):
        """Compute the observable plant output (specific growth rate :math:`\\mu`).

        Calculates specific growth rate according to the Monod growth kinetics formulation.

        :param state: Plant state tensor of shape ``(batch_size, 2)`` containing ``[x, s]``.
        :type state: torch.Tensor
        :param t: Current simulation time step or timestamp, optional.
        :type t: float or torch.Tensor, optional
        :returns: Computed specific growth rate :math:`\\mu` of shape ``(batch_size, 1)``.
        :rtype: torch.Tensor
        """
        s = state[:, 1:2]
        mu = (self.mu_max * s) / (self.Ks + s)
        return mu 

    def dynamics(self, x, s, u, t = None):
        """Calculate continuous-time system derivatives for the chemostat process.

        Evaluates differential equations for biomass growth and substrate depletion.

        :param x: Biomass concentration tensor of shape ``(batch_size, 1)``.
        :type x: torch.Tensor
        :param s: Substrate concentration tensor of shape ``(batch_size, 1)``.
        :type s: torch.Tensor
        :param u: Control input tensor (dilution rate :math:`D`) of shape ``(batch_size, 1)``.
        :type u: torch.Tensor
        :param t: Current simulation time, optional.
        :type t: float or torch.Tensor, optional
        :returns: Tuple containing ``(dxdt, dsdt)`` derivative tensors each of shape ``(batch_size, 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        mu = (self.mu_max * s) / (self.Ks + s)
        dxdt = mu * x - u * x
        dsdt = u * (self.sR - s) - (mu * x / self.Y)
        return dxdt, dsdt

    def step(self, state, u, t, dt):
        """Perform dynamic state integration using the Dormand-Prince (RK45) scheme.

        Advances the system state forward across time step ``dt`` using a 5th-order accurate 
        embedded Runge-Kutta numerical solver step.

        :param state: Current state tensor of shape ``(batch_size, 2)``.
        :type state: torch.Tensor
        :param u: Applied control input (dilution rate) tensor of shape ``(batch_size, 1)``.
        :type u: torch.Tensor
        :param t: Current simulation time.
        :type t: float or torch.Tensor
        :param dt: Time increment step size.
        :type dt: float or torch.Tensor
        :returns: Tuple ``(state_next, y_next)`` containing updated state tensor of shape 
            ``(batch_size, 2)`` and observable output tensor of shape ``(batch_size, 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x, s = state[:, 0:1], state[:, 1:2]
        
        # Butcher tableau coefficients for Dormand-Prince
        # k1
        dx1, ds1 = self.dynamics(x, s, u)
        
        # k2
        x2 = x + dt * (1/5 * dx1)
        s2 = s + dt * (1/5 * ds1)
        dx2, ds2 = self.dynamics(x2, s2, u)
        
        # k3
        x3 = x + dt * (3/40 * dx1 + 9/40 * dx2)
        s3 = s + dt * (3/40 * ds1 + 9/40 * ds2)
        dx3, ds3 = self.dynamics(x3, s3, u)
        
        # k4
        x4 = x + dt * (44/45 * dx1 - 56/15 * dx2 + 32/9 * dx3)
        s4 = s + dt * (44/45 * ds1 - 56/15 * ds2 + 32/9 * ds3)
        dx4, ds4 = self.dynamics(x4, s4, u)
        
        # k5
        x5 = x + dt * (19372/6561 * dx1 - 25360/2187 * dx2 + 64448/6561 * dx3 - 212/729 * dx4)
        s5 = s + dt * (19372/6561 * ds1 - 25360/2187 * ds2 + 64448/6561 * ds3 - 212/729 * ds4)
        dx5, ds5 = self.dynamics(x5, s5, u)
        
        # k6
        x6 = x + dt * (9017/3168 * dx1 - 355/33 * dx2 + 46732/5247 * dx3 + 49/176 * dx4 - 5103/18656 * dx5)
        s6 = s + dt * (9017/3168 * ds1 - 355/33 * ds2 + 46732/5247 * ds3 + 49/176 * ds4 - 5103/18656 * ds5)
        dx6, ds6 = self.dynamics(x6, s6, u)

        # 5th-order accurate update (Primary step)
        x_next = x + dt * (35/384 * dx1 + 500/1113 * dx3 + 125/192 * dx4 - 2187/6784 * dx5 + 11/84 * dx6)
        s_next = s + dt * (35/384 * ds1 + 500/1113 * ds3 + 125/192 * ds4 - 2187/6784 * ds5 + 11/84 * ds6)

        state_next = torch.cat([x_next, s_next], dim=1)
        return state_next, self.get_y(state_next)

    def get_plot_config(self):
        """Retrieve visualization settings for plotting system time-series.

        :returns: List of configuration dictionaries specifying column groupings, 
            LaTeX formatted legends, figure titles, and axis labels.
        :rtype: list[dict]
        """
        return [
            {
                "cols": ["x1", "x2"],
                "labels": [r"$x_1$ [$\mathrm{g}\,\mathrm{L}^{-1}$]", r"$x_2$ [$\mathrm{g}\,\mathrm{L}^{-1}$]"],
                "title": "Chemostat State Evolution",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["y"],
                "labels": [r"$y$ [$\mathrm{h}^{-1}$]"],
                "title": "Growth Rate Inverse Learning",
                "ylabel": "Growth Rate [1/h]"
            },
            {
                "cols": ["u"],
                "labels": [r"$u$ [$\mathrm{L}\,\mathrm{h}^{-1}$]"],
                "title": "Control Input (D)",
                "ylabel": "Dilution Rate [1/h]"
            }
        ]

# Default hyperparameter configuration 
hyperparam_config_ChemostatPlant = {
    "plant" :{
        "mu-max": 0.5,      # Maximum growth rate [1/h]
        "Ks": 0.2,          # Half-saturation constant 
        "Y": 0.6,           # Yield coefficient
        "sR": 1.0,
        "input_dim": 1,     # number of plant outputs
        "output_dim": 1,    # number of plant control inputs
        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.5,

        

    },
    "signal": {
        
        "dt": 0.1
    },
    "train": {
        "k_folds": 2,
        "epochs": 20,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        
        "mini_batch_size": 1,
        
        
        "loss_function": "MSELoss", 
        "lr_decay_rate":1,
        
        "test_min_epochs": 3,
        "test_min_delta": 0.0005,

        "n_u": 2,
        "n_y": 2,
        "lookback_offset": 2,
        "test_patience_epochs": 3,

        
    },
    "training_data_cfg" : {
        "batch_size": 100,
        "seq_len": 501,
        "dt": 0.1,
        "input_dim": 1,   # number of plant outputs
        "output_dim": 1,   # number of plant control inputs,
        "min_correlation_threshold": -10, #0.7,
        "n_u": 2,
        "n_y": 2,

        "u_1_D_center_min": 0.15,
        "u_1_D_center_max": 0.2,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.5,

        "u_1_lambd": 20,
        "u_1_p": 0.05,

        

    },
    "mamba": {
        "d_state": 1,
        "expand": 1,
        "d_conv" : 1
    },

    "mamba_param_space" : {
        "mamba.d_conv":  {"type": "int", "low": 1, "high": 10},
        "mamba.d_state": {"type": "int", "low": 1, "high": 64},
        "mamba.expand":  {"type": "int", "low": 1, "high": 10},
        },

    "lstm": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "lstm_param_space" : {
        "lstm.hidden_size": {"type": "int", "low": 16, "high": 128},
        "lstm.num_layers": {"type": "int", "low": 1, "high": 4},
        "lstm.dropout": {"type": "float", "low": 0.0, "high": 0.5},
    },
    "esn": {
        "units": 200,   
        "lr": 0.5,
        "sr": 0.9,
        "ridge": 1e-7,    # Regularization coefficient  
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 401,
    }
}