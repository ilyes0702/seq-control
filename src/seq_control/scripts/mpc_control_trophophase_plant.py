import matplotlib.pyplot as plt
from src.seq_control.classes.plants.TrophophasePlant import *
from src.seq_control.classes.controllers.MPCController import *
from src.seq_control.utils.validation_utils import *

# ==========================================
# 3. CLOSED-LOOP EXPERIMENT SIMULATION
# =======================================
# 1. Instantiate concrete plant and wrap it
trophophase = TrophophasePlant(hyperparam_config_TrophophasePlant)
plant_adapter = TrophophaseWrapper(trophophase)

# 2. Instantiate MPC controller
mpc = MPCController(
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
sim_steps = 1000
dt = hyperparam_config_TrophophasePlant["training_data_cfg"]["dt"]
batch_size = 1

current_state = trophophase.get_initial_state(batch_size=batch_size)

# Create sinusoidal target growth rate trajectory (\mu)
target_time_steps = torch.linspace(0, 3.14 * 2, sim_steps, device=plant_adapter.device)
target_mu_trajectory = torch.sin(target_time_steps) * 0.0 + 0.015  # Target output signal [sim_steps]



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