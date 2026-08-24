======================
generate_signal_single
======================

.. function:: generate_signal_single(training_data_cfg, channel_idx=1)

   Generates smooth, band-limited control signals using an Active Shielding methodology. 
   
   This function guarantees that each unique Multi-Input Multi-Output (MIMO) channel stays completely within its hard boundaries. It utilizes independent, channel-specific bandwidth (:math:`\lambda`) and amplitude (:math:`p`) settings.

   :param dict training_data_cfg: Configuration dictionary containing global and channel-specific operational parameters:
   
      * **batch_size** (*int*) -- Number of trajectories to generate in parallel.
      * **seq_len** (*int*) -- Length of the signal trajectory sequence.
      * **dt** (*float*) -- Sampling time step for frequency analysis.
      * **lambd** (*float*) -- Global fallback wavelength/bandwidth parameter.
      * **p** (*float*) -- Global fallback target amplitude parameter.
      * **u_{channel_idx}_lambd** (*float, optional*) -- Channel-specific bandwidth cutoff limit.
      * **u_{channel_idx}_p** (*float, optional*) -- Channel-specific target amplitude.
      * **u_{channel_idx}_D_center_min** (*float*) -- Lower bound for baseline center distribution.
      * **u_{channel_idx}_D_center_max** (*float*) -- Upper bound for baseline center distribution.
      * **u_{channel_idx}_hard_min** (*float*) -- Strict lower physical threshold limit.
      * **u_{channel_idx}_hard_max** (*float*) -- Strict upper physical threshold limit.

   :param int channel_idx: Index specifying the target MIMO channel (default is ``1``).
   
   :return: A tuple containing:
   
      * **u_buffer** (*torch.Tensor*) -- Active-shielded, band-limited control signal tensor of shape ``(batch_size, seq_len)``.
      * **u_center** (*torch.Tensor*) -- Generated baseline centers for each trajectory of shape ``(batch_size, 1)``.
   
   :rtype: Tuple[torch.Tensor, torch.Tensor]

   Process Flow
   ------------

   1. **Dynamic Configuration Extraction**
      Extracts channel-specific parameters (``lambd``, ``p``, bounds) with immediate fallbacks to global defaults if channel overrides are omitted.

   2. **Frequency Filtering**
      Generates uniform noise, converts to frequency domain via Real Fast Fourier Transform (``rfft``), clips frequencies above cutoff frequency :math:`f_c = \frac{1}{\lambda}`, and transforms back via Inverse RFFT (``irfft``).

   3. **Normalization**
      Scales the smooth trajectories to the normalized range :math:`[-1, 1]`.

   4. **Active Shielding Safeguard**
      * Calculates distance from randomly assigned centers (:math:`u_{\text{center}}`) to hard boundaries (:math:`u_{\text{hard\_min}}`, :math:`u_{\text{hard\_max}}`).
      * Sets safe amplitude :math:`p_{\text{adaptive}} = \min(p_{\text{configured}}, 0.98 \cdot p_{\text{max\_safe}})`.
      * Clamps output signal to guarantee zero hard-limit breaches due to floating-point imprecision.

   Example Usage
   -------------

   .. code-block:: python

      import torch
      from signal_generator import generate_signal_single

      cfg = {
          "batch_size": 32,
          "seq_len": 100,
          "dt": 0.01,
          "lambd": 0.5,
          "p": 1.0,
          "u_1_lambd": 0.2,
          "u_1_p": 0.5,
          "u_1_D_center_min": -0.5,
          "u_1_D_center_max": 0.5,
          "u_1_hard_min": -2.0,
          "u_1_hard_max": 2.0,
      }

      u_buffer, u_center = generate_signal_single(cfg, channel_idx=1)