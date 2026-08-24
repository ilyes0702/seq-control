======================
MambaInverseController
======================

.. py:class:: MambaInverseController(hyperparam_config, feature_dim=None)

   Bases: :class:`torch.nn.Module`

   A Mamba (S6 Selective State Space Model) inverse controller designed for dynamic Multi-Input Multi-Output (MIMO) system identification and closed-loop control with arbitrary sliding window inputs.

   This module processes historical and target plant state vectors :math:`v_k` to predict optimal actuator control inputs :math:`u_k`. It supports both offline full-sequence training and online step-by-step stateful inference.

   :param dict hyperparam_config: Dictionary containing model architecture and plant specifications.
   :param int feature_dim: (Optional) Explicit feature dimension of input vector :math:`v_k`. If ``None``, it is computed automatically from lookback windows.

   .. rubric:: System Configuration Keys

   The ``hyperparam_config`` dictionary requires the following structure:

   .. code-block:: python

      hyperparam_config = {
          "plant": {
              "input_dim": int,    # Dimension of plant outputs (y)
              "output_dim": int,   # Dimension of plant controls (u)
          },
          "mamba": {
              "d_state": int,      # SSM state expansion factor
              "expand": int,       # Block expansion factor
              "d_conv": int,       # Convolution kernel size
          },
          "train": {
              "n_y": int,          # Lookback history size for outputs y
              "n_u": int,          # Lookback history size for controls u
          }
      }

   .. rubric:: Attributes

   .. py:attribute:: input_dim
      :type: int

      Dimension of the plant output vector :math:`y`.

   .. py:attribute:: output_dim
      :type: int

      Dimension of the plant control input vector :math:`u`.

   .. py:attribute:: d_model
      :type: int

      Feature length of the lookback vector :math:`v_k`. Computed as:

      .. math::

         d\_model = (n_u \cdot \text{input\_dim}) + ((n_y + 2) \cdot \text{output\_dim})

   .. py:attribute:: core
      :type: Mamba

      The underlying Mamba S6 state-space architecture core.

   .. py:attribute:: output_proj
      :type: torch.nn.Linear

      Linear projection layer mapping state dimensions back to control/actuator dimensions.

   .. py:attribute:: conv_state
      :type: torch.Tensor or None

      Inference memory state buffer for 1D convolutions.

   .. py:attribute:: ssm_state
      :type: torch.Tensor or None

      Inference memory state buffer for recurrent SSM states.


Methods
-------

.. py:method:: forward(v_seq)

   Performs a standard 3D sequence training forward pass over batch sequences.

   :param torch.Tensor v_seq: Stacked feature sequence tensor of shape ``(Batch, Seq_Len, d_model)``.
   :returns: Predicted control sequence tensor of shape ``(Batch, Seq_Len, input_dim)``.
   :rtype: torch.Tensor

.. py:method:: reset_memory(batch_size=1, device="cuda")

   Allocates or clears zero-filled recurrent state memory buffers (``conv_state`` and ``ssm_state``).
   **Must be called prior to rolling closed-loop evaluation via** :meth:`step`.

   :param int batch_size: Evaluation batch size (default: ``1``).
   :param str device: Computing device (default: ``"cuda"``).

.. py:method:: step(v_k_single)

   Executes a single-step recurrent transition for real-time closed-loop evaluation. Consumes the current state vector :math:`v_k` and updates internal SSM memory states.

   :param torch.Tensor v_k_single: Lookback feature vector of shape ``(Batch, d_model)`` or ``(Batch, 1, d_model)``.
   :returns: Target control instruction tensor of shape ``(Batch, input_dim)``.
   :rtype: torch.Tensor
   :raises RuntimeError: If called before initializing state buffers via :meth:`reset_memory`.


State-Space Model Properties
----------------------------

The controller exposes underlying SSM linear system parameters for analysis and visualization:

.. py:attribute:: A

   State transition matrix :math:`A` extracted from the Mamba core.

.. py:attribute:: B

   Input matrix :math:`B` extracted from the Mamba core.

.. py:attribute:: C

   Output matrix :math:`C` extracted from the Mamba core.

.. py:attribute:: D

   Feedthrough matrix :math:`D` extracted from the Mamba core.

.. py:attribute:: mamba_dt

   Discretization timescale factor :math:`\Delta t` associated with the SSM core.

