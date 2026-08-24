=====================
LSTMInverseController
=====================

.. py:class:: LSTMInverseController(hyperparam_config, feature_dim=None)

   Bases: :class:`torch.nn.Module`

   An LSTM-based inverse controller designed for dynamic Multi-Input Multi-Output (MIMO) system identification and closed-loop control with arbitrary sliding window inputs.

   This module processes historical and target plant state vectors :math:`v_k` to predict optimal actuator control inputs :math:`u_k`. It uses standard Recurrent Neural Network state gating to support both full-sequence training and online step-by-step stateful inference.

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
          "lstm": {
              "hidden_size": int,  # Hidden layer feature dimension
              "num_layers": int,   # Number of stacked LSTM layers
              "dropout": float,    # Dropout probability (applied if num_layers > 1)
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

   .. py:attribute:: hidden_dim
      :type: int

      Dimension of the hidden state vectors in the LSTM core.

   .. py:attribute:: num_layers
      :type: int

      Number of recurrent layers stacked in the LSTM core.

   .. py:attribute:: d_model
      :type: int

      Feature length of the lookback vector :math:`v_k`. Computed as:

      .. math::

         d\_model = (n_u \cdot \text{input\_dim}) + ((n_y + 2) \cdot \text{output\_dim})

   .. py:attribute:: core
      :type: torch.nn.LSTM

      The underlying PyTorch LSTM architecture core.

   .. py:attribute:: output_proj
      :type: torch.nn.Linear

      Linear projection layer mapping latent hidden dimensions back to control/actuator dimensions.

   .. py:attribute:: lstm_state
      :type: tuple(torch.Tensor, torch.Tensor) or None

      Inference memory state buffer storing the hidden and cell state tuple ``(h_0, c_0)``.


Methods
-------

.. py:method:: forward(v_seq)

   Performs a standard 3D sequence training forward pass over batch sequences.

   :param torch.Tensor v_seq: Stacked feature sequence tensor of shape ``(Batch, Seq_Len, d_model)``.
   :returns: Predicted control sequence tensor of shape ``(Batch, Seq_Len, input_dim)``.
   :rtype: torch.Tensor

.. py:method:: reset_memory(batch_size=1, device="cuda")

   Allocates or clears zero-filled recurrent state memory buffers (``h_0`` and ``c_0``).
   **Must be called prior to rolling closed-loop evaluation via** :meth:`step`.

   :param int batch_size: Evaluation batch size (default: ``1``).
   :param str device: Computing device (default: ``"cuda"``).

.. py:method:: step(v_k_single)

   Executes a single-step recurrent transition for real-time closed-loop evaluation. Consumes the current state vector :math:`v_k` and updates internal hidden/cell states.

   :param torch.Tensor v_k_single: Lookback feature vector of shape ``(Batch, d_model)`` or ``(Batch, 1, d_model)``.
   :returns: Target control instruction tensor of shape ``(Batch, input_dim)``.
   :rtype: torch.Tensor
   :raises RuntimeError: If called before initializing state buffers via :meth:`reset_memory`.