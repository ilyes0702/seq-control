# Import inverse controller architectures
from seq_control.classes.sequence_models.MambaInverseController import *
from seq_control.classes.sequence_models.ESNInverseController import *
from seq_control.classes.sequence_models.LSTMInverseController import *
from seq_control.classes.sequence_models.TransformerInverseController import *

controller_dict = {
        "MambaInverseController": MambaInverseController,
        "LSTMInverseController": LSTMInverseController,
        "TransformerInverseController":TransformerInverseController,
        "ESNInverseController":TransformerInverseController
    }