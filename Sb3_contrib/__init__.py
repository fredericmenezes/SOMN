import os

from Sb3_contrib.ars import ARS
from Sb3_contrib.ppo_mask import MaskablePPO
from Sb3_contrib.ppo_recurrent import RecurrentPPO
from Sb3_contrib.qrdqn import QRDQN
from Sb3_contrib.tqc import TQC
from Sb3_contrib.trpo import TRPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "ARS",
    "MaskablePPO",
    "RecurrentPPO",
    "QRDQN",
    "TQC",
    "TRPO",
]
