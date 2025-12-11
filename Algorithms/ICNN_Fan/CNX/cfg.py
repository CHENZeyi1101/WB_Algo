from dataclasses import dataclass, field
import pickle
from pathlib import Path
import optimal_transport_modules.cfg


@dataclass
class CNXCfgGaussian(optimal_transport_modules.cfg.Cfg3loop_F):
    NUM_GMM_COMPONENT: list = field(default_factory=list)
    MEAN: list = field(default_factory=list)
    COV: list = field(default_factory=list)
    NUM_NEURON: int = 32
    NUM_NEURON_h: int = 32
    INPUT_DIM_fg: int = 0
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 3
    epochs: int = 40
    N_TEST: int = 10000

    def get_save_path(self):
        return './CNX_outputs/Gaussian'

    def get_save_path_F(self):
        return './CNX_outputs/Gaussian'
    

@dataclass
class CNXCfgCustom(optimal_transport_modules.cfg.Cfg3loop_F):
    NUM_NEURON: int = 32
    NUM_NEURON_h: int = 32
    INPUT_DIM_fg: int = 0
    NUM_LAYERS: int = 3
    NUM_LAYERS_h: int = 3
    epochs: int = 40
    N_TEST: int = 5000
    DIM: int = None  # Initialize with a default value (None)
    NUM_DISTRIBUTION: int = None

    def __post_init__(self):
        # Prompt user for `dim` if not set
        if self.DIM is None:
            self.DIM = int(input("Please enter the value for dim: "))
        if self.NUM_DISTRIBUTION is None:
            self.NUM_DISTRIBUTION = int(input("Please enter the value for num_distribution: "))

    def get_save_path(self, path_name = None):
        if path_name is None:
            return f'./CNX_outputs/Custom_dim{self.DIM}_measures{self.NUM_DISTRIBUTION}'
        else:
            return path_name

    def get_save_path_F(self, path_name = None):
        if path_name is None:
            return f'./CNX_outputs/Custom_dim{self.DIM}_measures{self.NUM_DISTRIBUTION}'
        else:
            return path_name
    
