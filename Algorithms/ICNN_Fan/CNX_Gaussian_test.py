import numpy as np
import GPUtil

from CNX.cfg import CNXCfgGaussian as Cfg_class
import CNX.compare_dist_results as CDR
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.data_utils as DTU
import jacinle.io as io
import pandas as pd

cfg = Cfg_class()

# gpus_choice = GPUtil.getFirstAvailable(
#     order='random', maxLoad=0.5, maxMemory=0.5, attempts=5, interval=900, verbose=False)
# PTU.set_gpu_mode(True, gpus_choice[0])
PTU.set_gpu_mode(False, 0)

#! For the error:
epoch_to_load = 2

cfg.MEAN, cfg.COV, cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_GMM_COMPONENT, cfg.high_dim_flag = DTU.get_gmm_param(
    cfg.TRIAL)

barycenter_samples = CDR.barycenter_sampler(
    cfg, PTU.device, load_epoch=epoch_to_load
)

df = pd.DataFrame(barycenter_samples.detach().numpy())
df.to_csv("./CNX_outputs/Gaussian/Gaussian_samples.csv",index=False, header=False)
