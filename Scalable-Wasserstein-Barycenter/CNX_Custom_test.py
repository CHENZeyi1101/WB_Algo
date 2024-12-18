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
for epoch_to_load in range(1, 2):
    barycenter_samples = CDR.barycenter_sampler(
        cfg, PTU.device, load_epoch=epoch_to_load
    )

    df = pd.DataFrame(barycenter_samples.detach().numpy())
    df.to_csv(f"./CNX_outputs/Custom/outputs_NWBFanTaghvaeiChen_samples_epoch_{epoch_to_load}.csv",index=False, header=False)
