from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import optimal_transport_modules.icnn_modules as NN_modules



def load_generator_h(results_save_path, generator_h, epochs, device=None):
    model_save_path = results_save_path + '/storing_models'
    generator_h.load_state_dict(torch.load(
        model_save_path + '/generator_h_epoch{0}.pt'.format(epochs), map_location=device))

    # try:
    #     generator_h.load_state_dict(torch.load(
    #         model_save_path + '/generator_h_epoch{0}.pt'.format(epochs), map_location=device))
    # except:
    #     print("no such file")
    #     print(model_save_path + '/generator_h_epoch{0}.pt'.format(epochs))

    # return generator_h.cuda(device)
    return generator_h.cpu()
