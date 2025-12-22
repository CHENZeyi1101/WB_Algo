from __future__ import print_function
import logging
# import GPUtil

import sys
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# Get the parent folder path (folder K)
parent_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

print(parent_folder_path)

sys.path.append(parent_folder_path)

from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from ...Algorithms.ICNN_Fan.optimal_transport_modules import log_utils as LLU
from ...Algorithms.ICNN_Fan.optimal_transport_modules import generate_data as g_data
from ...Algorithms.ICNN_Fan.optimal_transport_modules import generate_NN as g_NN
from ...Algorithms.ICNN_Fan.optimal_transport_modules import pytorch_utils as PTU
from ...Algorithms.ICNN_Fan.optimal_transport_modules.record_mean_cov import select_mean_and_cov
from ...Algorithms.ICNN_Fan.CNX.cfg import CNXCfgCustom as Cfg_class
from ...Algorithms.ICNN_Fan.CNX import compare_dist_results as CDR
from .samplers_dim2 import *

##### For computing the constraint loss of negtive weights ######
def compute_constraint_loss(list_of_params):
    loss_val = 0
    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Training function definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
############## For each function here, it's an epoch ##################

def train(epoch, csv_path):
    convex_f.train()
    convex_g.train()
    generator_h.train()

    # These values are just for saving data
    w2_loss_value_epoch = 0
    g_OT_loss_value_epoch = [0] * cfg.NUM_DISTRIBUTION
    g_constraints_loss_value_epoch = 0
    remaining_f_loss_value_epoch = [0] * cfg.NUM_DISTRIBUTION
    mu_2moment_loss_value_epoch = 0
    miu_mean_value_epoch = 0
    miu_var_value_epoch = 0

    """""""""""""""""""""""""""""""""""""""""""""""""""
                            Data
    """""""""""""""""""""""""""""""""""""""""""""""""""
    total_data = torch.empty(cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION + 1)

    for marg_id in range(cfg.NUM_DISTRIBUTION):
        df = pd.read_csv(f"{csv_path}/input_measure_samples_{marg_id}.csv", header=None)
        total_data[:, :, marg_id] = torch.from_numpy(df.to_numpy())

    total_data[:, :, -1] = torch.randn(cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM)

    train_loader = torch.utils.data.DataLoader(
        total_data, batch_size=cfg.BATCH_SIZE, shuffle=True, **kwargs)

    for batch_idx, real_data in enumerate(train_loader):
        # real_data = real_data.cuda(PTU.device)
        real_data = real_data.cpu()

        miu_i = real_data[:, :, 0:cfg.NUM_DISTRIBUTION]
        epsilon = real_data[:, :, cfg.NUM_DISTRIBUTION]
        miu_i = Variable(miu_i, requires_grad=True)
        epsilon = Variable(epsilon)

        # containing four distribution
        g_OT_loss_value_batch = [0] * cfg.NUM_DISTRIBUTION
        g_constraints_loss_value_batch = 0  # containing four g networks
        remaining_f_loss_value_batch = [0] * cfg.NUM_DISTRIBUTION
        mu_2moment_loss_value_batch = 0
        miu_mean_value_batch = torch.zeros([cfg.INPUT_DIM])
        miu_var_value_batch = np.zeros(
            [cfg.INPUT_DIM, cfg.INPUT_DIM])

        ######################################################
        #                Medium Loop Begin                   #
        ######################################################
        ######### Here iterate over a given number: cfg.N_Fnet_ITERS=4 ##
        for medium_iter in range(1, cfg.N_Fnet_ITERS + 1):

            ######################################################
            #                Inner Loop Begin                   #
            ######################################################
            ######### Here iterate over a given number: cfg.N_Gnet_ITERS=16 ##
            for inner_iter in range(1, cfg.N_Gnet_ITERS + 1):

                loss_g = torch.ones(cfg.NUM_DISTRIBUTION)
                for i in range(cfg.NUM_DISTRIBUTION):
                    optimizer_g[i].zero_grad()

                    # Get the gradient of g(y):=g(miu_i_data)
                    tmp_miu_i = miu_i[:, :, i]
                    g_of_y = convex_g[i](tmp_miu_i).sum()
                    grad_g_of_y = torch.autograd.grad(
                        g_of_y, tmp_miu_i, create_graph=True)[0]

                    # For each distribution you need to calculate a f(gradient of y)
                    # it's the mean of the batch
                    f_grad_g_y = convex_f[i](grad_g_of_y).mean()
                    # The 1st loss part useful for f/g parameters
                    loss_g[i] = f_grad_g_y - torch.dot(
                        grad_g_of_y.reshape(-1), miu_i[:, :, i].reshape(-1)) / cfg.BATCH_SIZE
                    g_OT_loss_value_batch[i] += loss_g[i].item()

                total_loss_g = loss_g.sum()
                total_loss_g.backward()
                # The 2nd loss part useful for g parameters:
                g_positive_constraints_loss = cfg.LAMBDA_CVX * \
                    compute_constraint_loss(
                        g_positive_params)
                g_constraints_loss_value_batch += g_positive_constraints_loss.item()
                g_positive_constraints_loss.backward()

                # ! update g
                for i in range(cfg.NUM_DISTRIBUTION):
                    optimizer_g[i].step()

                # Just for the last iteration keep the gradient on f intact
                if inner_iter != cfg.N_Gnet_ITERS:
                    for i in range(cfg.NUM_DISTRIBUTION):
                        optimizer_f[i].zero_grad()

            ######################################################
            #                Inner Loop Ends                     #
            ######################################################
            miu = generator_h(epsilon)
            miu_mean = miu.mean(dim=0).cpu()
            miu_var = np.cov(miu.cpu().detach().numpy().T)
            miu_mean_value_batch += miu_mean
            miu_var_value_batch += miu_var

            remaining_f_loss = torch.ones(cfg.NUM_DISTRIBUTION)
            # The 3rd loss part useful for f/h parameters
            for i in range(cfg.NUM_DISTRIBUTION):
                remaining_f_loss[i] = - convex_f[i](miu).mean()
                remaining_f_loss_value_batch[i] += remaining_f_loss[i].item()
            total_remaining_f_loss = remaining_f_loss.sum()
            total_remaining_f_loss.backward(retain_graph=True)

            # Flip the gradient sign for parameters in convex f
            # Because we need to solve "sup" of the loss for f
            for p in list(convex_f.parameters()):
                p.grad.copy_(-p.grad)
            # ! update f
            for i in range(cfg.NUM_DISTRIBUTION):
                optimizer_f[i].step()

            # Clamp the positive constraints on the convex_f_params
            for p in f_positive_params:
                p.data.copy_(torch.relu(p.data))

            if medium_iter != cfg.N_Fnet_ITERS:
                optimizer_h.zero_grad()

        ######################################################
        #               Medium Loop Ends                     #
        ######################################################
        # The 4th loss part useful for h parameters:
        # ? keep untouched for all weights
        # mu_2moment_loss_value_batch is total 4 distributions combined F
        mu_2moment_loss = 0.5 * \
            miu.pow(2).sum(dim=1).mean() * cfg.NUM_DISTRIBUTION
        mu_2moment_loss_value_batch += mu_2moment_loss.item() / cfg.NUM_DISTRIBUTION

        # ! update h
        mu_2moment_loss.backward()
        optimizer_h.step()
        # The four parts loss gradients are accumulated

        miu_mean_value_batch = miu_mean_value_batch / cfg.N_Fnet_ITERS
        miu_var_value_batch = miu_var_value_batch / cfg.N_Fnet_ITERS

        g_OT_loss_value_batch[:] = [
            item / (cfg.N_Gnet_ITERS * cfg.N_Fnet_ITERS) for item in g_OT_loss_value_batch]
        remaining_f_loss_value_batch[:] = [
            item / cfg.N_Fnet_ITERS for item in remaining_f_loss_value_batch]
        g_constraints_loss_value_batch /= (cfg.N_Gnet_ITERS *
                                           cfg.N_Fnet_ITERS)

        ##### Calculate W2 batch loss ###############
        w2_loss_value_batch = (sum(g_OT_loss_value_batch) + sum(remaining_f_loss_value_batch)) / cfg.NUM_DISTRIBUTION + \
            mu_2moment_loss_value_batch + 0.5 * \
            miu_i.pow(2).sum(dim=1).mean().item()
        w2_loss_value_batch *= 2
        # miu_i.pow(2).sum(dim=1).mean().item() is already the mean of all distributions

        ##### Calculate all epoch loss ###############
        w2_loss_value_epoch += w2_loss_value_batch
        miu_mean_value_epoch += miu_mean_value_batch
        miu_var_value_epoch += miu_var_value_batch

        g_OT_loss_value_epoch = [
            a + b for a,
            b in zip(
                g_OT_loss_value_epoch,
                g_OT_loss_value_batch)]
        g_constraints_loss_value_epoch += g_constraints_loss_value_batch
        remaining_f_loss_value_epoch = [
            a + b for a,
            b in zip(
                remaining_f_loss_value_epoch,
                remaining_f_loss_value_batch)]
        mu_2moment_loss_value_epoch += mu_2moment_loss_value_batch

        if batch_idx % cfg.log_interval == 0:
            logging.info('Train_Epoch: {} [{}/{} ({:.0f}%)] avg_dstb_g_OT_loss: {:.4f} avg_dstb_remaining_f_loss: {:.4f} mu_2moment_loss: {:.4f} g_constraint_loss: {:.4f} W2_loss: {:.4f} '.format(
                epoch,
                batch_idx * len(real_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                sum(g_OT_loss_value_batch) / cfg.NUM_DISTRIBUTION,
                sum(remaining_f_loss_value_batch) / cfg.NUM_DISTRIBUTION,
                mu_2moment_loss_value_batch,
                miu_mean_value_batch.mean().tolist(),
                miu_var_value_batch.mean().tolist(),
                g_constraints_loss_value_batch,
                w2_loss_value_batch
            ))

    w2_loss_value_epoch /= len(train_loader)
    g_OT_loss_value_epoch[:] = [
        item / len(train_loader) for item in g_OT_loss_value_epoch]
    g_constraints_loss_value_epoch /= len(train_loader)
    remaining_f_loss_value_epoch[:] = [
        item / len(train_loader) for item in remaining_f_loss_value_epoch]
    mu_2moment_loss_value_epoch /= len(train_loader)
    miu_mean_value_epoch /= len(train_loader)
    miu_var_value_epoch /= len(train_loader)
    if cfg.high_dim_flag:
        results.add(epoch=epoch,
                    w2_loss_train_samples=w2_loss_value_epoch,
                    g_OT_train_loss=g_OT_loss_value_epoch,
                    g_constraints_train_loss=g_constraints_loss_value_epoch,
                    remaining_f_train_loss=remaining_f_loss_value_epoch,
                    mu_2moment_train_loss=mu_2moment_loss_value_epoch
                    )
    else:
        results.add(epoch=epoch,
                    w2_loss_train_samples=w2_loss_value_epoch,
                    g_OT_train_loss=g_OT_loss_value_epoch,
                    g_constraints_train_loss=g_constraints_loss_value_epoch,
                    remaining_f_train_loss=remaining_f_loss_value_epoch,
                    mu_2moment_train_loss=mu_2moment_loss_value_epoch,
                    miu_mean_train=miu_mean_value_epoch.tolist(),
                    miu_var_train=miu_var_value_epoch.tolist()
                    )
    results.save()


if __name__ == '__main__':
    dim = 2
    num_samples = 5000
    num_measures = 5
    seed = 1009
    cfg = Cfg_class(DIM = dim, NUM_DISTRIBUTION=num_measures)

    csv_path = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)

    # gpus_choice = GPUtil.getFirstAvailable(
    #     order='random', maxLoad=0.5, maxMemory=0.5, attempts=5, interval=900, verbose=False)
    # PTU.set_gpu_mode(True, gpus_choice[0])
    PTU.set_gpu_mode(False, 0)

    cfg.INPUT_DIM = dim
    cfg.OUTPUT_DIM = cfg.INPUT_DIM
    cfg.NUM_DISTRIBUTION = num_measures
    cfg.high_dim_flag = False
    cfg.epochs = 500
    _, _, results, testresults = LLU.init_path(cfg)
    results_save_path = f'./dim{dim}_data/ICNN_Fan_outputs/CNX_outputs/Custom_dim{dim}_measures{num_measures}'
    model_save_path = results_save_path + '/storing_models'
    
    # kwargs = {'num_workers': 4, 'pin_memory': True}
    kwargs = {'pin_memory': True}

    
    convex_f, convex_g, generator_h = g_NN.generate_FixedWeight_NN(cfg)

    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            Initialization with some positive parameters
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    f_positive_params = []
    g_positive_params = []

    for i in range(cfg.NUM_DISTRIBUTION):
        for p in list(convex_f[i].parameters()):
            if hasattr(p, 'be_positive'):
                f_positive_params.append(p)

        for p in list(convex_g[i].parameters()):
            if hasattr(p, 'be_positive'):
                g_positive_params.append(p)

        # convex_f[i].cuda(PTU.device)
        # convex_g[i].cuda(PTU.device)
        convex_f[i].cpu()
        convex_g[i].cpu()
    # generator_h.cuda(PTU.device)
    generator_h.cpu()

    optimizer_f = []
    optimizer_g = []
    if cfg.optimizer is 'Adam':
        for i in range(cfg.NUM_DISTRIBUTION):
            optimizer_f.append(optim.Adam(convex_f[i].parameters(), lr=cfg.LR_f))
            optimizer_g.append(
                optim.Adam(convex_g[i].parameters(), lr=cfg.LR_g))
        optimizer_h = optim.Adam(
            generator_h.parameters(),
            lr=cfg.LR_h)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        Real Training Process
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    for epoch in range(1, cfg.epochs + 1):
        # Start training
        train(epoch, csv_path)
        if cfg.schedule_learning_rate:
            if epoch % cfg.lr_schedule_per_epoch == 0:
                for i in range(cfg.NUM_DISTRIBUTION):
                    optimizer_f[i].param_groups[0]['lr'] *= cfg.lr_schedule_scale
                    optimizer_g[i].param_groups[0]['lr'] *= cfg.lr_schedule_scale
                optimizer_h.param_groups[0]['lr'] *= cfg.lr_schedule_scale

        LLU.dump_nn(generator_h, convex_f, convex_g, epoch,
                    model_save_path, num_distribution=cfg.NUM_DISTRIBUTION, save_f=cfg.save_f)
            


