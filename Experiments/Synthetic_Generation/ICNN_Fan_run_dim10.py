from .ICNN_Fan_run_dim2 import *

if __name__ == '__main__':
    dim = 10
    num_samples = 5000
    num_measures = 10
    seed = 1009
    cfg = Cfg_class(DIM = dim, NUM_DISTRIBUTION=num_measures)

    csv_path = "./WB_Algo/Experiments/Synthetic_Generation/dim10_data/input_samples/csv_files"
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
            


