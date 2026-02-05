import Algorithms.CWB_Li.cwb.templates

import numpy as np
import yaml
import os
import scipy.linalg
import pickle
import argparse
import subprocess

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

g_mixture_template_name = 'mixture_template.yaml'
g_gaussian_template_name = 'gaussian_template.yaml'
g_cube_template_name = 'cube_template.yaml'
g_poisson_template_name = 'poisson_template.yaml'
g_customized_template_name = 'customized_template.yaml'

g_sample_dir = 'barycenter_dir'
g_sample_npy = 'barycenter.npy'
g_conf_filename = 'config.yaml'

def get_template_name(exp):
    if exp == 'gaussian':
        return g_gaussian_template_name
    elif exp == 'mixture':
        return g_mixture_template_name
    elif exp == 'cube':
        return g_cube_template_name
    elif exp == 'poisson':
        return g_poisson_template_name
    elif exp == 'BikeSharing' or exp == 'SyntheticGeneration':
        return g_customized_template_name
    else:
        raise Exception('No template for experiment {}!'.format(exp))

def run(exp, dim, sample_count, working_dir, data_dir, result_dir, result_filename, input_csv_path, num_measures, MC_size):
    '''
    data_dir: directory containing the input distributions (csv files)
    result_dir: directory to save the output barycenter samples
    '''
    # original_working_dir = os.getcwd()
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with pkg_resources.path(Algorithms.CWB_Li.cwb.templates, get_template_name(exp)) as template_file:
        conf = yaml.safe_load(open(template_file))

    conf['exp'] = exp
    conf['point_dim'] = dim

    source_list = []
    # data_files = sorted([p for p in os.listdir(data_dir) if p.endswith('.pkl')])
    # # ---- CSV inputs instead of PKL ----
    # data_files = sorted([
    #     p for p in os.listdir(data_dir)
    #     if p.lower().endswith(".csv")
    # ])

    # if len(data_files) == 0:
    #     raise FileNotFoundError(f"No .csv files found in data_dir={data_dir}")

    num_sources = num_measures
    weight = 1 / num_sources
    for i in range(num_sources):
        name = 'source{}'.format(i)
        source_list.append({
            'name': name,
            'dim': dim,
            # 'pkl_path': os.path.abspath(os.path.join(data_dir, p)),
            #'csv_path': os.path.abspath(os.path.join(data_dir, p)),
            'weight': weight,
            'shape' : 'customized_sampler'})  # customized_sampler indicates we will use input_csv_path to sample points
    # ---------------------------------
    conf['num_sources'] = num_sources
    conf['input_csv_path'] = input_csv_path
    conf['potential_ckpt_dir'] = os.path.join(working_dir, 'checkpoints', 'potential')
    conf['map_ckpt_dir'] = os.path.join(working_dir, 'checkpoints', 'map')
    # --------------------------------
    conf['distribution_list'] = source_list
    conf['test'] = {
            'sample_barycenter': {
                    'num_samples': sample_count * MC_size,
                    'npy_dir': os.path.join(working_dir, g_sample_dir),
                    'npy_file': g_sample_npy,
                    'vis_name': 'barycenter.png',
                    'vis_dir': os.path.join(working_dir, 'barycenter_vis_dir')
                }
            }

    open(os.path.join(working_dir, g_conf_filename), 'w').write(yaml.dump(conf, indent=4))

    conf_path = os.path.join(working_dir, g_conf_filename)
    conf_path = os.path.abspath(conf_path)

    # os.chdir(working_dir)
    subprocess.run(['python -m Algorithms.CWB_Li.cwb.barycenter --train --test --reseed {}'.format(conf_path)], shell=True)
    # os.chdir(original_working_dir)
    src = os.path.join(working_dir, g_sample_dir, g_sample_npy)
    dst = os.path.join(result_dir, result_filename)

    # ensure destination directory exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected output not found: {src}")

    subprocess.run(
        ["cp", src, dst],
        check=True
    )

    # subprocess.run(['cp {} {}'.format(
    #     os.path.join(working_dir, g_sample_dir, g_sample_npy),
    #     os.path.join(result_dir, result_filename))], shell=True)


if __name__ == '__main__':
    # check the current directory:
    print("Current directory:", os.getcwd())

    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp', type=str)
    # parser.add_argument('dim', type=int)
    # parser.add_argument('sample_count', type=int)
    # parser.add_argument('working_dir', type=str)
    # parser.add_argument('data_dir', type=str)
    # parser.add_argument('result_dir', type=str)
    # parser.add_argument('result_filename', type=str)
    # args = parser.parse_args()

    # exp = args.exp
    # dim = args.dim
    # sample_count = args.sample_count
    # working_dir = args.working_dir
    # data_dir = args.data_dir
    # result_dir = args.result_dir
    # result_filename = args.result_filename
    exp = "poisson"
    dim = 2
    sample_count = 10
    num_measures = 5
    MC_size = 20
    working_dir = "./"
    data_dir = "../../../../WB_Data/Synthetic_Generation/dim2_data/input_samples/csv_files_InstanceTheta2000"
    print(sorted([
        p for p in os.listdir(data_dir)
        if p.lower().endswith(".csv")
    ]))

    result_dir = "result_dir"
    result_filename = "barycenter.npy"

    run(exp, dim, sample_count, working_dir, data_dir, result_dir, result_filename, data_dir, num_measures, MC_size)
