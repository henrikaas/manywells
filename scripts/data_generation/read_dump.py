"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 07 March 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no
"""

import os
import pickle

import pandas as pd


####################################################################
# Read data files, compile DataFrames and store them as .csv
####################################################################
def read_dump(dataset_name, data_path):
    # dataset_name = 'iteration_100'
    # data_path = '../results/param_set_1/run9'
    dump_path = os.path.join(data_path, dataset_name, 'dump')
    files = [f for f in os.listdir(dump_path) if os.path.isfile(os.path.join(dump_path, f))]

    df_all = None
    df_config = pd.DataFrame()
    name_map = dict()

    well_depths = []

    well_n = 0
    for i, fn in enumerate(files):
        with open(os.path.join(dump_path, fn), 'rb') as f:
            obj = pickle.load(f)

        # Add data
        # print(obj)
        # print(type(obj['data']))

        if obj['config'].wp.L in well_depths:
            print(f'Skipping well {i} with depth {obj["config"].wp.L} m, already exists.')
            continue

        df_i = obj['data']

        df_i['ID'] = well_n  # Well ID
        name_map[well_n] = fn

        if df_all is None:
            df_all = df_i.copy()
        else:
            df_all = pd.concat([df_all, df_i], ignore_index=True)

        # Add config
        c = obj['config']
        well_depths.append(c.wp.L)
        # print(c)
        # print(type(c))
        c_dict = {
            'ID': well_n,
            'wp.L': c.wp.L,
            'wp.D': c.wp.D,
            'wp.rho_l': c.wp.rho_l,
            'wp.R_s': c.wp.R_s,
            'wp.cp_g': c.wp.cp_g,
            'wp.cp_l': c.wp.cp_l,
            'wp.f_D': c.wp.f_D,
            'wp.h': c.wp.h,
            'wp.inflow.class_name': c.wp.inflow.__class__.__name__,
            'wp.inflow.w_l_max': c.wp.inflow.w_l_max,  # NOTE: Assumes Vogel's IPR
            'wp.inflow.f_g': c.wp.inflow.f_g,
            'wp.choke.class_name': c.wp.choke.__class__.__name__,
            'wp.choke.K_c': c.wp.choke.K_c,
            'wp.choke.cpr': c.wp.choke.cpr,
            'wp.choke.chk_profile': c.wp.choke.chk_profile,
            'bc.p_r': c.bc.p_r,
            'bc.p_s': c.bc.p_s,
            'bc.T_r': c.bc.T_r,
            'bc.T_s': c.bc.T_s,
            'bc.u': c.bc.u,
            'bc.w_lg': c.bc.w_lg,
            'gas.name': c.gas.name,
            'gas.R_s': c.gas.R_s,
            'gas.cp': c.gas.cp,
            'oil.name': c.oil.name,
            'oil.rho': c.oil.rho,
            'oil.cp': c.oil.cp,
            'water.name': c.water.name,
            'water.rho': c.water.rho,
            'water.cp': c.water.cp,
            'fraction.gas': c.fractions[0],
            'fraction.oil': c.fractions[1],
            'fraction.water': c.fractions[2],
            'has_gas_lift': c.has_gas_lift,
        }
        df_config_i = pd.DataFrame([c_dict])
        df_config = pd.concat([df_config, df_config_i], ignore_index=True)

        well_n += 1

    #print(name_map)
    print()
    print('Data:')
    print(df_all)
    print()
    print('Configurations:')
    print(df_config)

    # Store data
    filename = f'{dataset_name}.csv'
    file_path = os.path.join(data_path, dataset_name, filename)
    df_all.to_csv(file_path, index=False)

    # Store configurations
    c_filename = f'{dataset_name}_config.csv'
    c_file_path = os.path.join(data_path, dataset_name, c_filename)
    df_config.to_csv(c_file_path, index=False)

if __name__ == '__main__':
    # dataset_name = 'iteration_100'

    # main_path = '../results/param_set_2/'
    main_path = '../results/prod_runs/'

    n_runs = 20  # Number of runs to read
    for k in range(2, 3):
        for i in range(n_runs):
            data_path = f"{main_path}/gaslift_choke50%_highnoise/run{i}"
            for j in range(10, 101, 10):
                dataset_name = f'iteration_{j}'
                if not os.path.exists(os.path.join(data_path, dataset_name)):
                    print(f"Reached end of runs for {dataset_name} in {data_path} at {j}.")
                    break
                read_dump(dataset_name, data_path)
