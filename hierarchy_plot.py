#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weiyujie
"""

import os
import argparse
import pickle
import numpy as np
from hierarchy_plot_package import *

def generate_plots(file_name, plot_type, output_formats):
    """
    Load the pickle file and generate requested plots.
    """
    print(f"Loading data from {file_name}...")
    try:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return

    # Extract parameters from the saved dictionary
    # Assuming title format: [n, alpha, ini_pC, b, c_set, G_set]
    n = data['title'][0]
    alpha = data['title'][1]
    ini_pC = data['title'][2]
    b = data['title'][3]
    c_set = data['title'][4]
    G_set = data['title'][5]
    test_num = data['title'][6]
    

    print(f"Parameters loaded: n={n}, alpha={alpha}, ini_pC={ini_pC}")

    # ==========================================
    # Plotting Logic
    # ==========================================
    
    os.makedirs('plots', exist_ok=True)
    
    if plot_type in ['aveCD', 'all']:
        print("Plotting Average CD Heatmap...")
        Plot_aveCD_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats)

    if plot_type in ['aveStep', 'all']:
        print("Plotting Average Step Heatmap...")
        Plot_aveStep_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats)

    if plot_type in ['aveLevels', 'all']:
        print("Plotting Average Levels Heatmap...")
        Plot_aveLevels_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats)

    if plot_type in ['aveEntropy', 'all']:
        print("Plotting Average Entropy Heatmap...")
        Plot_aveEntropy_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats)

    print("Plotting completed successfully!")


if __name__ == '__main__':
    # Set up argument parser for Command Line Interface (CLI)
    parser = argparse.ArgumentParser(description="Generate heatmaps from Evolutionary Game Theory simulation data.")
    
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=True, 
        help="Path to the .pickle data file (e.g., alpha_0.5_inipC_0.9_201x201_CD_n_10_testnum=200.pickle)"
    )
    
    parser.add_argument(
        '-t', '--type', 
        type=str, 
        default='all',
        choices=['aveCD', 'aveStep', 'aveLevels', 'aveOverMaxG', 'aveEntropy', 'all'],
        help="Type of plot to generate. Default is 'all'. Options: aveCD, aveStep, aveLevels, aveOverMaxG, aveEntropy"
    )

    parser.add_argument(
        '--format', 
        type=str,
        nargs='+',
        default=['png'],
        choices=['pdf', 'png', 'eps'],
        help="Output file format(s), e.g. --format pdf png eps"
    )

    args = parser.parse_args()

    # Execute plotting function
    generate_plots(file_name=args.file, plot_type=args.type, output_formats=args.format)


# # 在這裡改參數！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

# test_num = 200

# start_index = 0
# end_index = 1

# # [10, 50, 100, 1000]
# N = [50]

# # [0.2, 0.5, 0.8]
# A = [0.5]

# # [0, 1, 2, 3, 4]
# B = [0, 1, 2, 3, 4]
# C = []
# D = []

# # --------------------------------------------------------------------------------

# for n in N:
#     for alpha in A:
        
#         file_name = f'alpha_{alpha}_inipC_0.9_201x201_CD_n_{n}_testnum={test_num}.pickle'
        
#         for i in range(0, 5 + 1):
#             case   = B[i]   #  0: aveCD, 1: aveStep, 2: aveLevels, 3: aveEntropy

#             print(f'for n = {n} test_num = {test_num} alpha = {alpha} t = {case}')

#             with open(file_name, 'rb') as handle:
#                 data = pickle.load(handle)

#             n = data['title'][0]
#             alpha = data['title'][1]
#             ini_pC = data['title'][2]
#             b = data['title'][3]
#             c_set = data['title'][4]
#             G_set = data['title'][5]

#             if case == 0:
#                 Plot_aveCD_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data)
#             elif case == 1:
#                 Plot_aveStep_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data)
#             elif case == 2:
#                 Plot_aveLevels_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data)
#             elif case == 3:
#                 Plot_aveEntropy_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data)
#             else:
#                 print("Don't consider ave this time.")

#             Z = []
#             for G in G_set:
#                 z=[]
#                 for c in c_set:
                
#                     name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)

#                     z.append(data[name]['C_ave_step'])
#                 Z.append(np.log(z))