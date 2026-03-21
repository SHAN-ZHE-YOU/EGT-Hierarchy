#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ==========================================
# Heatmap Generators (General Population)
# ==========================================

def Plot_aveCD_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats):
    x = np.array(c_set)
    y = np.array(G_set)
    X, Y = np.meshgrid(x, y)

    # Create the original Z array with y-axis inverted
    original_Z = []
    for G in reversed(G_set):  # Reverse the order of G_set
        z_row = []
        for c in c_set:
            name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)
            
            z_row.append(data[name]['ave_CD_ratio'])
        original_Z.append(z_row)

    original_Z_array = np.array(original_Z)

    n_square = 20# Reshape Z to an 8x8 grid
    step_size = 10  # 201 / 10 is approximately 20
    reshaped_Z = np.zeros((n_square, n_square))
    for i in range(n_square):
        for j in range(n_square):
            # Calculate average of each 25x25 block
            block = original_Z_array[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
            average = np.mean(block)
            reshaped_Z[i, j] = average

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0.25,0.5,0.75,1], fontsize=18)
    plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)

    im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0, vmax=0.55)
    cbar = fig.colorbar(im, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.ax.tick_params(labelsize=16)
    
    if 'png' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_CD_heatmap.png', dpi=600, format='png')
    if 'eps' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_CD_heatmap.eps', dpi=600, format='eps')
    if 'pdf' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_CD_heatmap.pdf', dpi=600, format='pdf')
  
def Plot_aveStep_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats):
    x = np.array(c_set)
    y = np.array(G_set)
    X, Y = np.meshgrid(x, y)

    # Create the original Z array with y-axis inverted
    original_Z = []
    for G in reversed(G_set):  # Reverse the order of G_set
        z_row = []
        for c in c_set:
            name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)
            z_row.append(np.log(data[name]['ave_step']))
        original_Z.append(z_row)

    original_Z_array = np.array(original_Z)

    n_square = 20  # Reshape Z to a 20x20 grid
    step_size = 10  # 201 / 20 is approximately 10
    reshaped_Z = np.zeros((n_square, n_square))
    for i in range(n_square):
        for j in range(n_square):
            # Calculate average of each 10x10 block
            block = original_Z_array[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
            average = np.mean(block)
            reshaped_Z[i, j] = average

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0.25,0.5,0.75,1], fontsize=18)
    plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)

    im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0, vmax=3.5)
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.tick_params(labelsize=16)

    if 'png' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Step_heatmap.png', dpi=600, format='png')
    if 'eps' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Step_heatmap.eps', dpi=600, format='eps')
    if 'pdf' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Step_heatmap.pdf', dpi=600, format='pdf')

def Plot_aveLevels_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats):
    x = np.array(c_set)
    y = np.array(G_set)
    original_Z = []
    for G in reversed(G_set):
        z_row = []
        for c in c_set:
            name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)
            z_row.append(data[name]['ave_num_levels'])
        original_Z.append(z_row)

    original_Z_array = np.array(original_Z)
    reshaped_Z = np.zeros((20, 20))
    step_size = 10
    for i in range(20):
        for j in range(20):
            block = original_Z_array[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
            average = np.mean(block)
            reshaped_Z[i, j] = average

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0.25,0.5,0.75,1], fontsize=18)
    plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)
    
    im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=1.9, vmax=4)
    cbar = fig.colorbar(im, ticks=[2, 2.5, 3, 3.5, 4])
    cbar.ax.tick_params(labelsize=16)


    if 'png' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Levels_heatmap.png', dpi=600, format='png')
    if 'eps' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Levels_heatmap.eps', dpi=600, format='eps')
    if 'pdf' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Levels_heatmap.pdf', dpi=600, format='pdf')

def Plot_aveEntropy_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_formats):
    x = np.array(c_set)
    y = np.array(G_set)
    original_Z = []
    for G in reversed(G_set):
        z_row = []
        for c in c_set:
            name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)
            z_row.append(data[name]['ave_entropy'])
        original_Z.append(z_row)

    original_Z_array = np.array(original_Z)
    reshaped_Z = np.zeros((20, 20))
    step_size = 10
    for i in range(20):
        for j in range(20):
            block = original_Z_array[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
            average = np.mean(block)
            reshaped_Z[i, j] = average

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0.25,0.5,0.75,1], fontsize=18)
    plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)
    
    im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0225, vmax=0.0475)
    cbar = fig.colorbar(im, ticks=[0.025, 0.03, 0.035, 0.04, 0.045])
    cbar.ax.tick_params(labelsize=16)

    if 'png' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Entropy_heatmap.png', dpi=600, format='png')
    if 'eps' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Entropy_heatmap.eps', dpi=600, format='eps')
    if 'pdf' in output_formats:
        plt.savefig(f'plots/n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Entropy_heatmap.pdf', dpi=600, format='pdf')