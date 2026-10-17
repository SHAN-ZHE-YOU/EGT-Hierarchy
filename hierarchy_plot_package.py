#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from collections import Counter
from scipy.stats import entropy

def Save_plot(fig, file_name, output_format):
    
    dir_name = 'plots'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    if 'png' in output_format:
        fig.savefig(f'{dir_name}/{file_name}.png', dpi=600, format='png')
    elif 'eps' in output_format:
        fig.savefig(f'{dir_name}/{file_name}.eps', dpi=600, format='eps')
    elif 'pdf' in output_format:
        fig.savefig(f'{dir_name}/{file_name}.pdf', dpi=600, format='pdf')

def Plot_aveCD_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
    Z = np.zeros((len(G_set), len(c_set)))
    for i, G in enumerate(reversed(G_set)):
        for j, c in enumerate(c_set):
            # 統一使用 round(x, 11) 對齊 main.py 的 Key
            name = f'n={n},alpha={alpha},ini_pC={ini_pC},c/b={round(c/b,11)},G={round(G,11)},testnum={test_num}'
            Z[i, j] = data[name]['ave_CD_ratio'] if name in data else 0

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)

    im = ax.imshow(Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)

    Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_CD_heatmap', output_format)

def Plot_aveStep_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
    Z = np.zeros((len(G_set), len(c_set)))
    for i, G in enumerate(reversed(G_set)):
        for j, c in enumerate(c_set):
            name = f'n={n},alpha={alpha},ini_pC={ini_pC},c/b={round(c/b,11)},G={round(G,11)},testnum={test_num}'
            # 取 ln()，500步的上限大約是 6.2
            Z[i, j] = np.log(data[name]['ave_step']) if name in data else 0

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)

    im = ax.imshow(Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0, vmax=6.5)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)

    Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_Step_heatmap', output_format)

def Plot_aveLevels_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
    Z = np.zeros((len(G_set), len(c_set)))
    for i, G in enumerate(reversed(G_set)):
        for j, c in enumerate(c_set):
            name = f'n={n},alpha={alpha},ini_pC={ini_pC},c/b={round(c/b,11)},G={round(G,11)},testnum={test_num}'
            Z[i, j] = data[name]['ave_num_levels'] if name in data else 1

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    
    im = ax.imshow(Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=1.0, vmax=4.0)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)

    Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_num_levels_heatmap', output_format)

# def Plot_aveEntropy_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
#     x = np.array(c_set)
#     y = np.array(G_set)
#     original_Z = []
#     for G in reversed(G_set):
#         z_row = []
#         for c in c_set:
#             name = 'n='+str(n)+',alpha='+str(alpha)+',ini_pC='+str(ini_pC)+',c/b='+str(round(c/b,4))+',G='+str(G)+',testnum='+str(test_num)
#             z_row.append(data[name]['ave_entropy'])
#         original_Z.append(z_row)

#     original_Z_array = np.array(original_Z)
#     reshaped_Z = np.zeros((20, 20))
#     step_size = 10
#     for i in range(20):
#         for j in range(20):
#             block = original_Z_array[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
#             average = np.mean(block)
#             reshaped_Z[i, j] = average

#     fig = plt.figure(figsize=(9, 7.5))
#     ax = fig.add_subplot(111)
#     plt.xlabel('c/b', fontsize=20)
#     plt.ylabel('G', fontsize=20)
#     plt.xticks([0.25,0.5,0.75,1], fontsize=18)
#     plt.yticks([0,0.25,0.5,0.75,1], fontsize=18)
    
#     im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0225, vmax=0.0475)
#     cbar = fig.colorbar(im, ticks=[0.025, 0.03, 0.035, 0.04, 0.045])
#     cbar.ax.tick_params(labelsize=16)
    
#     Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_average_entropy_heatmap', output_format)

# ... 保留原有的 Save_plot, Plot_aveCD_heatmap, Plot_aveStep_heatmap, Plot_aveLevels_heatmap ...

def Plot_aveEntropy_heatmap(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
    """
    修正後的 Entropy 計算：直接從 final_L_i 計算各階層人數比例的 Shannon Entropy
    理論上限：3層約為 ln(3) = 1.098
    """
    x = np.array(c_set)
    y = np.array(G_set)
    
    # 計算正確的 Shannon Entropy
    reshaped_Z = np.zeros((len(G_set), len(c_set)))
    
    for i, G in enumerate(reversed(G_set)):
        for j, c in enumerate(c_set):
            name = f'n={n},alpha={alpha},ini_pC={ini_pC},c/b={round(c/b,11)},G={round(G,11)},testnum={test_num}'
            if name in data:
                # 提取這 100 runs 的最終階層分佈
                run_entropies = []
                for L_i in data[name]['final_L_i']:
                    counts = list(Counter(L_i).values())
                    probs = [count / n for count in counts]
                    run_entropies.append(entropy(probs)) # 預設 base=e
                reshaped_Z[i, j] = np.mean(run_entropies)
            else:
                reshaped_Z[i, j] = 0.0

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111)
    plt.xlabel('c/b', fontsize=20)
    plt.ylabel('G', fontsize=20)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=18)
    
    # 注意這裡的 vmin 與 vmax 已經放寬以符合 ln(3) 的範圍
    im = ax.imshow(reshaped_Z, interpolation='nearest', cmap=cm.jet, extent=[0,1,0,1], vmin=0.0, vmax=1.1)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    
    Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_corrected_entropy_heatmap', output_format)

def Plot_H_trajectory(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format):
    """
    新增的函式：畫出 H (階層穩定度) 隨時間的變化軌跡，用於診斷教授提到的鎖定效應。
    我們挑選一組代表性的參數，例如 G=0.5, c/b=0.5
    """
    # 找尋最接近 0.5 的 G 與 c/b
    target_G = min(G_set, key=lambda x: abs(x - 0.5))
    target_c = min(c_set, key=lambda x: abs(x - 0.5))
    
    name = f'n={n},alpha={alpha},ini_pC={ini_pC},c/b={round(target_c/b,11)},G={round(target_G,11)},testnum={test_num}'
    
    if name not in data:
        print(f"Data not found for trajectory plotting: {name}")
        return

    trajectories = data[name]['H_trajectories']
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 畫出前 10 次 run 的軌跡，避免畫面太亂
    for idx, traj in enumerate(trajectories[:10]):
        ax.plot(traj, alpha=0.6, label=f'Run {idx+1}' if idx < 3 else "")
        
    # 畫出這 100 runs 的平均軌跡
    max_len = max(len(t) for t in trajectories)
    padded_trajs = [t + [t[-1]] * (max_len - len(t)) for t in trajectories]
    mean_traj = np.mean(padded_trajs, axis=0)
    
    ax.plot(mean_traj, color='black', linewidth=3, label='Mean H')
    
    plt.xlabel('Step (Iterations)', fontsize=16)
    plt.ylabel('Hierarchical Score (H)', fontsize=16)
    plt.title(f'H Trajectory (n={n}, alpha={alpha}, G=0.5, c/b=0.5)', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    Save_plot(fig, f'n={n}_alpha={alpha}_inipC={ini_pC}_testnum={test_num}_H_trajectory', output_format)