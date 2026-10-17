#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def merge_plots_by_params(plot_dir='plots', output_dir='merged_plots'):
    os.makedirs(output_dir, exist_ok=True)

    # 根據你的輸出，我們有這四組實驗參數
    configs = [
        ("10", "0.2"),
        ("10", "0.8"),
        ("100", "0.2"),
        ("100", "0.8")
    ]

    # 設定要合併的圖片順序與對應的子標題
    plot_types = [
        "average_CD_heatmap",
        "average_num_levels_heatmap",
        "average_Step_heatmap",
        "corrected_entropy_heatmap",
        "H_trajectory"
    ]
    
    plot_titles = [
        "Cooperation Ratio (CD)",
        "Average Hierarchical Levels",
        "Convergence Steps",
        "System Entropy (Corrected)",
        "Hierarchical Score (H) Trajectory"
    ]

    for n, alpha in configs:
        # 建立 2x3 的拼貼畫布
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Pilot Simulation: Group Size (N)={n}, Alpha={alpha}", fontsize=28, fontweight='bold')
        
        axes = axes.flatten()

        for i, p_type in enumerate(plot_types):
            filename = f"n={n}_alpha={alpha}_inipC=0.9_testnum=100_{p_type}.png"
            filepath = os.path.join(plot_dir, filename)

            if os.path.exists(filepath):
                img = mpimg.imread(filepath)
                axes[i].imshow(img)
                axes[i].set_title(plot_titles[i], fontsize=18, pad=10)
            else:
                axes[i].text(0.5, 0.5, f'Missing:\n{filename}', ha='center', va='center', fontsize=12)
            
            # 關閉圖片的座標軸
            axes[i].axis('off')

        # 關閉第 6 個多餘的空白子圖
        axes[5].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92) # 留出頂部大標題的空間

        output_path = os.path.join(output_dir, f"Merged_N{n}_Alpha{alpha}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"成功輸出合併圖表: {output_path}")

if __name__ == '__main__':
    merge_plots_by_params()