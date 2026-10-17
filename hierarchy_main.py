#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pickle
import logging
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List
from scipy.stats import entropy
from tqdm import tqdm

from hierarchy_package import (
    states, ini_levels, L_distribution, hierarchicalness, 
    level_next, income_CD, strategy_update_fermi
)

@dataclass
class SimConfig:
    """Pilot Simulation configuration parameters"""
    # 依照教授要求調整 Pilot 規格
    alpha_set: List[float] = field(default_factory=lambda: [0.2, 0.8])
    n_set: List[int] = field(default_factory=lambda: [10, 100])
    test_num: int = 100      # 測試次數降為 100 次
    b: int = 1
    ini_pC: float = 0.9
    maxstep: int = 500
    
    # 網格降到 21x21
    c_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 21)))
    G_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 21)))
    
    # 新增的模型參數
    r: float = 2.5           # 公共財乘數 (先用 2 到 3 中間值)
    epsilon: float = 0.0     # 偷懶者的升遷機率
    beta: float = 1.0        # Fermi imitation 的選擇強度

logging.basicConfig(filename='simulation_pilot.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

def run_simulation(config: SimConfig):
    # 改為依序跑不同的 n 與 alpha
    for n in config.n_set:
        for alpha in config.alpha_set:
            
            data = {}
            data['title'] = [n, alpha, config.ini_pC, config.b, config.c_set, config.G_set, config.test_num]
            logging.info(f"Starting Pilot: n={n}, alpha={alpha}, test_num={config.test_num}")
            
            start_time_all = time.time()
            total_iterations = len(config.c_set) * len(config.G_set)
            
            # 設定升遷名額：先試每輪一人和 max(1, ⌈N/100⌉) 
            quota = max(1, math.ceil(n / 100))
            
            with tqdm(total=total_iterations, desc=f"Simulating n={n}, a={alpha}") as pbar:
                for c in config.c_set:
                    for G in config.G_set:
                        start_time = time.time()
                        
                        name = (f"n={n},alpha={alpha},ini_pC={config.ini_pC},"
                                f"c/b={round(c/config.b, 11)},G={round(G,11)},testnum={config.test_num}")
                        
                        data[name] = {
                            'final_L_i': [], 'CD_ratio': [], 'num_levels': [],
                            'times_overMaxG': [], 'steps': [], 'entropy': [],
                            'H_trajectories': [] # 新增追蹤 H 軌跡
                        }
                        
                        for i in range(config.test_num):
                            step = 1
                            overMaxG = 0 
                            H_history = [] # 記錄這單次跑的 H 變化
                            
                            # 初始化
                            S_i = states(config.ini_pC, n)
                            L_i = ini_levels(n)
                            distr = L_distribution(L_i, S_i, n)
                            H = hierarchicalness(distr, n)
                            H_history.append(H)
                            
                            # 第一回合升遷
                            L_i = level_next(L_i, S_i, distr, n, G, H, alpha, epsilon=config.epsilon, quota=quota)
                            distr = L_distribution(L_i, S_i, n)
                            H = hierarchicalness(distr, n)
                            H_history.append(H)
                            
                            # 第一回合分配 (取得個體 income_i)
                            income_i, mark, W_C, W_D = income_CD(distr, G, n, config.b, c, H, config.r)
                            if mark == 1: overMaxG += 1
                     
                            # 進入迴圈 (原本是用 pC_Next，現在改用 strategy_update_fermi)
                            while len(set(S_i)) > 1 and step <= config.maxstep - 1:
                                # 個體更新策略
                                S_i = strategy_update_fermi(S_i, income_i, beta=config.beta)
                                
                                # 升遷
                                L_i = level_next(L_i, S_i, distr, n, G, H, alpha, epsilon=config.epsilon, quota=quota)
                                distr = L_distribution(L_i, S_i, n)
                                H = hierarchicalness(distr, n)
                                H_history.append(H)
                                
                                # 分配       
                                income_i, mark, W_C, W_D = income_CD(distr, G, n, config.b, c, H, config.r)
                                if mark == 1: overMaxG += 1
                                step += 1 
                        
                            # 記錄這 1 run 的結果
                            data[name]['final_L_i'].append(L_i)
                            data[name]['CD_ratio'].append(round(S_i.count('C') / len(S_i), 4))
                            data[name]['num_levels'].append(len(list(set(L_i))))
                            data[name]['steps'].append(step)
                            data[name]['times_overMaxG'].append(overMaxG / step)
                            data[name]['entropy'].append(entropy(L_i, ini_levels(n)))
                            data[name]['H_trajectories'].append(H_history) # 存入 H 軌跡
                            
                        # 計算平均
                        data[name]['ave_overMaxG'] = np.mean(data[name]['times_overMaxG'])  
                        data[name]['ave_CD_ratio'] = round(np.mean(data[name]['CD_ratio']), 3)  
                        data[name]['ave_num_levels'] = np.mean(data[name]['num_levels'])  
                        data[name]['ave_step'] = np.mean(data[name]['steps'])  
                        data[name]['ave_entropy'] = np.mean(data[name]['entropy'])  

                        pbar.update(1)
                        
            # 每個 (n, alpha) 組合跑完就存一個檔
            os.makedirs('artifacts', exist_ok=True)
            filename = f'artifacts/pilot_alpha_{alpha}_n_{n}_21x21.pickle'
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
            
            logging.info(f"Pilot for n={n}, alpha={alpha} saved to {filename}")

if __name__ == '__main__':
    config = SimConfig()
    run_simulation(config)
    print("Pilot simulation completed! Check the artifacts folder.")