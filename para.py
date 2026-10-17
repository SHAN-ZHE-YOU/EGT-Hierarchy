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
from concurrent.futures import ProcessPoolExecutor, as_completed

from hierarchy_package import (
    states, ini_levels, L_distribution, hierarchicalness, 
    level_next, income_CD, strategy_update_fermi
)

@dataclass
class SimConfig:
    """Pilot Simulation configuration parameters"""
    alpha_set: List[float] = field(default_factory=lambda: [0.2, 0.8])
    n_set: List[int] = field(default_factory=lambda: [10, 100])
    test_num: int = 100
    b: int = 1
    ini_pC: float = 0.9
    maxstep: int = 500
    
    # 21x21 grid
    c_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 21)))
    G_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 21)))
    
    r: float = 2.5           
    epsilon: float = 0.0     
    beta: float = 1.0        

logging.basicConfig(filename='simulation_pilot.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

def simulate_single_config(args):
    """獨立的工作函數：負責跑單一個 (n, alpha, c, G) 組合的 100 次測試"""
    n, alpha, c, G, config = args
    
    # 重要：為每個獨立進程重新初始化亂數種子，避免所有核心算出長得一模一樣的隨機結果
    np.random.seed()
    import random
    random.seed()
    
    name = (f"n={n},alpha={alpha},ini_pC={config.ini_pC},"
            f"c/b={round(c/config.b, 11)},G={round(G,11)},testnum={config.test_num}")
    
    result = {
        'final_L_i': [], 'CD_ratio': [], 'num_levels': [],
        'times_overMaxG': [], 'steps': [], 'entropy': [],
        'H_trajectories': []
    }
    
    quota = max(1, math.ceil(n / 100))
    
    for i in range(config.test_num):
        step = 1
        overMaxG = 0 
        H_history = []
        
        S_i = states(config.ini_pC, n)
        L_i = ini_levels(n)
        distr = L_distribution(L_i, S_i, n)
        H = hierarchicalness(distr, n)
        H_history.append(H)
        
        L_i = level_next(L_i, S_i, distr, n, G, H, alpha, epsilon=config.epsilon, quota=quota)
        distr = L_distribution(L_i, S_i, n)
        H = hierarchicalness(distr, n)
        H_history.append(H)
        
        income_i, mark, W_C, W_D = income_CD(distr, G, n, config.b, c, H, config.r)
        if mark == 1: overMaxG += 1
 
        while len(set(S_i)) > 1 and step <= config.maxstep - 1:
            S_i = strategy_update_fermi(S_i, income_i, beta=config.beta)
            L_i = level_next(L_i, S_i, distr, n, G, H, alpha, epsilon=config.epsilon, quota=quota)
            distr = L_distribution(L_i, S_i, n)
            H = hierarchicalness(distr, n)
            H_history.append(H)
            
            income_i, mark, W_C, W_D = income_CD(distr, G, n, config.b, c, H, config.r)
            if mark == 1: overMaxG += 1
            step += 1 
    
        result['final_L_i'].append(L_i)
        result['CD_ratio'].append(round(S_i.count('C') / len(S_i), 4))
        result['num_levels'].append(len(list(set(L_i))))
        result['steps'].append(step)
        result['times_overMaxG'].append(overMaxG / step)
        result['entropy'].append(entropy(L_i, ini_levels(n)))
        result['H_trajectories'].append(H_history)
        
    result['ave_overMaxG'] = np.mean(result['times_overMaxG'])  
    result['ave_CD_ratio'] = round(np.mean(result['CD_ratio']), 3)  
    result['ave_num_levels'] = np.mean(result['num_levels'])  
    result['ave_step'] = np.mean(result['steps'])  
    result['ave_entropy'] = np.mean(result['entropy'])  

    return name, result

def run_simulation(config: SimConfig):
    # 自動偵測可用的 CPU 核心數，並保留 2 個核心給作業系統，以免整台機器卡死
    max_workers = max(1, os.cpu_count() - 2)
    logging.info(f"Using {max_workers} CPU cores for multiprocessing.")
    print(f"啟動多核心運算，使用核心數：{max_workers}")
    
    for n in config.n_set:
        for alpha in config.alpha_set:
            data = {}
            data['title'] = [n, alpha, config.ini_pC, config.b, config.c_set, config.G_set, config.test_num]
            logging.info(f"Starting Pilot: n={n}, alpha={alpha}, test_num={config.test_num}")
            
            # 把 21x21 = 441 個座標點轉換成獨立的任務清單
            tasks = []
            for c in config.c_set:
                for G in config.G_set:
                    tasks.append((n, alpha, c, G, config))
            
            # 將任務丟入工作池，並與 tqdm 綁定顯示進度條
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(simulate_single_config, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Simulating n={n}, a={alpha}"):
                    name, result = future.result()
                    data[name] = result
            
            os.makedirs('artifacts', exist_ok=True)
            filename = f'artifacts/pilot_alpha_{alpha}_n_{n}_21x21.pickle'
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
            
            logging.info(f"Pilot for n={n}, alpha={alpha} saved to {filename}")

if __name__ == '__main__':
    config = SimConfig()
    run_simulation(config)
    print("Multiprocessing Pilot simulation completed! Check the artifacts folder.")