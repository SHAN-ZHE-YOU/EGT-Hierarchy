#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
Description: Evolutionary game theory simulation with hierarchy and Gini coefficient.
Refactored for readability and maintainability.
"""

import time
import pickle
import os
import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass

from hierarchy_package import (
    states, ini_levels, L_distribution, hierarchicalness, 
    level_next, income_CD, pC_Next
)

@dataclass
class SimulationConfig:
    alpha: float = 0.5     # (g/h) ratio
    n: int = 50           # group size (number of players)
    test_num: int = 200     # number of simulation repetitions
    b: float = 1.0         # benefit contributed to the common pool by cooperators
    ini_pC: float = 0.9    # initial proportion of cooperators in the population
    maxstep: int = 500     # maximum number of iterations (steps) for each simulation
    
    @property
    def initial_levels(self):
        """Helper to generate the initial level state cleanly."""
        return ini_levels(self.n)


def run_single_trial(c: float, G: float, config: SimulationConfig) -> dict:
    """Runs a single evolutionary game simulation trial until convergence or maxstep."""
    
    # --- 1. Initialization ---
    pC = config.ini_pC
    S_i = states(pC, config.n)
    L_i = config.initial_levels
    
    # Calculate initial distribution and hierarchy
    distr = L_distribution(L_i, S_i, config.n)
    H = hierarchicalness(distr, config.n)
    
    overMaxG = 0
    step = 1

    # --- 2. Core Simulation Loop ---
    while step <= config.maxstep:
        # Promotion Phase
        L_i = level_next(L_i, S_i, distr, config.n, G, H, config.alpha)
        
        # Update distributions based on new levels
        distr = L_distribution(L_i, S_i, config.n)
        H = hierarchicalness(distr, config.n)
        
        # Allocation Phase
        income_i, mark, W_C, W_D = income_CD(distr, G, config.n, config.b, c, H)
        
        if mark == 1:
            overMaxG += 1

        # Check for Convergence (Only Cooperators or Only Defectors remain)
        if len(set(S_i)) <= 1:
            break
            
        # Preparation Phase for the next step
        pC = pC_Next(pC, W_C, W_D)
        S_i = states(pC, config.n)
        
        step += 1
        
    # --- 3. Return Trial Results ---
    return {
        'final_L_i': L_i,
        'final_S_i': S_i,
        'steps': step,
        'overMaxG_ratio': overMaxG / step,
        'entropy': entropy(L_i, config.initial_levels)
    }


def run_simulation_set(c: float, G: float, config: SimulationConfig) -> dict:
    """Runs `test_num` trials for a specific (c, G) pair and aggregates the statistics."""
    
    # 用於儲存 test_num 次模擬的原始數據
    L_i_all = []
    CD_ratio_all = []
    num_levels_all = []
    steps_all = []
    overMaxG_all = []
    entropy_all = []

    # 執行所有模擬
    for _ in range(config.test_num):
        res = run_single_trial(c, G, config)
        
        S_i = res['final_S_i']
        L_i = res['final_L_i']
        c_count = S_i.count('C')
        cd_ratio = round(c_count / len(S_i), 4)
        num_levels = len(set(L_i))
        
        # 紀錄統計數據
        L_i_all.append(L_i)
        CD_ratio_all.append(cd_ratio)
        num_levels_all.append(num_levels)
        steps_all.append(res['steps'])
        overMaxG_all.append(res['overMaxG_ratio'])
        entropy_all.append(res['entropy'])

    # 彙整並回傳最終字典格式
    aggregated_data = {
        # 原始數據陣列
        'final_L_i': L_i_all,
        'CD_ratio': CD_ratio_all,
        'num_levels': num_levels_all,
        'steps': steps_all,
        'times_overMaxG': overMaxG_all,
        'entropy': entropy_all,
        
        # 計算平均值
        'ave_step': np.mean(steps_all),
        'ave_CD_ratio': round(np.mean(CD_ratio_all), 3),
        'ave_num_levels': np.mean(num_levels_all),
        'ave_overMaxG': np.mean(overMaxG_all),
        'ave_entropy': np.mean(entropy_all),
    }

    return aggregated_data



if __name__ == '__main__':
    # Initialize Configuration
    config = SimulationConfig()
    
    # Generate search space (201 steps from 0.0 to 1.0)
    c_set = list(np.linspace(0, 1, 201))   
    G_set = list(np.linspace(0, 1, 201))   

    # Initialize the main dictionary intended for the pickle file
    data = {
        'title': [config.n, config.alpha, config.ini_pC, config.b, c_set, G_set, config.test_num]
    }
    
    start_time_all = time.time()
    
    for c in c_set:
        for G in G_set:
            start_time = time.time()
            
            # Generate key name
            c_b_ratio = round(c / config.b, 11)
            name = f"n={config.n},alpha={config.alpha},ini_pC={config.ini_pC},c/b={c_b_ratio},G={G},testnum={config.test_num}"
            print(name)
            
            # Execute simulation block
            data[name] = run_simulation_set(c, G, config)
            
            # Print console metrics
            res = data[name]
            print(f"ave_step: {res['ave_step']:.2f} | ave_CD: {res['ave_CD_ratio']} | "
                  f"ave_num_levels: {res['ave_num_levels']:.2f} | OverMaxG: {res['ave_overMaxG']:.4f}")
            
            # Time Tracking
            end_time = time.time()
            total_seconds = end_time - start_time_all 
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            
            print(f"Execution time: {end_time - start_time:.6f} seconds | Cumulative execution time: {hours} hours {minutes} minutes {seconds:.6f} seconds\n")

    # Save to disk
    os.makedirs('artifacts', exist_ok=True)
    filename = f'artifacts/alpha_{config.alpha}_inipC_{config.ini_pC}_201x201_CD_n_{config.n}_testnum={config.test_num}.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(data, file)