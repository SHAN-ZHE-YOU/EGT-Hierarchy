#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
Description: Evolutionary game theory simulation with hierarchy and Gini coefficient.
"""

import os
import time
import pickle
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.stats import entropy
from tqdm import tqdm

from hierarchy_package import (
    states, ini_levels, L_distribution, hierarchicalness, 
    level_next, income_CD, pC_Next
)

@dataclass
class SimConfig:
    """Simulation configuration parameters"""
    alpha: float = 0.5      # (g/h) ratio: Gini-driven vs hierarchical-stable promotion
    n: int = 10             # Group size (total number of players)
    test_num: int = 10     # Number of repeating times per configuration
    b: int = 1              # Benefit magnitude contributed by a cooperator
    ini_pC: float = 0.9     # Initial proportion of cooperators in the population
    maxstep: int = 500      # Maximum iterations per simulation
    
    # Keeping a benefit of magnitude c for defector [0.0, ..., 1.0]
    c_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 201)))
    # Value of Gini Coefficient [0.0, ..., 1.0] representing inequality
    G_set: List[float] = field(default_factory=lambda: list(np.linspace(0, 1, 201)))

# Configure logging to write to a file, keeping the console clean for the progress bar
logging.basicConfig(
    filename='simulation.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def run_simulation(config: SimConfig):
    data = {}
    data['title'] = [config.n, config.alpha, config.ini_pC, config.b, config.c_set, config.G_set, config.test_num]
    
    logging.info(f"Starting simulation: n={config.n}, test_num={config.test_num}")
    
    start_time_all = time.time()
    total_iterations = len(config.c_set) * len(config.G_set)
    
    # Progress bar wrapping the nested loops
    with tqdm(total=total_iterations, desc="Simulating") as pbar:
        for c in config.c_set:
            for G in config.G_set:
                start_time = time.time()
                
                # c/b ratio rounded to 11 decimal places
                name = (f"n={config.n},alpha={config.alpha},ini_pC={config.ini_pC},"
                        f"c/b={round(c/config.b, 11)},G={G},testnum={config.test_num}")
                
                data[name] = {}
                logging.info(f"--- Running Configuration: {name} ---")
                
                # --- Initialize Data Storage ---
                # General records
                data[name]['final_L_i'] = []      
                data[name]['CD_ratio'] = []       
                data[name]['num_levels'] = []     
                data[name]['times_overMaxG'] = [] 
                data[name]['steps'] = []          
                data[name]['entropy'] = []                    

                # --- Run Repeating Tests ---
                for i in range(config.test_num):
                    
                    # Initialization
                    step = 1
                    overMaxG = 0 
                    pC = config.ini_pC
                    
                    # Distribution setup
                    S_i = states(config.ini_pC, config.n)
                    L_i = ini_levels(config.n)
                    distr = L_distribution(L_i, S_i, config.n)
                    H = hierarchicalness(distr, config.n)
                    
                    # Promotion (Step 1)
                    L_i = level_next(L_i, S_i, distr, config.n, G, H, config.alpha)
                    distr = L_distribution(L_i, S_i, config.n)
                    H = hierarchicalness(distr, config.n)
                    
                    # Allocation (Step 1)
                    income_i, mark, W_C, W_D = income_CD(distr, G, config.n, config.b, c, H)
                    if mark == 1:
                        overMaxG += 1
             
                    # Iteration (From Step 2 to maxstep)
                    while len(set(S_i)) > 1 and step <= config.maxstep - 1:
                        # Update proportion
                        pC = pC_Next(pC, W_C, W_D)
                        
                        # Promotion
                        S_i = states(pC, config.n)
                        L_i = level_next(L_i, S_i, distr, config.n, G, H, config.alpha)
                        distr = L_distribution(L_i, S_i, config.n)
                        H = hierarchicalness(distr, config.n)  
                        
                        # Allocation         
                        income_i, mark, W_C, W_D = income_CD(distr, G, config.n, config.b, c, H)
                        if mark == 1:
                            overMaxG += 1
                
                        step += 1 
                
                    # --- Record Results ---
                    data[name]['final_L_i'].append(L_i)
                    data[name]['CD_ratio'].append(round(S_i.count('C') / len(S_i), 4))
                    data[name]['num_levels'].append(len(list(set(L_i))))
                    data[name]['steps'].append(step)
                    data[name]['times_overMaxG'].append(overMaxG / step)
                    data[name]['entropy'].append(entropy(L_i, ini_levels(config.n)))

                # --- Calculate and Log Averages ---
                data[name]['ave_overMaxG'] = np.mean(data[name]['times_overMaxG'])  
                data[name]['ave_CD_ratio'] = round(np.mean(data[name]['CD_ratio']), 3)  
                data[name]['ave_num_levels'] = np.mean(data[name]['num_levels'])  
                data[name]['ave_step'] = np.mean(data[name]['steps'])  
                data[name]['ave_entropy'] = np.mean(data[name]['entropy'])  

                logging.info(
                    f"Results -> ave_step: {data[name]['ave_step']}, "
                    f"ave_CD: {data[name]['ave_CD_ratio']}, "
                    f"ave_num_levels: {data[name]['ave_num_levels']}, "
                    f"OverMaxG: {data[name]['ave_overMaxG']}"
                )
                
                # Logging execution time
                end_time = time.time()
                total_seconds = end_time - start_time_all 
                hours, rem = divmod(total_seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                
                logging.info(
                    f"Execution time: {end_time - start_time:.6f} s | "
                    f"Accumulated time: {int(hours)} h {int(minutes)} m {seconds:.6f} s\n"
                )
                
                # Update progress bar
                pbar.update(1)
                
    # --- Save Data ---
    os.makedirs('artifacts', exist_ok=True)
    filename = f'artifacts/alpha_{config.alpha}_inipC_{config.ini_pC}_201x201_CD_n_{config.n}_testnum={config.test_num}.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    
    logging.info(f"Simulation completed. Data saved to {filename}")

if __name__ == '__main__':
    config = SimConfig()
    run_simulation(config)
    print("Simulation completed. Check 'simulation.log' for details and 'artifacts/' for results.")