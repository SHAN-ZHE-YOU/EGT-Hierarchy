#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
Description: Utility functions for an evolutionary game theory simulation 
incorporating hierarchical structures and Gini coefficient-based distributions.
"""

import random
import statistics
import numpy as np
from scipy.stats import entropy


def states(pC, n):
    """
    Assigns initial states ('C' for Cooperator, 'D' for Defector) to individuals.
    
    Args:
        pC (float): Proportion of cooperators in the population.
        n (int): Total number of individuals.
        
    Returns:
        list: A randomly shuffled list of 'C' and 'D' states.
    """
    states = []
    num_C = round(n * pC)  # Calculate the integer number of cooperators
    
    # Assign 'C' to the first num_C individuals, and 'D' to the rest
    for i in range(1, n + 1):
        if i <= num_C:
            states.append('C')
        else:
            states.append('D')
    
    # Shuffle the list to ensure a random distribution of states
    random.shuffle(states)
    
    return states

def ini_levels(n):
    """
    Initializes the starting levels for all individuals (default level is 1).
    """
    Level = []
    for i in range(n):
        Level.append(1)
    return Level

def L_distribution(L_i, S_i, n):
    """
    Calculates the distribution of levels and states within the population.
    
    Returns:
        list: Distribution matrix where each element is 
              [level_value, total_individuals, C_count, D_count].
    """
    Level_num = []  
    
    sort_L = sorted(L_i)
    sim_L = sorted(list(set(L_i))) 
    
    # Initialize the distribution list for each unique level
    for k in range(len(sim_L)):
        Level_num.append([sim_L[k], 0, 0, 0]) 
        
    k = 0
    for i in range(len(L_i)):
        # Count total individuals per level
        if sim_L[k] == sort_L[i]:
            Level_num[k][1] += 1 
        else: 
            k += 1 
            Level_num[k][1] += 1 

        # Count 'C' and 'D' states per level
        if S_i[i] == 'C':  
            Level_num[k][2] += 1
        elif S_i[i] == 'D':  
            Level_num[k][3] += 1
        else:
            raise ValueError('State must be "C" or "D"')
        
    return Level_num

def hierarchicalness(distr, n):
    """
    Calculates the hierarchicalness (Global Reaching Centrality / GRC) of the network.
    """
    x = distr[-1][1]  # Number of nodes on the highest level
    C_R_max = (n - x) / (n - 1)  # Maximum possible relative level index

    howmanylower = 0  # Cumulative count of nodes at lower levels
    GCR_terms = []
    k = 0             # Current level index
    num = distr[k][1] # Total nodes in the current level
    
    for i in range(n):
        # Move to the next level when the current node index exceeds the current level's bounds
        if i + 1 > num:  
            howmanylower += distr[k][1]
            k += 1  
            num += distr[k][1]  
        
        C_R_i = howmanylower / (n - 1)  # Relative level index for the current node
        GCR_terms.append(C_R_max - C_R_i)  # Contribution to overall hierarchicalness

    H = round(sum(GCR_terms) / (n - 1), 3)

    return H

def nn_maxG(distr, n):
    """
    Calculates the maximum possible Gini Coefficient ($G$) with the given 
    level distribution, assuming no negative incomes.
    """
    if len(distr) == 1:
        maxG = 0
    else:
        f_last = distr[-1][1]
        maxG = round(1 - f_last / n, 3)
        
    return maxG

def income_pool(distr, b, H):
    """
    Calculates the total income pool. Each 'C' has a probability $H$ 
    to contribute benefit $b$ into the common pool.
    """
    pool = 0
    for i in range(len(distr)):
        for j in range(distr[i][2]):
            r = random.random()
            if r <= H:
                pool += b
    return pool

def twolevel_div(distr, G, n, pool):
    """
    Divides the income pool for exactly two levels, ensuring no negative incomes.
    """
    f1 = distr[0][1]
    f2 = distr[1][1]
    h1 = 1 - G - f2 / n
    
    div_i = []
    for i in range(n):
        if i + 1 <= f1:
            div_i.append(round(pool * h1 / f1, 4))
        else:
            div_i.append(round(pool * (1 - h1) / f2, 4))
    return div_i

def G_LCarea(X, Y, pool): 
    """
    Calculates the area under the Lorenz Curve (area 'B') to check the Gini coefficient.
    Used when the number of levels >= 3.
    """
    subarea = []
    for i in range(len(Y) - 1):
        subarea.append((Y[i] / pool) * (X[i+1] - X[i]))
        subarea.append((Y[i] / pool) * (X[i+2] - X[i+1]))
    subarea.append((Y[-1] / pool) * (X[-1] - X[-2]))
    
    return 1 - sum(subarea)

def find_para(upper_a, lower_a, xdata, distr, G, pool, n):
    """
    Numerical division into 3 or more levels using a parabola fit approach.
    Iteratively finds the exponent 'a' that satisfies the target Gini coefficient.
    """
    if pool == 0:
        guess_a = -1
        guess_G = -1
        div_income = []
        for i in range(n):
            div_income.append(0) 
    else:
        err = 1
        ii = 1
        tolerance = 0.0001
        
        # Track the best solution (minimum error)
        best_err = float('inf')
        best_a = None
        best_G = None
        best_div_income = None
        
        while ii <= 2000:
            # Dynamically adjust tolerance
            if ii > 1000:
                tolerance = 0.001
            
            # Check for convergence
            if abs(err) <= tolerance:
                break
                
            guess_a = (upper_a + lower_a) / 2
            div_income = []
            cumu_income = []
            
            for i in range(len(distr)):
                cumu_income.append(round((xdata[i+1]**guess_a) * pool, 5))
                for j in range(distr[i][1]):
                    div_income.append(round((xdata[i+1]**guess_a - xdata[i]**guess_a) * pool / distr[i][1], 5))
                    
            guess_G = G_LCarea(xdata, cumu_income, pool) 
            err = G - guess_G
            
            # Update best solution
            if abs(err) < best_err:
                best_err = abs(err)
                best_a = guess_a
                best_G = guess_G
                best_div_income = div_income.copy()
            
            # Adjust search boundaries
            if err > 0:
                lower_a = guess_a
            elif err < 0:
                upper_a = guess_a
            
            if ii > 1000:
                print('-----', ii, '-----', G, '-----', f'tolerance: {tolerance}, best_err: {best_err:.6f}')
            
            ii = ii + 1
        
        # Fallback to the best solution if max iterations reached without strict convergence
        if ii > 2000:
            print(f'WARNING: Parameter combination G={G} did not converge after 2000 iterations. Using the best solution (error={best_err:.6f})')
            guess_a = best_a
            guess_G = best_G
            div_income = best_div_income

    return guess_a, guess_G, div_income

def parabola_div(distr, G, n, pool):  
    """
    Divides the pool into 3 or more levels by fitting a power-law curve ($y = x^a$).
    """
    xdata = [0]
    accumulate_x = 0
    
    for i in range(len(distr)):
        xdata.append(round((distr[i][1] + accumulate_x) / n, 3))
        accumulate_x = accumulate_x + distr[i][1]
    
    upper_a = 5000
    lower_a = 1
    
    findpara = find_para(upper_a, lower_a, xdata, distr, G, pool, n)
    div_income = findpara[2]
    
    return div_income

def div_pool(distr, G, n, b, c, H):
    """
    Allocates individual income based on the targeted Gini coefficient.
    
    Returns:
        tuple: (divpool_i, mark) where mark indicates adjustments made to G.
    """
    pool = income_pool(distr, b, H)
    MaxG = nn_maxG(distr, n)
    
    divpool_i = []
    
    if len(distr) == 1: 
        # Only 1 level exists -> evenly divide the pool
        for i in range(n):
            divpool_i.append(round(pool / n, 4))
        mark = 2
    
    elif G == 0: 
        # G=0 -> evenly divide the pool
        for i in range(n):
            divpool_i.append(round(pool / n, 4))
        mark = 2
        
    else:
        if G > MaxG:
            G_input = MaxG * 1
            mark = 1  # 1 indicates G was capped at MaxG
        else:
            G_input = G
            mark = 0  # 0 indicates original G was used
        
        if len(distr) == 2:
            divpool_i = twolevel_div(distr, G_input, n, pool)
        else:
            divpool_i = parabola_div(distr, G_input, n, pool)

    return divpool_i, mark

def income_CD(distr, G, n, b, c, H):
    """
    Calculates final incomes considering 'C' contributions and 'D' retentions.
    Defectors ('D') keep magnitude 'c' for themselves.
    """
    div = div_pool(distr, G, n, b, c, H)
    divpool_i = div[0]
    mark = div[1]
    
    income_i = []
    num = 0
    sumC = 0
    sumD = 0
    nC = 0
    nD = 0
    
    for i in range(len(distr)):
        for j in range(distr[i][2]):  # Process 'C' individuals
            income_i.append(divpool_i[num])
            sumC = sumC + divpool_i[num]
            nC += 1
            num += 1
            
        for k in range(distr[i][3]):  # Process 'D' individuals
            income_i.append(divpool_i[num] + c)
            sumD = sumD + (divpool_i[num] + c)
            nD += 1
            num += 1
    
    W_C = round(sumC / nC, 4) if nC > 0 else 0
    W_D = round(sumD / nD, 4) if nD > 0 else 0
    
    return income_i, mark, W_C, W_D

def pC_Next(pC, W_C, W_D): 
    """
    Calculates the proportion of Cooperators for the next iteration using replicator dynamics.
    Equation: $W(C) / (W(C) + W(D))$
    """
    pD = 1 - pC
    if W_C == 0 and W_D == 0:
        return 0
    else:
        return round((pC * W_C) / ((pC * W_C) + (pD * W_D)), 4)

def gh(n, G, H, L_i):
    """
    Calculates promotion probabilities based on hierarchy and inequality.
    
    Args:
        h (list): Probability of promotion related to hierarchicalness.
        g (list): Probability of promotion related to Gini coefficient.
    """
    h_i = []                                
    g_i = []                             
    pPR = []                                
    
    # Calculate performance relative to the population (pseudo-PR value)
    sigma = statistics.stdev(L_i)  # Standard deviation
    ave_L = sum(L_i) / len(L_i)    # Mean
    range_PR = 3                   # Setting boundary bounds
    
    for i in range(n):
        if sigma == 0:
            psuedo_PR = 0
        else:
            psuedo_PR = (L_i[i] - ave_L) / sigma

        # Clamp pseudo_PR to bounds [-3, 3]
        if psuedo_PR >= range_PR: 
            psuedo_PR = range_PR
        elif psuedo_PR <= -range_PR: 
            psuedo_PR = -range_PR
            
        pPR.append(psuedo_PR)
            
        # Calculate h value (Linear mapping: Ax+By+C = 0)
        h_x0 = 0.5 - (psuedo_PR / range_PR) * 0.5
        h_value = h_x0 * (1 - H * 0.9)  # Give some chances even when H=1
        h_i.append(h_value)
        
        # Calculate g value (Linear mapping: Ax+By+C = 0)
        g_x1 = 0.5 - (psuedo_PR / range_PR) * 0.5
        g_value = g_x1 * G
        g_i.append(g_value)
        
    return h_i, g_i

def level_next(L_i, S_i, distr, n, G, H, alpha):
    """
    Determines the next level for all individuals based on their state and promotion odds.
    """
    L_next = []
    
    # Fetch promotion probabilities
    g_and_h = gh(n, G, H, L_i) 
    h = g_and_h[0]  # Hierarchy-related promotion
    g = g_and_h[1]  # Gini-related promotion
    
    for i in range(n):
        # Determine base probability of promotion
        if len(distr) == 1: 
            # All nodes are on the same level
            pp = 1 / n 
        else:
            # Combination of g and h ($0 \le pp \le 1$)
            pp = alpha * g[i] + (1 - alpha) * h[i] 
    
        # Roll the die to decide promotion
        if S_i[i] == 'C':  
            r = random.random()
            if r < pp:
                L_next.append(L_i[i] + 1)  # C gets promoted
            else:
                L_next.append(L_i[i])      # C doesn't get promoted
        else:  
            # D never gets promoted
            L_next.append(L_i[i]) 

    return L_next