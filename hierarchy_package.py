#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ginny Wei
Description: Utility functions for the evolutionary game theory simulation.
Refactored for readability, standard Python conventions, and full English documentation.
"""

import random
import statistics



def states(pC: float, n: int) -> list:
    """
    Generates the initial states ('C' for Cooperator, 'D' for Defector) of individuals.
    
    Args:
        pC: Proportion of cooperators.
        n: Total number of individuals.
    Returns:
        A shuffled list of 'C' and 'D' states.
    """
    num_C = round(n * pC)
    
    # Create the list with exact numbers of C and D, then shuffle
    player_states = ['C'] * num_C + ['D'] * (n - num_C)
    random.shuffle(player_states)
    
    return player_states

def ini_levels(n: int) -> list:
    """
    Initializes the hierarchy levels for all individuals to level 1.
    """
    return [1] * n


def L_distribution(L_i: list, S_i: list, n: int) -> list:
    """
    Calculates the distribution of individuals across hierarchy levels.
    
    Returns:
        A list of lists in the format: [[level, total_count, C_count, D_count], ...]
        Sorted from the lowest level to the highest level.
    """
    distribution_map = {}
    
    # Group individuals by their current level
    for level, state in zip(L_i, S_i):
        if level not in distribution_map:
            distribution_map[level] = {'total': 0, 'C': 0, 'D': 0}
            
        distribution_map[level]['total'] += 1
        
        if state in ['C', 'D']:
            distribution_map[level][state] += 1
        else:
            raise ValueError('State must be strictly "C" or "D"')
            
    # Format into the required list of lists, sorted by level
    level_distribution = []
    for level in sorted(distribution_map.keys()):
        counts = distribution_map[level]
        level_distribution.append([level, counts['total'], counts['C'], counts['D']])
        
    return level_distribution


def hierarchicalness(distr: list, n: int) -> float:
    """
    Calculates the hierarchicalness (GCR - Gini-like Coefficient of rank).
    """
    highest_level_count = distr[-1][1]
    max_relative_rank = (n - highest_level_count) / (n - 1)

    nodes_below = 0
    gcr_sum = 0.0
    
    # Calculate GCR by iterating through levels instead of every individual (O(L) instead of O(n))
    for level_data in distr:
        level_count = level_data[1]
        
        # All individuals at this level share the same relative rank calculation
        relative_rank_i = nodes_below / (n - 1)
        contribution = max_relative_rank - relative_rank_i
        
        gcr_sum += contribution * level_count
        nodes_below += level_count

    return round(gcr_sum / (n - 1), 3)


def nn_maxG(distr: list, n: int) -> float:
    """
    Calculates the maximum possible Gini Coefficient given the level distribution,
    assuming strictly non-negative incomes.
    """
    if len(distr) == 1:
        return 0.0
    
    highest_level_count = distr[-1][1]
    return round(1 - (highest_level_count / n), 3)


def income_pool(distr: list, b: float, H: float) -> float:
    """
    Calculates the total income pool. 
    Each Cooperator ('C') has a probability 'H' to contribute benefit 'b' to the pool.
    """
    pool = 0.0
    for level_data in distr:
        num_cooperators = level_data[2]
        for _ in range(num_cooperators):
            if random.random() <= H:
                pool += b
    return pool


def twolevel_div(distr: list, G: float, n: int, pool: float) -> list:
    """
    Divides the income pool into exactly TWO levels based on the Gini coefficient.
    Ensures no negative incomes.
    """
    count_level_1 = distr[0][1]
    count_level_2 = distr[1][1]
    
    # Income proportion for the lower level based on Gini mathematics
    h1 = 1 - G - (count_level_2 / n)
    
    divided_income = []
    for i in range(n):
        if i < count_level_1:
            divided_income.append(round(pool * h1 / count_level_1, 4))
        else:
            divided_income.append(round(pool * (1 - h1) / count_level_2, 4))
            
    return divided_income


def G_LCarea(X: list, Y: list, pool: float) -> float:
    """
    Calculates the Gini coefficient using the Lorenz Curve area check.
    Uses numerical trapezoidal/rectangular approximation.
    """
    subarea = []
    for i in range(len(Y) - 1):
        subarea.append((Y[i] / pool) * (X[i+1] - X[i]))
        subarea.append((Y[i] / pool) * (X[i+2] - X[i+1]))
        
    subarea.append((Y[-1] / pool) * (X[-1] - X[-2]))
    return 1 - sum(subarea)


def find_para(upper_a: float, lower_a: float, xdata: list, distr: list, G: float, pool: float, n: int):
    """
    Uses binary search to find the parabola exponent 'a' that fits the target Gini coefficient.
    """
    if pool == 0:
        return -1, -1, [0.0] * n

    iteration = 1
    tolerance = 0.0001
    
    best_error = float('inf')
    best_a = None
    best_G = None
    best_div_income = None
    
    while iteration <= 2000:
        # Relax tolerance after 1000 iterations to force convergence
        if iteration > 1000:
            tolerance = 0.001
            
        guess_a = (upper_a + lower_a) / 2
        div_income = []
        cumu_income = []
        
        # Calculate income distributions with the guessed exponent
        for i, level_data in enumerate(distr):
            level_count = level_data[1]
            cumu_val = round((xdata[i+1] ** guess_a) * pool, 5)
            cumu_income.append(cumu_val)
            
            level_income_share = ((xdata[i+1] ** guess_a) - (xdata[i] ** guess_a)) * pool
            individual_income = round(level_income_share / level_count, 5)
            div_income.extend([individual_income] * level_count)
            
        guess_G = G_LCarea(xdata, cumu_income, pool)
        error = G - guess_G
        
        # Track the best approximation found so far
        if abs(error) < best_error:
            best_error = abs(error)
            best_a = guess_a
            best_G = guess_G
            best_div_income = div_income.copy()
            
        if abs(error) <= tolerance:
            break
            
        # Adjust binary search bounds
        if error > 0:
            lower_a = guess_a
        elif error < 0:
            upper_a = guess_a
            
        iteration += 1
        
    # Failsafe if mathematical convergence wasn't reached
    if iteration > 2000:
        print(f'WARNING: Parameters (G={G}) did not converge after 2000 iterations. Using best result (error={best_error:.6f})')
        return best_a, best_G, best_div_income

    return guess_a, guess_G, div_income


def parabola_div(distr: list, G: float, n: int, pool: float) -> list:
    """
    Divides the pool into 3 or more levels using a parabolic fit approach (y = x^a).
    """
    xdata = [0]
    accumulated_x = 0
    
    for level_data in distr:
        accumulated_x += level_data[1]
        xdata.append(round(accumulated_x / n, 3))
    
    upper_bound = 5000
    lower_bound = 1
    sol_a, sol_G, div_income = find_para(upper_bound, lower_bound, xdata, distr, G, pool, n)
    
    return div_income


def div_pool(distr: list, G: float, n: int, b: float, c: float, H: float):
    """
    Determines how the pool is divided among individuals based on the Gini Coefficient.
    Returns the divided incomes and a 'mark' (status flag for Gini ceiling).
    """
    pool = income_pool(distr, b, H)
    max_G = nn_maxG(distr, n)
    
    divided_pool = []
    
    if len(distr) == 1 or G == 0:
        # If only 1 level exists, or Gini is 0, distribute perfectly evenly
        divided_pool = [round(pool / n, 4)] * n
        mark = 2  # Status 2: Even distribution triggered
        
    else:
        if G > max_G:
            G_input = max_G
            mark = 1  # Status 1: Requested G exceeds maximum possible, capped at max_G
        else:
            G_input = G
            mark = 0  # Status 0: Normal operation
        
        if len(distr) == 2:
            divided_pool = twolevel_div(distr, G_input, n, pool)
        else:
            divided_pool = parabola_div(distr, G_input, n, pool)

    return divided_pool, mark


def income_CD(distr: list, G: float, n: int, b: float, c: float, H: float):
    """
    Calculates final individual incomes, applying the benefit 'b' and defector retention 'c'.
    """
    divided_pool, mark = div_pool(distr, G, n, b, c, H)
    
    income_i = []
    sum_C = 0.0
    sum_D = 0.0
    count_C = 0
    count_D = 0
    
    index = 0
    for level_data in distr:
        num_C_in_level = level_data[2]
        num_D_in_level = level_data[3]
        
        # Allocate income to Cooperators
        for _ in range(num_C_in_level):
            base_income = divided_pool[index]
            income_i.append(base_income)
            sum_C += base_income
            count_C += 1
            index += 1
            
        # Allocate income to Defectors (they keep 'c' for themselves)
        for _ in range(num_D_in_level):
            base_income = divided_pool[index] + c
            income_i.append(base_income)
            sum_D += base_income
            count_D += 1
            index += 1
            
    # Calculate average wealth for Cooperators and Defectors
    W_C = round(sum_C / count_C, 4) if count_C > 0 else 0.0
    W_D = round(sum_D / count_D, 4) if count_D > 0 else 0.0
    
    return income_i, mark, W_C, W_D


def pC_Next(pC: float, W_C: float, W_D: float) -> float:
    """
    Calculates the proportion of Cooperators for the next iteration step.
    Formula: (pC * W_C) / ((pC * W_C) + (pD * W_D))
    """
    pD = 1 - pC
    if W_C == 0 and W_D == 0:
        return 0.0
    return round((pC * W_C) / ((pC * W_C) + (pD * W_D)), 4)


def gh(n: int, G: float, H: float, L_i: list):
    """
    Calculates the promotion probabilities for individuals.
    'h': Probability related to hierarchicalness
    'g': Probability related to the Gini coefficient
    """
    h_probabilities = []
    g_probabilities = []
    
    # Calculate PR values to gauge relative standing in the hierarchy
    sigma = statistics.stdev(L_i) if len(L_i) > 1 else 0
    ave_L = sum(L_i) / len(L_i)
    range_PR = 3
    
    for i in range(n):
        if sigma == 0:
            pseudo_PR = 0.0
        else:
            pseudo_PR = (L_i[i] - ave_L) / sigma

        # Clamp the pseudo_PR value between -3 and 3
        pseudo_PR = max(min(pseudo_PR, range_PR), -range_PR)
            
        # Calculate 'h' (Probability related to hierarchy)
        h_x0 = 0.5 - (pseudo_PR / range_PR) * 0.5
        h_value = h_x0 * (1 - H * 0.9)  # Prevents h from hitting zero when H=1
        h_probabilities.append(h_value)
        
        # Calculate 'g' (Probability related to Gini)
        g_x1 = 0.5 - (pseudo_PR / range_PR) * 0.5
        g_value = g_x1 * G
        g_probabilities.append(g_value)
        
    return h_probabilities, g_probabilities


def level_next(L_i: list, S_i: list, distr: list, n: int, G: float, H: float, alpha: float) -> list:
    """
    Determines the hierarchy levels of individuals for the next step.
    Cooperants have a chance to be promoted, Defectors are never promoted.
    """
    L_next = []
    
    h_probs, g_probs = gh(n, G, H, L_i)
    
    for i in range(n):
        # Generate the probability of promotion
        if len(distr) == 1:
            promotion_prob = 1 / n
        else:
            # Alpha controls the weight between Gini-driven and Hierarchy-driven promotion
            promotion_prob = (alpha * g_probs[i]) + ((1 - alpha) * h_probs[i])
    
        # Roll the dice for promotion
        if S_i[i] == 'C': 
            if random.random() < promotion_prob:
                L_next.append(L_i[i] + 1)  # Promoted
            else:
                L_next.append(L_i[i])      # Not promoted
        else:  
            # Defectors ('D') do not get promoted
            L_next.append(L_i[i])

    return L_next