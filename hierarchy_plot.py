#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weiyujie
Description: Script to generate heatmaps from evolutionary game theory simulation data.
"""

import argparse
import pickle
import sys
import re
from hierarchy_plot_package import *


def parse_arguments():
    """
    Parses command-line arguments for the plotting script.
    """
    parser = argparse.ArgumentParser(description="Generate heatmaps from hierarchy simulation data.")
    
    # Required argument for the input file
    parser.add_argument('-f', '--file', type=str, required=True, 
                        help="Path to the pickle data file (e.g., artifacts/alpha_0.5...pickle)")
    
    # Optional argument for the plot type
    parser.add_argument('-t', '--type', type=str, default='all',
                        choices=['all', 'aveCD', 'aveStep', 'aveLevels', 'aveOverMaxG', 'aveEntropy'],
                        help="Type of plot to generate. Default is 'all'.")
    
    # Optional argument for the output format
    parser.add_argument('-o', '--format', type=str, default='png',
                        choices=['png', 'pdf', 'eps'],
                        help="Output format for the plots (png, pdf, eps). Default is 'png'.")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    file_name = args.file
    plot_type = args.type
    output_format = args.format

    # Load the pickle data
    try:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        sys.exit(1)

    # Extract parameters from the loaded data
    n = data['title'][0]
    alpha = data['title'][1]
    ini_pC = data['title'][2]
    b = data['title'][3]
    c_set = data['title'][4]
    G_set = data['title'][5]
    test_num = data['title'][6]

    print(f"Loaded data from: {file_name}")
    print(f"Parameters: n = {n}, alpha = {alpha}, test_num = {test_num}, format = {output_format}")

    # Map the plot types to their corresponding functions
    plot_functions = {
        'aveCD': Plot_aveCD_heatmap,
        'aveStep': Plot_aveStep_heatmap,
        'aveLevels': Plot_aveLevels_heatmap,
        'aveEntropy': Plot_aveEntropy_heatmap
    }

    # Determine which plots to generate based on user input
    if plot_type == 'all':
        targets = list(plot_functions.keys())
    else:
        targets = [plot_type]

    # Execute the plotting functions
    for target in targets:
        print(f"Plotting {target}...")
        func = plot_functions[target]
        
        # Call the mapped function, passing the output_format at the end
        func(n, alpha, ini_pC, b, c_set, G_set, test_num, data, output_format)
        
    print("Plotting completed.")


if __name__ == '__main__':
    main()