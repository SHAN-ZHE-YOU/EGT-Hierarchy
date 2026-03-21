# Multi-Level Leadership and Cooperation in Evolutionary Games

This repository contains the simulation source code and data analysis scripts for the study on multi-level leadership and hierarchical cooperation. 

## Overview
This project develops an evolutionary game theory model that extends traditional hierarchical cooperation frameworks to incorporate multi-level leadership structures. It simulates groups of $N$ individuals who can adopt either cooperative (C) or defective (D) strategies while occupying different hierarchical levels within an evolving organizational structure. 

## Core Model Mechanisms
The simulation operates through iterative rounds, each consisting of four distinct phases:

1. **Hierarchical Promotion:** Cooperators have the opportunity to advance to the next hierarchical level. Defectors cannot be promoted. The promotion probability is governed by inequality (Gini coefficient) and structural stability (Hierarchical score):

$$P_{promotion} = \alpha \cdot g + (1-\alpha) \cdot h$$

   Where $g$ is the inequality-driven promotion probability, $h$ is the stability-driven probability, and $\alpha$ balances their influence.

2. **Contribution to Common Pool:** Cooperators contribute a benefit $b$ to the common pool with a probability equal to the hierarchical score $H$. Defectors contribute nothing and retain a private benefit $c$.

3. **Payoff Distribution:** Total contributions are distributed among all players based on their hierarchical levels and the defined Gini coefficient ($G$).

4. **Strategy Update:** Strategy evolution follows replicator dynamics, updating the proportion of cooperators based on the average payoffs of cooperators versus defectors.

## Key Findings
The simulation explores the parameter space of the Gini coefficient ($G$) and the cost-to-benefit ratio ($c/b$), yielding several key insights:

* **Cooperation Stability:** Moderate inequality, by creating promotion incentives, helps maintain cooperation, whereas overly egalitarian environments are highly vulnerable to defection temptations.
* **Hierarchical Complexity:** Complex multi-level structures are most likely to emerge when both inequality and defection costs are low. The number of hierarchical levels exhibits a logarithmic growth relationship with group size.
* **Convergence Dynamics:** The system experiences critical slowing down (requiring significantly more iterations to reach a steady state) near the boundary between cooperation and defection.
* **System Predictability:** Based on Shannon entropy calculations, the highest uncertainty and unpredictability occur in regions with low inequality and mid-to-low defection costs, where evolutionary pressures are most balanced.

## Quick Start

**1. Install Dependencies**
Ensure you have Python 3 installed, then install the required scientific libraries:
```bash
pip install numpy scipy matplotlib
```

**2. Run the Simulation**
Execute the main script to start the evolutionary simulation. The output will be saved as a `.pickle` file in the `artifacts/` directory.
```bash
python hierarchy_main.py
```

**3. Generate Plots**
Use the plotting script to generate heatmaps for average cooperation, hierarchy levels, steps, and entropy. Replace `<filename>` with your generated data file.
```bash
python hierarchy_plot.py -f artifacts/<filename>.pickle -t all
```

## Citation
If you use this code or model in your research, please cite our paper:
*(citation)
