# game-theoretic-free-energy-principle
Simulation code and data for the Game-Theoretic Free Energy Principle: neural ensembles, schooling fish, and multi-agent RL. Reproduces all figures from the Nature submission (2026).
This repository contains the complete simulation code and generated data for the paper: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19629592.svg)](https://doi.org/10.5281/zenodo.19629592)

> **"A Variational Principle Unifying Bayesian Inference, Thermodynamics, and Game‑Theoretic Equilibrium in Collective Systems"**  
> *Submitted to Nature (2026)*  
> Djamel Bouchaffra, Faycal Ykhlef, Mustapha Lebbah, Hanane Azzag

The code reproduces all main results of the three validation domains:
- **Neural ensemble**  – analytic Gaussian coalition model (Shapley value derived from coalition free energies).
- **Schooling fish**   - analytic Gaussian coalition model (Shapley value derived from coalition free energies)
- **Multi-agent RL**   – analytic Gaussian coalition model (counterfactual Shapley values)

All simulations implement the **Game‑Theoretic Free Energy Principle** introduced in the paper, which bridges Bayesian inference, statistical physics, and game theory.

- **game-theoretic-free-energy-principle**
  - **neural_ensemble/
    - true-neural-ensemble-simulation.py` – Main simulation (LIF network, coalition free energy, Shapley)
    - plot_figure.py` – Generates Fig.1. (neural influence vs. precision)
    - data/` – Pre‑computed influence values (CSV)
  - **schooling_fish/**
    - true-school-fish-simulation.py` – Agent‑based fish school with precision‑dependent alignment
    - plot_figure.py` – Generates Fig.2. (fish influence vs. precision)
    - data/` – Pre‑computed influence values (CSV)
  - **marl/**
    - true-marl-simulation.py` – Implements the Game‑Theoretic Free Energy Principle (coalition free energies, Harsanyi dividends, Shapley value) for the multi‑agent RL domain.
    - plot_figure.py` – Generates Fig.3. (MARL influence vs. precision)
    - data/` – Pre‑computed influence values (CSV)
  - **combined_figure/**
    - plot_combined.py` – Generates Fig. 2d (normalised overlay)
  - requirements.txt` – Python dependencies (pip)
  - environment.yml` – Conda environment (optional)
  - LICENSE` – MIT License
  - README.md` – This file

## Requirements

- Python 3.9 or higher
- Packages listed in requirements.txt

Install with pip:
bash
pip install -r requirements.txt
Or create a conda environment:

bash
conda env create -f environment.yml
conda activate gt-fep
The main dependencies are:

numpy, scipy, matplotlib (core numerical and plotting)

torch (PyTorch for MARL)

pettingzoo (multi‑agent environment interface)

sklearn (quadratic fitting, normalisation)

Reproducing the figures
All figures from the paper can be regenerated without re‑running the simulations by using the pre‑computed data in the data/ subfolders.

Neural ensembles (Fig. 2a)
bash
cd neural_ensemble
python plot_figure.py
Schooling fish (Fig. 2b)
bash
cd schooling_fish
python plot_figure.py
Multi‑agent RL (Fig. 2c)
bash
cd marl
python plot_figure.py
Combined overlay (Fig. 2d)
bash
cd combined_figure
python plot_figure.py
Each script outputs a high‑resolution PNG image (600 dpi) ready for publication.

Running simulations from scratch
If you wish to verify or modify the simulations, you can run the main scripts. Warning: MARL training may take several hours on a standard CPU/GPU.

Neural ensemble
bash
cd neural_ensemble
python simulate_neural.py --beta_min 0.25 --beta_max 2.0 --steps 35 --runs 50
All parameters have default values matching the paper. Use --help to see all options.

Schooling fish
bash
cd schooling_fish
python simulate_fish.py --beta_min 0.05 --beta_max 4.0 --steps 18 --runs 80
Multi‑agent RL (MADDPG)
bash
cd marl
python train_marl.py --beta_min 0.2 --beta_max 5.0 --steps 15 --runs 100
This script trains independent policies for each precision value and computes counterfactual Shapley values.

All scripts use a fixed random seed (42) to ensure reproducibility.

Data files
The data/ folders contain CSV files with the following columns:

beta : sensory precision value

mean_influence : average influence (Shapley value or drop in directional persistence)

std_influence : standard deviation across runs

These files are provided so that figures can be regenerated without re‑running expensive simulations. They also allow independent verification of the statistical results.

License
This project is licensed under the MIT License – see the LICENSE file for details. You are free to use, modify, and distribute the code with attribution.

Citation
If you use this code or data in your own research, please cite our paper (DOI will be added upon publication):

bibtex
@article{Bouchaffra2026GTFEP,
  author    = {Bouchaffra, Djamel and Ykhlef, Faycal and Lebbah, Mustapha and Azzag, Hanane},
  title     = {A Variational Principle Unifying Bayesian Inference, Thermodynamics, and Game-Theoretic Equilibrium in Collective Systems},
  journal   = {Nature},
  year      = {2026},
  note      = {Submitted}
}
For the code itself, you may also cite this repository:

bibtex
@misc{gtfep2026code,
  author = {Bouchaffra, Djamel and Ykhlef, Faycal and Lebbah, Mustapha and Azzag, Hanane},
  title  = {Game-Theoretic Free Energy Principle: Simulation Code and Data},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/dbouchaffra/game-theoretic-energy-principle}
}
Contact
For questions, issues, or requests, please open an issue on this repository or contact the corresponding author:
Djamel Bouchaffra – djamel.bouchaffra@uvsq.fr
DAVID Lab, University of Paris-Saclay, UVSQ, 78035 Versailles, France.

