# Noise Simulation Scripts

This repository contains Python scripts used for simulating, analyzing, and visualizing noise effects in recurrent neural network (RNN) models of motor and prefrontal cortex (PFC) activity.

## Contents

- **run_noise_simulations.py**  
  Runs the main batch of noise simulations across different parameter settings and model conditions. Handles configuration, parallel simulation runs, and output generation.

- **motor_rnns_pfc.py**  
  Core neural network model implementation for motor control and PFC function simulation. Contains architecture definitions, training routines, and setup for experiments.

- **plot_noise_results.py**  
  Loads simulation output and generates figures to visualize noise impacts on RNN performance and neural dynamics.

## Usage

1. Configure parameters and run simulations using `run_noise_simulations.py`.
2. The simulated data makes use of models defined in `motor_rnns_pfc.py`.
3. Once simulations are complete, use `plot_noise_results.py` to create summary visualizations.

These scripts were developed for research into neural computation and the effects of noise on motor and PFC model function. Feel free to use, modify, or cite as appropriate!
# pfc-motor-rnn-simulations
A collection of scripts for running and visualizing noise simulations in motor/PFC recurrent neural network models for neuroscience research.
