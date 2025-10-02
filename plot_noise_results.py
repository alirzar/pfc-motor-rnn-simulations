# -*- coding: utf-8 -*-
"""
Plots the results of the noise experiment, showing performance (MSE)
and manifold overlap as a function of feedback noise.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from motor_rnns_pfc import RNN, PFC_RNN, get_controlled_manifold

# --- SETUP ---
loaddir = 'data/noise_experiment/'
savdir = 'figures/noise_experiment/'
if not os.path.exists(savdir):
    os.makedirs(savdir)

# Load results and original data
results = np.load(os.path.join(loaddir, 'noise_experiment_results.npy'), allow_pickle=True).item()
run_data = np.load('data/fig2/experiment_results.npy', allow_pickle=True).item()
pert_data = np.load('data/fig2/relearning_results.npy', allow_pickle=True).item()

# Get original manifold basis
original_manifold_basis = run_data['manifold']['evec'][:, :run_data['decoding']['reduced_dim']]

# Get task info
stimulus = run_data['stimulus']
target = run_data['target']
pulse_length = run_data['params']['pulse_length']
wmp_decoder = pert_data['perturbations']['T_within']
omp_decoder = pert_data['perturbations']['T_outside']

# --- ANALYSIS LOOP ---
noise_factors = sorted(results.keys())
mse_results = {'wmp': [], 'omp': []}
overlap_results = {'wmp': [], 'omp': []}

for noise in noise_factors:
    print(f"Analyzing results for noise factor: {noise}")
    
    # --- Setup trained networks for analysis ---
    # Within-manifold
    pfc_wmp = PFC_RNN(N=1000, N_in=6, N_m1=800)
    pfc_wmp.W = results[noise]['wmp']['pfc_weights']
    pfc_wmp.W_projection = results[noise]['wmp']['proj_weights']
    m1_wmp = RNN(N=800)
    m1_wmp.W = results[noise]['wmp']['m1_weights']
    
    # Outside-manifold
    pfc_omp = PFC_RNN(N=1000, N_in=6, N_m1=800)
    pfc_omp.W = results[noise]['omp']['pfc_weights']
    pfc_omp.W_projection = results[noise]['omp']['proj_weights']
    m1_omp = RNN(N=800)
    m1_omp.W = results[noise]['omp']['m1_weights']
    
    # --- Calculate Manifold and MSE ---
    # WMP
    manifold_wmp = get_controlled_manifold(pfc_wmp, m1_wmp, 50, stimulus, pulse_length)
    cursor_vel_wmp = manifold_wmp['activity_reshaped'] @ wmp_decoder.T
    mse_wmp = np.mean((cursor_vel_wmp - target[manifold_wmp['order'], pulse_length:])**2)
    
    # OMP
    manifold_omp = get_controlled_manifold(pfc_omp, m1_omp, 50, stimulus, pulse_length)
    cursor_vel_omp = manifold_omp['activity_reshaped'] @ omp_decoder.T
    mse_omp = np.mean((cursor_vel_omp - target[manifold_omp['order'], pulse_length:])**2)
    
    mse_results['wmp'].append(mse_wmp)
    mse_results['omp'].append(mse_omp)
    
    # --- Calculate Manifold Overlap ---
    # Project new covariance onto original basis
    cov_wmp_proj = original_manifold_basis.T @ manifold_wmp['cov'] @ original_manifold_basis
    overlap_wmp = np.trace(cov_wmp_proj) / np.trace(manifold_wmp['cov'])
    
    cov_omp_proj = original_manifold_basis.T @ manifold_omp['cov'] @ original_manifold_basis
    overlap_omp = np.trace(cov_omp_proj) / np.trace(manifold_omp['cov'])
    
    overlap_results['wmp'].append(overlap_wmp)
    overlap_results['omp'].append(overlap_omp)

# --- PLOTTING ---
plt.style.use('seaborn-v0_8-notebook')

# Plot 1: MSE vs. Noise Factor
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(noise_factors, mse_results['wmp'], 'o-', label='Within-Manifold', color='darkred', lw=2)
ax1.plot(noise_factors, mse_results['omp'], 'o-', label='Outside-Manifold', color='darkblue', lw=2)
ax1.set_xlabel('Noise Factor (alpha)')
ax1.set_ylabel('Final Mean Squared Error (MSE)')
ax1.set_title('Performance vs. Feedback Noise')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
fig1.savefig(os.path.join(savdir, 'mse_vs_noise.png'), dpi=300)
plt.show()

# Plot 2: Manifold Overlap vs. Noise Factor
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(noise_factors, np.array(overlap_results['wmp']) * 100, 'o-', label='Within-Manifold', color='darkred', lw=2)
ax2.plot(noise_factors, np.array(overlap_results['omp']) * 100, 'o-', label='Outside-Manifold', color='darkblue', lw=2)
ax2.set_xlabel('Noise Factor (alpha)')
ax2.set_ylabel('Overlap with Original Manifold (%)')
ax2.set_title('Manifold Stability vs. Feedback Noise')
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
fig2.tight_layout()
fig2.savefig(os.path.join(savdir, 'overlap_vs_noise.png'), dpi=300)
plt.show()