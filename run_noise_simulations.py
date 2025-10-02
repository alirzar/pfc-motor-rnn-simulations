# -*- coding: utf-8 -*-
"""
Simulation script to test the 2-RNN PFC-M1 model's robustness to 
feedback noise for both within- and outside-manifold perturbations.
"""
import numpy as np
import os
import copy
from motor_rnns_pfc import RNN, PFC_RNN, create_population_targeted_projection, get_controlled_manifold, add_noise_to_feedback

# --- SETUP ---
savdir = 'data/noise_experiment/'
if not os.path.exists(savdir):
    os.makedirs(savdir)

# Load the pre-trained M1-RNN and experimental data
run_data = np.load('data/fig2/experiment_results.npy', allow_pickle=True).item()
m1_rnn = RNN(N=run_data['params']['N'], g=run_data['params']['g'], p=run_data['params']['p'])
m1_rnn.W = np.load('data/fig2/W_stabilized.npy')

# Load task parameters and decoders from the initial run
stimulus = run_data['stimulus']
target = run_data['target']
pulse_length = run_data['params']['pulse_length']
n_learning_trials = 80

# Load the perturbation decoders and feedback matrices
pert_data = np.load('data/fig2/relearning_results.npy', allow_pickle=True).item()
wmp_decoder = pert_data['perturbations']['T_within']
omp_decoder = pert_data['perturbations']['T_outside']
wmp_feedback = pert_data['relearning']['feedback_within']
omp_feedback = pert_data['relearning']['feedback_outside']

# --- EXPERIMENT LOOP ---
noise_factors = [2, 3, 4, 5]
results = {}

for noise in noise_factors:
    print(f"\n--- RUNNING SIMULATION FOR NOISE FACTOR: {noise} ---")
    
    # --- Within-Manifold Condition ---
    print("Training for Within-Manifold Perturbation...")
    noisy_wmp_feedback = add_noise_to_feedback(wmp_feedback, alpha=noise)
    
    pfc_rnn_wmp = PFC_RNN(N=1000, N_in=6, N_m1=800)
    m1_rnn_wmp = copy.deepcopy(m1_rnn)
    W_proj_wmp = create_population_targeted_projection(pfc_rnn_wmp.N, m1_rnn_wmp.N)
    
    loss_wmp = pfc_rnn_wmp.relearn(
        trials=n_learning_trials, ext=stimulus, ntstart=pulse_length,
        decoder=wmp_decoder, feedback=noisy_wmp_feedback, target=target,
        delta_pfc=5000.0, delta_m1=5e6, lr_proj=1e-7, # Tuned learning rates
        M1_RNN=m1_rnn_wmp, W_projection=W_proj_wmp
    )
    
    # --- Outside-Manifold Condition ---
    print("Training for Outside-Manifold Perturbation...")
    noisy_omp_feedback = add_noise_to_feedback(omp_feedback, alpha=noise)

    pfc_rnn_omp = PFC_RNN(N=1000, N_in=6, N_m1=800)
    m1_rnn_omp = copy.deepcopy(m1_rnn)
    W_proj_omp = create_population_targeted_projection(pfc_rnn_omp.N, m1_rnn_omp.N)

    loss_omp = pfc_rnn_omp.relearn(
        trials=n_learning_trials, ext=stimulus, ntstart=pulse_length,
        decoder=omp_decoder, feedback=noisy_omp_feedback, target=target,
        delta_pfc=5000.0, delta_m1=5e6, lr_proj=1e-7, # Tuned learning rates
        M1_RNN=m1_rnn_omp, W_projection=W_proj_omp
    )

    # --- SAVE RESULTS FOR THIS NOISE LEVEL ---
    results[noise] = {
        'wmp': {
            'loss': loss_wmp,
            'pfc_weights': pfc_rnn_wmp.W,
            'm1_weights': m1_rnn_wmp.W,
            'proj_weights': W_proj_wmp,
        },
        'omp': {
            'loss': loss_omp,
            'pfc_weights': pfc_rnn_omp.W,
            'm1_weights': m1_rnn_omp.W,
            'proj_weights': W_proj_omp,
        }
    }

# Save all results to a single file
np.save(os.path.join(savdir, 'noise_experiment_results.npy'), results)
print("\n--- All simulations complete. Results saved. ---")