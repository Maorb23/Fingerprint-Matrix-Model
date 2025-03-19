import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy.stats import ortho_group
import argparse
import pandas as pd
import logging
from pathlib import Path
#import onehot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve, auc

from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import numpy as np
import json
import os



class SRM_train:
    def __init__(self, tol=1e-3, max_iter=100, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def W_i_calc3(self, X_i, S):
        X_S_T = np.dot(X_i, S.T)
        U_i, Sigma, V_i_T = np.linalg.svd(X_S_T, full_matrices=False)
        W_i = np.dot(U_i, V_i_T)
        return W_i

    def SRM(self, X, tol=1e-3, max_iter=1000):
        X = np.stack(X)
        dist_vec = []
        indices = []
        delta_S_list = []
        delta_W_list = []
        m, n_voxels, n_timepoints = X.shape
        k = n_voxels
        W_i_new_group = np.array([np.eye(n_voxels, k) for _ in range(m)])
        S_old = np.random.rand(k, n_timepoints)
        iter_count = 0
        converged = False

        while not converged and iter_count < max_iter:
            # Update S
            S = sum(np.dot(W_i.T, X_i) for W_i, X_i in zip(W_i_new_group, X)) / m
            delta_S = np.linalg.norm(S - S_old, 'fro')**2
            delta_S_list.append(delta_S)
            delta_W_sum = 0.0
            reconstruction_error_sum = 0.0
            recon_len = 0

            for j, X_i in enumerate(X):
                W_old = W_i_new_group[j].copy()
                W_i_new_group[j] = self.W_i_calc3(X_i, S)
                delta_W = np.linalg.norm(W_i_new_group[j] - W_old, 'fro')**2
                delta_W_sum += delta_W
                dist = np.linalg.norm(X_i - np.dot(W_i_new_group[j], S), 'fro')**2
                reconstruction_error_sum += dist
                recon_len += 1
                #print(f"Subject {j}, Delta W: {delta_W}")

            mean_W = delta_W_sum / m
            delta_W_list.append(mean_W)
            mean_dist = reconstruction_error_sum / recon_len
            dist_vec.append(mean_dist)
            indices.append(iter_count)

            if self.verbose:
                self.logger.info(f"Iteration: {iter_count}, Mean Distance: {mean_dist}, Mean W: {mean_W}, Delta S: {delta_S}")

            if mean_dist < tol or mean_W < tol:
                self.logger.info(f'Converged at iteration {iter_count} with mean distance: {mean_dist}, mean W: {mean_W}, and delta S: {delta_S}')
                converged = True

            S_old = S.copy()
            iter_count += 1
            
        self.logger.info(f'Final mean distance: {mean_dist}, mean W: {mean_W}, and delta S: {delta_S}')
        return iter_count, W_i_new_group, S, dist_vec, delta_S_list, delta_W_list


    def Stochastic_SRM(self, X, batch_size, tol=1e-3, max_iter=1000, learning_rate=0.1,verbose=False):
        X = np.stack(X)
        m, n_voxels, n_timepoints = X.shape
        k = n_voxels
        # Initialize W and S
        W_i_new_group = np.array([ortho_group.rvs(dim=k) for _ in X])
        S = np.random.rand(k, n_timepoints)
        S_old = S.copy()

        dist_vec = []
        delta_S_list = []
        delta_W_list = []
        indices = []

        iter_count = 0
        converged = False

        while not converged and iter_count < max_iter:
            sample_indices = np.arange(m)
            np.random.shuffle(sample_indices)

            # For a full iteration, we accumulate updates:
            W_change_accum = 0.0
            dist_accum = 0.0

            # Option 1: Update S after each batch (incrementally)
            # Or accumulate and update once at the end.
            # Below: accumulate and update once at the end of iteration:
            S_accum = np.zeros_like(S)

            for batch_start in range(0, m, batch_size):
                batch_indices = sample_indices[batch_start : batch_start + batch_size]

                # Compute batch contribution to S
                # Here we use the current W of just the batch subjects
                S_batch = sum(np.dot(W_i_new_group[i].T, X[i]) for i in batch_indices) / len(batch_indices)
                
                # Accumulate this batch's contribution for a full update at the end:
                S_accum +=  S_batch

                # Update W_i for this batch:
                batch_W_change = 0.0
                batch_dist = 0.0
                for subj_idx in batch_indices:
                    X_i = X[subj_idx]
                    W_old = W_i_new_group[subj_idx].copy()
                    W_i_new_group[subj_idx] = self.W_i_calc3(X_i, S_batch)
                    batch_W_change += np.linalg.norm(W_i_new_group[subj_idx] - W_old, 'fro')**2
                    batch_dist += np.linalg.norm(X_i - np.dot(W_i_new_group[subj_idx], S_batch), 'fro')**2

                # Accumulate statistics
                W_change_accum += batch_W_change
                dist_accum += batch_dist

            # After processing all batches in this iteration:
            # Update S using accumulated average
            S_new = S_accum
            delta_S = np.linalg.norm(S_new - S, 'fro')**2
            delta_S_list.append(delta_S)
            S = S_new.copy()

            mean_W = W_change_accum / m
            mean_dist = dist_accum / m

            delta_W_list.append(mean_W)
            dist_vec.append(mean_dist)
            indices.append(iter_count)

            if self.verbose:
                self.logger.info(f"Iteration {iter_count}, Mean Distance: {mean_dist}, Mean Delta W: {mean_W}, Delta S: {delta_S}")

            if mean_dist < tol or mean_W < tol:
                self.logger.info(f'Converged at iteration {iter_count} with mean distance: {mean_dist}, mean W: {mean_W}, and delta S: {delta_S}')
                converged = True

            iter_count += 1
        self.logger.info(f'Final mean distance: {mean_dist}, mean W: {mean_W}, and delta S: {delta_S}')
        return iter_count, W_i_new_group, S, dist_vec, delta_S_list, delta_W_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SRM model')
    parser.add_argument('--input_path', type=str, default='data/processed', help='Path to input data')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Path to save the output data')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for stochastic SRM')
    parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for convergence')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic SRM')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    SRM_trainer = SRM_train(tol=args.tol, max_iter=args.max_iter, verbose=args.verbose)

    logger.info(f"Loading data from {input_path}")
    data_boring = list(np.load(input_path / 'boring_arrays.npz').values())
    data_interesting = list(np.load(input_path / 'interesting_arrays.npz').values())
    data_knock = list(np.load(input_path / 'knock_arrays.npz').values())
    data_standup = list(np.load(input_path / 'standup_arrays.npz').values())


    iter_count, W_list_boring, S_boring, dist_vec_boring, delta_S_list_boring,delta_W_boring = SRM_trainer.SRM(data_boring)
    logger.info(f"SRM completed in {iter_count} iterations. Saving results to {output_path}")
    # Save all matrices in a single .npz file
    np.savez(output_path / 'srm_matrices.npz',
            W_list=W_list_boring,
            S=S_boring,
            distances=dist_vec_boring,
            delta_S=W_list_boring,
            delta_W=delta_W_boring)
    
    iter_count, W_list_interesting, S_interesting, dist_vec_interesting, delta_S_list_interesting,delta_W_interesting = SRM_trainer.SRM(data_interesting)
    logger.info(f"SRM completed in {iter_count} iterations. Saving results to {output_path}")
    # Save all matrices in a single .npz file
    np.savez(output_path / 'srm_matrices.npz',
            W_list=W_list_interesting,
            S=S_interesting,
            distances=dist_vec_interesting,
            delta_S=W_list_interesting,
            delta_W=delta_W_interesting)
    
    iter_count, W_list_knock, S_knock, dist_vec_knock, delta_S_list_knock,delta_W_knock = SRM_trainer.SRM(data_knock)
    logger.info(f"SRM completed in {iter_count} iterations. Saving results to {output_path}")
    # Save all matrices in a single .npz file
    np.savez(output_path / 'srm_matrices.npz',
            W_list=W_list_knock,
            S=S_knock,
            distances=dist_vec_knock,
            delta_S=W_list_knock,
            delta_W=delta_W_knock)
    
    iter_count, W_list_standup, S_standup, dist_vec_standup, delta_S_list_standup,delta_W_standup = SRM_trainer.SRM(data_standup)
    logger.info(f"SRM completed in {iter_count} iterations. Saving results to {output_path}")
    # Save all matrices in a single .npz file
    np.savez(output_path / 'srm_matrices.npz',
            W_list=W_list_standup,
            S=S_standup,
            distances=dist_vec_standup,
            delta_S=W_list_standup,
            delta_W=delta_W_standup)



    