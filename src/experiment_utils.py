"""Module for experiment utils."""

import numpy as np
import os
import pickle

from src.train import cv


def perform_experiments(X, y, experiment_dict, experiment_path="experiment_results"):
    for exp_name, exp_config in experiment_dict.items():
        print(f"Experiment {exp_name} in progress...")
        experiment_dict[exp_name]["scores"] = cv(
            X=X,
            y=y,
            experiment_config=exp_config,
            k_folds=5,
        )
        pickle_name = (
            exp_name + "_" + str(int(np.mean(experiment_dict[exp_name]["scores"])))
        )
        with open(os.path.join(experiment_path, pickle_name), "wb") as f:
            pickle.dump(experiment_dict[exp_name], f)


def find_best_experiments(k=5, experiment_path="experiment_results"):
    exp_names = []
    scores = []
    for pickle_names in os.listdir(experiment_path):
        exp_name, score = pickle_names.rsplit("_", 1)

        exp_names.append(exp_name)
        scores.append(int(score))

    max_score_indices = np.argsort(scores)[-min(k, len(scores)) :]
    best_experiments = []

    for index in max_score_indices:
        path_to_best = exp_names[index] + "_" + str(scores[index])
        with open(os.path.join(experiment_path, path_to_best), "rb") as f:
            best_experiments.append(pickle.load(f))

    return best_experiments
