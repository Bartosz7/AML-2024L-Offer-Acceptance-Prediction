import numpy as np
from tqdm import tqdm


def prepare_cv_indices(n_observations, k_folds):
    """
    Function creates cross-validation indices for k folds.

    Arguments:
        n_observations: Number of observations in whole dataset used in cross-validation
        k_folds: number of folds for cross-validation

    Returns:
        splits: Training and testing indices
    """
    indices = np.arange(n_observations)
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, n_observations // k_folds, dtype=int)
    fold_sizes[: n_observations % k_folds] += 1  # Distribute the remainder

    current = 0
    splits = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_indices, val_indices))
        current = stop

    return splits


def calculate_score(model, X_train, X_test, y_train, y_test):
    """
    Function calculates custom score. It takes 1000 observations from test set with highest
    probability of success and checks how many of them are truly 1. For each properly classified
    observation it adds 10 to score. Then it dimishes score by 200 for each feature in train set.

    Arguments:
        model: model used for fit and predictions
        X_train: numpy array containing training predictors
        X_test: numpy array containing test predictors
        y_train: numpy array containing training target variable
        y_test: numpy array containing test target variable

    Returns:
        score: custom score value for given data and model
    """
    model.fit(X_train, y_train)
    proba_preds = model.predict_proba(X_test)
    best_indices = np.argsort(proba_preds[:, 1])[-1000:]

    properly_classfied_count = np.sum(y_test[best_indices])
    n_feats = X_train.shape[1]

    print(
        f"Using {n_feats} features, we properly classified {properly_classfied_count} clients."
    )

    score = 10 * properly_classfied_count - 200 * n_feats
    return score


def cv(X, y, model, k_folds):
    """
    Function performs cross validation with custom scoring function

    Arguments:
        X: numpy array with predictors
        y: numpy array with target variable
        model: model used in cross-validation
        k_folds: number of folds for cross-validation

    Returns:
        scores: List of scores from custom metric for each cross-validation split
    """
    fold_indices = prepare_cv_indices(n_observations=X.shape[0], k_folds=k_folds)

    scores = []
    for train_indices, test_indices in fold_indices:
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        scores.append(calculate_score(model, X_train, X_test, y_train, y_test))

    return scores
