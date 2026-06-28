import numpy as np
import itertools
import time
from sklearn.model_selection import KFold

def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.mean(np.abs(y_pred - y_true))

def cv(X, y, method, parameters, loss_function = mean_absolute_error, nfolds = 10, nrepetitions = 5):
    param_names = list(parameters.keys())
    param_values = list(parameters.values())

    combinations = list(itertools.product(*param_values))
    total_jobs = len(combinations) * nrepetitions * nfolds
    completed = 0

    start_time = time.time()

    losses = []

    for combination in combinations:
        params = dict(zip(param_names, combination))

        fold_losses = []

        for repetition in range(nrepetitions):
            kf = KFold(n_splits=nfolds, shuffle=True, random_state=repetition)

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = method(**params)
                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

                loss = loss_function(y_test, prediction)
                fold_losses.append(loss)

                completed += 1

                elapsed_time = time.time() - start_time
                avg = elapsed_time / completed
                remaining = avg * (total_jobs - completed)

                print(
                    f"\rProgress: {completed}/{total_jobs} "
                    f"({100 * completed / total_jobs:.1f}%) "
                    f"Remaining: {remaining:.1f}s",
                    end=""
                )
        
        avg_loss = np.mean(fold_losses)
        losses.append((avg_loss, params))
    
    print()

    if len(combinations) == 1:
        best_loss, best_params = losses[0]
    else:
        best_loss, best_params = min(losses, key=lambda x: x[0])

    best_model = method(**best_params)
    best_model.fit(X, y)

    best_model.cvloss = best_loss

    return best_model

class krr:
    
    def __init__(
            self,
            kernel='linear',
            kernelparameter=None,
            regularization=1.0
    ):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

        self.alpha = None
        self.Xtrain = None

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'polynomial':
            return (X1 @ X2.T + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            sigma = self.kernelparameter

            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)

            dist = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            return np.exp(-dist / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
            
    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        
        if self.regularization == 0:
            mean = np.mean(np.linalg.eigvalsh(self._kernel(Xtrain, Xtrain)))

            candidates = mean * np.logspace(-3, 3, 10)

            model = cv(
                Xtrain,
                ytrain, 
                krr, 
                {
                    'kernel': [self.kernel], 
                    'kernelparameter': [self.kernelparameter], 
                    'regularization': candidates
                }
            )

            self.regularization = model.regularization
        
        K = self._kernel(Xtrain, Xtrain)
        n = K.shape[0]

        self.alpha = np.linalg.solve(K + self.regularization * np.eye(n), ytrain)

        return self
    
    def predict(self, Xtest):
        Ktest = self._kernel(Xtest, self.Xtrain)
        return Ktest @ self.alpha