from stubs.sheet3 import krr, zero_one_loss, mean_absolute_error
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools as it
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

data = loadmat("data/qm7.mat")
X = data['X']
R = data['R']
Z = data['Z']
T = data['T'].T
P = data['P']
print(X.shape, R.shape, Z.shape, T.shape, P.shape)

eigvals, eigvecs = np.linalg.eig(X)
print(eigvals.shape)

distances_X = cdist(eigvals, eigvals, metric="euclidean")
distances_y = cdist(T, T, metric="cityblock")
print(distances_X.shape, distances_y.shape)

'''
plt.scatter(distances_X.ravel(), distances_y.ravel(), s=1)
plt.xlabel("X distances (Coulomb)")
plt.ylabel("Y distances (kcal /mol)")
plt.show()
'''

X_train, X_test, y_train, y_test = train_test_split(
    eigvals, T,
    train_size=5000,
    test_size=2165,
    shuffle=True,
    random_state=42
)

Xtrain, Xval, Ytrain, Yval = train_test_split(
    eigvals, T,
    train_size=2500,
    test_size=2500,
    shuffle=True,
    random_state=42
)

sigma = np.quantile(
    distances_X,
    [0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
)
C = np.logspace(-7, 0, 8)
params = {
    "kernel": ["gaussian"],
    "kernelparameter": sigma,
    "C": C
}
'''
best_method = cv(Xval, Yval, krr, params, mean_absolute_error, 5)
print(f"Best params: sigma: {best_method.kernelparameter}, C: {best_method.regularization}")
#best_method = best_method.fit(X_test, y_test)
y_pred = best_method.predict(X_test)
err = mean_absolute_error(y_test, y_pred)
print("mean abs error ", err)
'''
'''

krr_ = krr(kernel="gaussian", kernelparameter=37.1, regularization=0.0001)
errs = []
for n in range(100, 5000, 50):
    if n % 10 == 0:
        print(n)
    sub_X_train = X_train[:n]
    sub_y_train = y_train[:n]

    krr_ = krr_.fit(sub_X_train, sub_y_train)
    y_pred = krr_.predict(X_test)
    errs.append(mean_absolute_error(y_test, y_pred))

plt.plot(range(100, 5000, 50), errs)
plt.xlabel("Number of Training Samples")
plt.ylabel("Mean Absolute Error (kcal / mol)")
plt.show()
'''

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    n, d = X.shape
    cv_loss_min = np.inf
    param_combo = it.product(*params.values())
    n_params = np.prod([len(x) for x in params.values()])

    results = []

    for itr, param in enumerate(param_combo):
        if param[0] == "polynomial":
            if not isinstance(param[1], int):
                continue

        train_error = 0
        val_error = 0
        count = 0

        for _ in range(nrepetitions):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            X_partitions = np.array_split(X_shuffled, nfolds)
            y_partitions = np.array_split(y_shuffled, nfolds)

            for i in range(nfolds):
                X_train = np.concatenate(X_partitions[:i] + X_partitions[i+1:], axis=0)
                y_train = np.concatenate(y_partitions[:i] + y_partitions[i+1:], axis=0)
                X_test = X_partitions[i]
                y_test = y_partitions[i]

                model = method(*param)
                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                train_error += loss_function(y_train, y_pred_train)
                val_error += loss_function(y_test, y_pred_test)
                count += 1

        train_loss = train_error / count
        cv_loss = val_error / count

        results.append({
            "param": param,
            "train_loss": train_loss,
            "val_loss": cv_loss,
            "gap": cv_loss - train_loss
        })

        if n_params == 1:
            return cv_loss

        if cv_loss_min > cv_loss:
            cv_loss_min = cv_loss
            param_best = param

    train_losses = np.array([r["train_loss"] for r in results])
    val_losses = np.array([r["val_loss"] for r in results])
    gaps = np.array([r["gap"] for r in results])

    underfit_idx = np.argmax(train_losses + val_losses)
    overfit_idx = np.argmax(gaps)

    underfit_param = results[underfit_idx]["param"]
    overfit_param = results[overfit_idx]["param"]

    method_with_param = method(*param_best)
    method_with_param.fit(X, y)
    method_with_param.cvloss = cv_loss_min
    method_with_param.cv_results = results
    method_with_param.underfit_param = underfit_param
    method_with_param.overfit_param = overfit_param

    return method_with_param


sub_X_train = X_train[:1000]
sub_y_train = y_train[:1000]

params = {
    "kernel": ["linear", "gaussian"],
    "kernelparameter": sigma,
    "C": C
}

method = cv(sub_X_train, sub_y_train, krr, params, mean_absolute_error, nfolds=5)

# well fit

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

models = [
    ("Underfit", method.underfit_param),
    ("Best fit", (method.kernel, method.kernelparameter, method.regularization)),
    ("Overfit", method.overfit_param)
]

for a, (title, param) in zip(ax, models):
    model = krr(kernel=param[0], kernelparameter=param[1], regularization=param[2])
    model.fit(sub_X_train, sub_y_train)

    y_pred_train = model.predict(sub_X_train)
    y_pred_test = model.predict(X_test)

    a.scatter(sub_y_train, y_pred_train, s=10, color="tab:blue", label="Train")
    a.scatter(y_test, y_pred_test, s=10, color="tab:orange", label="Test")

    lo = min(sub_y_train.min(), y_test.min(), y_pred_train.min(), y_pred_test.min())
    hi = max(sub_y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())
    a.plot([lo, hi], [lo, hi], "k--")

    a.set_title(title)
    a.set_xlabel("Ground truth $y_i$")
    a.set_ylabel("Prediction $\\hat{y}_i$")
    a.legend()

plt.tight_layout()
plt.show()
