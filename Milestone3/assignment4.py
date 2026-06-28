from stubs.sheet3 import krr, zero_one_loss, mean_absolute_error
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools as it

subtask = "b"

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    n, d = X.shape
    cv_loss_min = np.inf
    param_combo = it.product(*params.values())
    n_params = np.prod([len(x) for x in params.values()])
    for itr, param in  enumerate(param_combo):
        failed = False

        method_with_param = method(*param)
        error = 0
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

                try:
                    method_with_param = method_with_param.fit(X_train, y_train)
                except np.linalg.LinAlgError as e:
                    print("Singular matrix:", e)
                    failed = True
                    break

                y_pred = method_with_param.predict(X_test)
                error += loss_function(y_test, y_pred)

            if failed:
                break

        if failed:
            continue

        cv_loss = error / nfolds / nrepetitions

        if n_params == 1:
            return cv_loss

        if cv_loss_min > cv_loss:
            cv_loss_min = cv_loss
            param_best = param

        if itr % 20 == 0:
            print(f"Validating parameter combination {itr+1} out of {n_params}.")


    method_with_param = method(*param_best).fit(X,y)
    method_with_param.cvloss = cv_loss_min

    return method_with_param, cv_loss_min

match subtask:
    case "a":
        results = {}
        subsets = ["banana", "diabetis", "flare-solar", "image", "ringnorm"]

        params1 = {
            "kernel": ["linear"],
            "kernelparameter": [1],
            "C": [0.01, 0.1, 1, 10, 100]
        }
        params2 = {
            "kernel": ["polynomial"],
            "kernelparameter": [1,2,3,4,5],
            "C": [0.01, 0.1, 1, 10, 100]
        }
        params3 = {
            "kernel": ["gaussian"],
            "kernelparameter": [0.01, 0.1, 1, 10, 100], #standard deviation for gaussian kernel
            "C": [0.01, 0.1, 1, 10, 100]
        }

        for sub in subsets:
            xtest = np.loadtxt("data/U04_" + sub + "-xtest.dat").T
            xtrain = np.loadtxt("data/U04_" + sub + "-xtrain.dat").T
            ytrain = np.loadtxt("data/U04_" + sub + "-ytrain.dat").T
            ytest = np.loadtxt("data/U04_" + sub + "-ytest.dat").T
            #print("------- ", xtrain.shape, xtest.shape)
            results[sub] = {}

            cvloss_best = np.inf
            for params in [params1, params2, params3]:
                method, cvloss = cv(xtrain, ytrain, krr, params, zero_one_loss)
                if cvloss < cvloss_best:
                    best_params = params
                    best_method = method
                    cvloss_best = cvloss


            print(f"Best parameters for dataset {sub}: kernel: {best_method.kernel}, kernelparameter: {best_method.kernelparameter}, C: {best_method.regularization}")
            results[sub]["kernel"] = best_method.kernel
            results[sub]["kernelparameter"] = best_method.kernelparameter
            results[sub]["regularization"] = best_method.regularization
            results[sub]["cvloss"] = cvloss
            with open("log.txt", "a") as f:
                f.write(f"{sub},{cvloss},{best_method.kernel},{best_method.kernelparameter},{best_method.regularization}\n")

        with open("results.p", "wb") as f:
            pickle.dump(results, f)

        with open("results.p", "rb") as f:
            results = pickle.load(f)
        print(results)

    case "b":
        def auc(y_true, y_pred, plot=False):
            o = np.argsort(y_pred)[::-1]
            y_sorted = y_true[o]

            delta_neg = 1 / np.sum(y_true == -1)
            delta_pos = 1 / np.sum(y_true == +1)

            fpr = np.zeros(len(y_true) + 1)
            tpr = np.zeros(len(y_true) + 1)

            for i in range(1, len(y_true) + 1):
                yi = y_sorted[i - 1]

                fpr[i] = fpr[i - 1] + 0.5 * (1 - yi) * delta_neg
                tpr[i] = tpr[i - 1] + 0.5 * (1 + yi) * delta_pos

            auc_val = np.trapz(tpr, fpr)

            if plot:
                plt.figure()
                plt.plot(fpr, tpr, marker='o', label=f'AUC = {auc_val:.3f}')
                plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.show()

            return auc_val

        def roc_fun(y_true, y_pred):
            biases = np.linspace(-np.max(np.abs(y_pred)) - 1,
                                np.max(np.abs(y_pred)) + 1,
                                100)

            fpr = []
            tpr = []

            for b in biases:
                y_hat = np.sign(y_pred + b)

                tp = np.sum((y_hat == 1) & (y_true == 1))
                fp = np.sum((y_hat == 1) & (y_true == -1))
                tn = np.sum((y_hat == -1) & (y_true == -1))
                fn = np.sum((y_hat == -1) & (y_true == 1))

                tpr.append(tp / (tp + fn))
                fpr.append(fp / (fp + tn))

            fpr = np.array(fpr)
            tpr = np.array(tpr)

            order = np.argsort(fpr)
            auc = np.trapz(tpr[order], fpr[order])

            return 1 - auc

        params1 = {
            "kernel": ["linear"],
            "kernelparameter": [1],
            "C": [0]
        }
        params2 = {
            "kernel": ["polynomial"],
            "kernelparameter": [1,2,3,4,5],
            "C": [0]
        }
        params3 = {
            "kernel": ["gaussian"],
            "kernelparameter": [0.01, 0.1, 1, 10, 100], #standard deviation for gaussian kernel
            "C": [0]
        }

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random', color="black")


        for sub, cc in zip(["banana", "diabetis", "flare-solar", "image", "ringnorm"], ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:purple"]):
            xtest = np.loadtxt("data/U04_" + sub + "-xtest.dat").T
            xtrain = np.loadtxt("data/U04_" + sub + "-xtrain.dat").T
            ytrain = np.loadtxt("data/U04_" + sub + "-ytrain.dat").T
            ytest = np.loadtxt("data/U04_" + sub + "-ytest.dat").T

            cvloss_best = np.inf
            for params in [params1, params2, params3]:
                method, cvloss = cv(xtrain, ytrain, krr, params, roc_fun)
                if cvloss < cvloss_best:
                    best_params = params
                    best_method = method
                    cvloss_best = cvloss

            y_pred = best_method.predict(xtest)

            o = np.argsort(y_pred)[::-1]
            y_sorted = ytest[o]

            delta_neg = 1 / np.sum(ytest == -1)
            delta_pos = 1 / np.sum(ytest == +1)

            fpr = np.zeros(len(ytest) + 1)
            tpr = np.zeros(len(ytest) + 1)

            for i in range(1, len(ytest) + 1):
                yi = y_sorted[i - 1]

                fpr[i] = fpr[i - 1] + 0.5 * (1 - yi) * delta_neg
                tpr[i] = tpr[i - 1] + 0.5 * (1 + yi) * delta_pos

            auc_val = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{sub} AUC = {auc_val:.3f}', color=cc)


            #auc(ytest, y_pred, plot=True)

            print(f"Best parameters for dataset {sub}: kernel: {best_method.kernel}, kernelparameter: {best_method.kernelparameter}, C: {best_method.regularization}")

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()