import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def test_classifier(train_set, test_set, classifier, parameters):
    x_train, y_train = train_set
    x_test, y_test = test_set

    clf = classifier(**parameters)
    clf.fit(x_train, y_test)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    return accuracy, f1


def best_classifier(train_set, valid_set, test_set, classifier, hyperparameters, debug):
    """

    """

    def_hyp = {hyp: val_list[0] for hyp, val_list in hyperparameters.items()}  # Default hyperparameters
    results_acc = {}
    baseline_acc, _ = test_classifier(train_set, valid_set, classifier, def_hyp)
    running_best_acc = baseline_acc
    running_best_params = def_hyp.copy()

    if debug:
        print("Baseline accuracy for {} params: {}".format(def_hyp, baseline_acc))

    for hyp, val_list in hyperparameters.items():
        results_acc[hyp] = {}
        results_acc[hyp][def_hyp[hyp]] = baseline_acc

        params = def_hyp.copy()

        for val in val_list[1:]:
            params[hyp] = val

            if debug:
                print("Set {} to {}.".format(hyp, val))
                print(params)
            results_acc[hyp][val], _ = test_classifier(train_set, valid_set, classifier, params)
            if results_acc[hyp][val] > running_best_acc:
                running_best_acc = results_acc[hyp][val]
                running_best_params = params.copy()

            if debug:
                print("Accuracy: {}".format(results_acc[hyp][val]))

    # Obtain the best hyperparameters from the results.
    best_params = {hyp: max(result_dict, key=result_dict.get)
                   for hyp, result_dict in results_acc.items()}

    # If the accuracy of clf with oob_score > accuracy of clf with bootstrap = False
    if 'bootstrap' in best_params.keys() and best_params['oob_score']:
        max_bootstrap_val = results_acc['bootstrap'][False]
        max_oob_val = results_acc['oob_score'][True]
        if max_oob_val > max_bootstrap_val:
            best_params['bootstrap'] = True
        else:
            best_params['oob_score'] = False

    try:
        best_clf_acc, best_clf_f1 = test_classifier(train_set, valid_set, classifier, best_params)
    except ValueError as e:
        print(best_params)
        raise e

    best_params = running_best_params if running_best_acc > best_clf_acc else best_params

    if debug:
        print("Best params: {}".format(best_params))
        print("Best accuracy: {}".format(best_clf_acc))
        print("Best F1: {}".format(best_clf_f1))
        print()

    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    # Taking the first half of the training and the valid set
    x_train_valid = np.concatenate((x_train[len(x_train)//2:, :], x_valid[len(x_valid)//2:, :]), axis=0)
    y_train_valid = np.concatenate((y_train[len(y_train)//2:], y_valid[len(y_valid)//2:]), axis=0)
    train_valid_set = (x_train_valid, y_train_valid)

    test_acc, test_f1 = test_classifier(train_valid_set, test_set, classifier, best_params)

    return test_acc, test_f1, best_params


def best_parameters(train_set, valid_set, test_set, classifier, hyperparameters, debug=False):
    acc, f1, params = best_classifier(train_set, valid_set, test_set, classifier, hyperparameters, debug)
    print("Best {} Parameters: {}".format(classifier.__class__.__name__, params))
    print("Best {} Accuracy: {}".format(classifier.__class__.__name__, acc))
    print("Best {} F1: {}".format(classifier.__class__.__name__, f1))

    return acc, f1, params
