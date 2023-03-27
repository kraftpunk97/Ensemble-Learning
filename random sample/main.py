import numpy as np
import warnings
from utils import best_parameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings('ignore')

num_clauses = (300, 500, 1000, 1500, 1800)
num_samples = (100, 1000, 5000)
template = "all_data/{}_c{}_d{}.csv"


def get_xy(filename):
    xy = np.loadtxt(filename, delimiter=',', dtype=np.int8)
    x = xy[:, :-1]
    y = xy[:, -1]

    return x, y


dt_records = {}
bagging_records = {}
gb_records = {}
rf_records = {}


def generate_latex():
    dt_text = ""
    rf_text = ""
    gb_text = ""
    bag_text = ""
    for combination in zip(num_clauses, num_samples):
        dt_text += " & ".join(dt_records[combination][1:])
        rf_text += " & ".join(rf_records[combination][1:])
        gb_text += " & ".join(gb_records[combination][1:])
        bag_text += " & ".join(bagging_records[combination][1:])


def main(debug=True):
    for clause in num_clauses:
        for samples in num_samples:
            print()
            print()
            print()
            print("Training classifiers on the {} clauses and {} samples.\n".format(clause, samples))

            train_file = template.format("train", clause, samples)
            valid_file = template.format("valid", clause, samples)
            test_file = template.format("test", clause, samples)

            train_set = get_xy(train_file)
            valid_set = get_xy(valid_file)
            test_set = get_xy(test_file)

            # DecisionTreeClassifier
            dt_clf_hyp = {
                'criterion': ('gini', 'entropy', 'log_loss'),  # The function to calculate the quality of the split.
                'splitter': ('best', 'random'),  # The strategy used to split each node.
                'min_samples_split': (2, 3, 4),  # The minimum number of samples needed to split a node.
                'min_samples_leaf': (1, 2, 5, 10),  # The minimum number of samples required to be at the leaf node.
                'max_features': (None, 2, 'sqrt', 'log2')
                # The number of features to consider when looking for the best split.
            }
            dt_acc, dt_f1, dt_params = best_parameters(train_set, valid_set, test_set,
                                                       DecisionTreeClassifier, dt_clf_hyp, debug=debug)
            dt_records[(clause, samples)] = (dt_params, dt_acc, dt_f1)

            # BaggingClassifier
            bagging_clf_hyp = {
                'n_estimators': (10, 5, 50, 100),
                'max_samples': (1.0, 0.5, 0.25, 2, 5, 10),
                'max_features': (1.0, 0.1, 0.25, 0.50),
                'bootstrap': (True, False),
                'bootstrap_features': (False, True),
                'oob_score': (False, True),
            }
            bag_acc, bag_f1, bag_params = best_parameters(train_set, valid_set, test_set,
                                                          BaggingClassifier, bagging_clf_hyp, debug=debug)
            bagging_records[(clause, samples)] = (bag_params, bag_acc, bag_f1)

            # RandomForestClassifier
            rf_clf_hyp = {
                'n_estimators': (100, 5, 10, 50),
                'criterion': ('gini', 'entropy', 'log_loss'),  # The function to calculate the quality of the split.
                'min_samples_split': (2, 3, 4),  # The minimum number of samples needed to split a node.
                'min_samples_leaf': (1, 2, 5, 10),  # The minimum number of samples required to be at the leaf node.
                'max_features': ('sqrt', 2, None, 'log2'),
                'bootstrap': (True, False),
                'oob_score': (False, True),
            }
            rf_acc, rf_f1, rf_params = best_parameters(train_set, valid_set, test_set,
                                                       RandomForestClassifier, rf_clf_hyp, debug=debug)
            rf_records[(clause, samples)] = (bag_params, rf_acc, rf_f1)

            # GradientBoostingClassifier
            gb_clf_hyp = {
                'loss': ('log_loss', 'exponential'),
                'learning_rate': (0.1, 0.01, 0.001, 1),
                'n_estimators': (100, 50, 200, 25, 400),
                'subsample': (1.0, 0.5, 0.1),
                'criterion': ('friedman_mse', 'squared_error'),
                'min_samples_split': (2, 3, 4),  # The minimum number of samples needed to split a node.
                'min_samples_leaf': (1, 2, 5, 10),  # The minimum number of samples required to be at the leaf node.
                'max_features': ('sqrt', 2, None, 'log2'),
            }
            gb_acc, gb_f1, gb_params = best_parameters(train_set, valid_set, test_set,
                                                       GradientBoostingClassifier, gb_clf_hyp, debug=debug)
            gb_records[(clause, samples)] = (bag_params, gb_acc, gb_f1)


if __name__ == '__main__':
    main(debug=False)
    #generate_latex()
