
# python standard library
import pickle
import warnings

# third party
import numpy
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from tabulate import tabulate

# this code
from common import TrainTestData, train_test_path, feature_map, TrainTestDataOne

with open(train_test_path, 'rb') as unpickler:
    train_test_data = pickle.load(unpickler)
#with open('saved_data.pkl', 'rb') as unpickler:
#    train_test_data = pickle.load(unpickler)
assert len(train_test_data.y_train) == 300
assert len(train_test_data.y_test) == 95

scorer = make_scorer(f1_score)
passing_ratio = (sum(train_test_data.y_test) +
                 sum(train_test_data.y_train))/float(len(train_test_data.y_test) +
                                                     len(train_test_data.y_train))
assert abs(passing_ratio - .67) < .01
model = LogisticRegression()
model.fit(train_test_data.X_train, train_test_data.y_train)
print(f1_score(model.predict(train_test_data.X_test), train_test_data.y_test))

def fit_grid(c_range, penalty=('l1', 'l2')):
    """
    :param:
     - `model`: LogisticRegression object
     - `c_range`: list of values for the 'C' parameter
     - `penalty`: names of the regularization penalties to use
    :return: GridSearchCV object fit to model
    """
    parameters = {'penalty': penalty,
                  'C': c_range,
                  'class_weight': [None, 'balanced', {1:passing_ratio, 0: 1-passing_ratio}]}
    grid = GridSearchCV(model, param_grid=parameters, scoring=scorer, cv=10, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return grid.fit(train_test_data.X_train, train_test_data.y_train)

def print_columns(grid):
    """
    :param:
     - `grid`: GridSearchCV object fitted to data
    """
    coefficients = grid.best_estimator_.coef_[0]
    odds = numpy.exp(coefficients)
    sorted_coefficients = sorted((column for column in coefficients), reverse=True)
    rows = []
    for coefficient in sorted_coefficients:
        if abs(coefficient) > 0:
            index = numpy.where(coefficients == coefficient)[0][0]
            column_name = train_test_data.X_train.columns[index]
            description = feature_map[column_name] if column_name in feature_map else ' = '.join(column_name.split('_'))
            rows.append([column_name, description, "{0:.2f}".format(coefficient), '{0:.2f}'.format(odds[index])])
    print(tabulate(rows, headers='Variable Description Coefficient Odds'.split(), tablefmt='rst'))

def print_best(grid):
    print("parameters")
    print("``````````\n")
    print(tabulate(grid.best_params_.items(), headers='Parameter Value'.split(), tablefmt='rst'))
    print('\nCoefficients')
    print('````````````\n')
    print_columns(grid)
    print('\nF1 score')
    print('````````\n')
    print("{0:.2f}".format(grid.score(train_test_data.X_test, train_test_data.y_test)))
    print('')

grid_01 = fit_grid(numpy.arange(.01, 1.1, .05))
print_best(grid_01)

grid_l1 = fit_grid(numpy.arange(.01, 1.1, .05), penalty=['l1'])
print_best(grid_l1)