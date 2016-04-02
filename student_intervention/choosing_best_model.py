
# python standard library
import pickle
import warnings

# third party
import matplotlib.pyplot as plot
import numpy
import seaborn
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from tabulate import tabulate

# this code
from common import (TrainTestData, train_test_path, feature_map,
                    student_data, print_image_directive, SCALE, STYLE)

seaborn.set_style(STYLE)

with open(train_test_path, 'rb') as unpickler:
    train_test_data = pickle.load(unpickler)

assert len(train_test_data.y_train) == 300
assert len(train_test_data.y_test) == 95
assert len(train_test_data.X_train.columns) == 48

scorer = make_scorer(f1_score)
passing_ratio = (sum(train_test_data.y_test) +
                 sum(train_test_data.y_train))/float(len(train_test_data.y_test) +
                                                     len(train_test_data.y_train))
assert abs(passing_ratio - .67) < .01
model = LogisticRegression()

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
                  'class_weight': [None, 'balanced', {1: passing_ratio, 0: 1 - passing_ratio}],
                  }
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

grid_01 = fit_grid(numpy.arange(.01, 1., .01))

print("Parameters")
print("``````````\n")
print(tabulate(grid_01.best_params_.items(), headers='Parameter Value'.split(), tablefmt='rst'))

print('\nCoefficients')
print('````````````\n')
print_columns(grid_01)

column_names = train_test_data.X_test.columns
coefficients = grid_01.best_estimator_.coef_[0]
sorted_coefficients = sorted((column for column in coefficients), reverse=True)

non_zero_coefficients = [coefficient for coefficient in sorted_coefficients
                         if coefficient != 0]
non_zero_indices = [numpy.where(coefficients==coefficient)[0][0] for coefficient in non_zero_coefficients]
non_zero_variables = [column_names[index] for index in non_zero_indices]

from sklearn.preprocessing import MinMaxScaler
#from bokeh.models.sources import ColumnDataSource
#from bokeh.palettes import Spectral7
#from bokeh.models import HoverTool
#from bokeh.plotting import show
#from bokeh.plotting import figure as b_figure
#from bokeh.embed import components

#tools = "pan,wheel_zoom,box_zoom,reset,resize,hover"
figure = plot.figure(figsize=(10,8))
axe = figure.gca()
#fig = b_figure(tools=tools, title='Variable Probabilities', plot_width=1000)
scaler = MinMaxScaler()

for v_index, variable in enumerate(non_zero_variables):
    index = numpy.where(train_test_data.X_test.columns==variable)
    x_input = numpy.zeros((1, len(train_test_data.X_test.columns)))
    x = numpy.array([value for value in sorted(train_test_data.X_test[variable].unique())])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = scaler.fit_transform(x)
    y = []
    x = [x[0], x[-1]]
    for value in x:
        x_input[0][index] = value
        y.append(grid_01.best_estimator_.predict_proba(x_input)[0][1])
    lines = plot.plot(x, y, label=variable)
    #source = ColumnDataSource({variable: x, 'passed': y, 'name': [variable for item in x]})    
    #hover = fig.select(dict(type=HoverTool))
    #hover.tooltips = [('Variable', '@name'), ("(x, y)", '($x, $y)')]
    #fig.line(x, y, source=source, line_color=Spectral7[v_index], legend=variable)

title = axe.set_title("Variable Probabilities")
legend = axe.legend(loc='lower left')
#fig.legend.location = 'bottom_left'
#handle = show(fig)
#script, div = components(fig)
#for line in script.split('\n'):
#    print("   {0}".format(line))
#for line in div.split('\n'):
#    print("   {0}".format(line))
filename = 'coefficient_probabilities'
print_image_directive(filename=filename, figure=figure, scale=SCALE)

def make_countplots(title, x_name, filename, hue='passed'):
    figure = plot.figure()
    axe = figure.gca()
    axe.set_title(title)
    lines = seaborn.countplot(x=x_name, hue=hue, data=student_data)
    print_image_directive(filename=filename, figure=figure, scale=SCALE)

make_countplots("Mother's Education vs Passing", 'Medu', 'mothers_education')

make_countplots(title="Age vs Passing", x_name='passed', filename='student_age', hue='age')

make_countplots("Family Relations vs Passing", 'famrel', 'family_relations')

make_countplots("Father's Education vs Passing", 'Fedu', 'fathers_education')

figure = plot.figure()
axe = figure.gca()
axe.set_title('Distribution of Absences')
axe = seaborn.kdeplot(student_data[student_data.passed=='yes'].absences, label='passed')
axe.set_xlim([0, 80])
axe = seaborn.kdeplot(student_data[student_data.passed=='no'].absences, ax=axe, label="didn't pass")
print_image_directive(figure=figure, filename='absences', scale=SCALE)

make_countplots("Going Out vs Passing", 'goout', 'going_out')

make_countplots("Past Failures vs Passing", 'failures', 'past_failures')

print("The best F1 score for the Logistic Regression classifier was {0:.2f}, which is a slight improvement over the default Logistic Regression classifier used earlier which had an f1 of approximately 0.81 for the test set when trained with 300 training instances.".format(grid_01.score(train_test_data.X_test, train_test_data.y_test)))