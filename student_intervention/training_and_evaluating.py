
#python standard library
import pickle
import time
from collections import namedtuple

# third party
import matplotlib.pyplot as plot
import numpy
import seaborn
from sklearn.metrics import f1_score

# this code
from common import TrainTestData, print_image_directive, RANDOM_STATE

with open('pickles/train_test_data.pkl', 'rb') as unpickler:
    data = pickle.load(unpickler)
X_train, X_test, y_train, y_test = data

REPETITIONS = 10
INTERESTING_SIZES = (100, 200, 300)
STEP_SIZE = 10
MAX_SIZE = 300 + STEP_SIZE
MIN_SIZE = 10

class Classifier(object):
    """
    Trains, predicts, evaluates classifier using f1 score
    """
    def __init__(self, classifier, x_train, y_train, x_test, y_test,
                 delim='\t', repetitions=1):
        """
        :param:
         - `classifier`: sklearn classifier object
         - `x_train`: feature training data
         - `y_train`: target training data
         - `x_test`: feature test data
         - `y_test`: target test data
         - `delim`: separator for the table row
         - `repetitions`: number of times to repeat fitting and predictions
        """
        self.clf = classifier
        self.repetitions = repetitions
        self._classifier = None
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self._f1_training = None
        self._f1_test = None
        self.delim = delim
        self._table_row = None
        self._training_time = None
        self._prediction_time = None
        return

    @property
    def f1_training(self):
        """
        :return: F1 score using training data
        """
        if self._f1_training is None:
            predictions, time_ = self.predict(self.x_train)
            self._f1_training = self.f1_score(predictions, self.y_train)
        return self._f1_training

    @property
    def f1_test(self):
        """
        :return: f1 score for test-set predictions
        :postcondition: self.prection_time set
        """
        if self._f1_test is None:
            predictions, self._prediction_time = self.predict(self.x_test)
            self._f1_test = self.f1_score(predictions, self.y_test)
        return self._f1_test

    @property
    def prediction_time(self):
        """
        :return: prediction time for test data
        """
        if self._prediction_time is None:
            predictions, self._prediction_time = self.predict(self.x_test)
            self._f1_test = self.f1_score(predictions, self.y_test)
        return self._prediction_time

    @property
    def training_time(self):
        """
        :return: training time in seconds
        """
        if self._training_time is None:
            times = []
            for repetition in range(self.repetitions):
                start = time.time()
                self._classifier = self.clf.fit(self.x_train, self.y_train)
                times.append(time.time() - start)
            self._training_time = numpy.min(times)
        return self._training_time
        
    @property
    def classifier(self):
        """
        :return: trained classifier
        """
        if self._classifier is None:
            times = []
            for repetition in range(self.repetitions):                                    
                 start = time.time()
                 self._classifier = self.clf.fit(self.x_train, self.y_train)
                 times.append(time.time() - start)
            self._training_time = numpy.min(times)
        return self._classifier

    def f1_score(self, predictions, target):
        """
        :param:
         - `predictions`: predicted values for model
         - `target`: actual outcomes from data
        :return: f1 score for predictions
        """
        return numpy.median([f1_score(target.values, prediction, pos_label=1) for prediction in predictions])

    def predict(self, features):
        """
        :param:
         - `features`: array of feature data
        :return: predicted values, time to execute
        """
        times = []
        all_predictions = []
        for repetition in range(self.repetitions):
            start = time.time()
            predictions = self.classifier.predict(features)
            elapsed = time.time() - start
            times.append(elapsed)
            all_predictions.append(predictions)
        return all_predictions, numpy.min(times)

    def train_and_predict(self):
        """
        :return: time, f1 score for training and testing data
        """
        train_predictions, train_predictions_time = self.predict(self.x_train)
        train_f1_score = self.f1_score(train_predictions, self.y_train)
        
        test_predictions, test_predictions_time = self.predict(self.x_test)
        test_f1_score = self.f1_score(test_predictions, self.y_test)
        return (train_predictions_time, train_f1_score,
                test_predictions_time, test_f1_score)
    
    @property
    def table_row(self):
        """
        :return: string of training size, training time, prediction time, f1 train, f1 test
        """
        if self._table_row is None:
            self._table_row = self.delim.join([str(len(self.x_train))] +
                                              ["{0:.4f}".format(item) for item in (self.training_time,
                                                                                   self.prediction_time,
                                                                                   self.f1_training,
                                                                                   self.f1_test)])
        return self._table_row

def train_and_predict(model, delimiter=','):
    """
    :param:
     - `model`: classifier object
     - `delimiter`: separator for terms
    :return: best scores and dict of classifier-containers for each size
    """
    scores = []
    all_tests = {}
    for size in range(MIN_SIZE, MAX_SIZE, STEP_SIZE):
        x_train_subset, y_train_subset = X_train[:size], y_train[:size]
        classifier = Classifier(model, x_train_subset, y_train_subset,
                                X_test, y_test, delim=delimiter,
                                repetitions=REPETITIONS)
        if size in INTERESTING_SIZES:
            print("   {0}".format(classifier.table_row))
            scores.append((classifier.f1_test, size, classifier))
        all_tests[size] = classifier
    return max(scores) , all_tests

ClassifierData = namedtuple('ClassifierData', 'score size name container'.split())

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

classifiers = [LogisticRegression(),
               RandomForestClassifier(),
               KNeighborsClassifier()]
best_scores = []
line_width = 80
containers = {}
for classifier in classifiers:    
    print('')
    print('.. csv-table:: {0}'.format(classifier.__class__.__name__))
    print("   :header: Size,Time (train),Time (predict),Train F1,Test F1")
    print('')
    scores, all_containers = train_and_predict(classifier)
    best_score, best_size, classifier_container = scores
    containers[classifier.__class__.__name__] = all_containers
    print('Best score and size of data-set that gave the best test score:')
    print(" * best score: {0:.2f}".format(best_score))
    print(" * best size: {0}".format(best_size))
    data = ClassifierData(score=best_score, size=best_size,
                          name=classifier.__class__.__name__,
                          container=classifier_container)
    best_scores.append(data)
print('')

#print(".. table:: Ranked by Score")
from tabulate import tabulate
table = [[best.name, "{0:.2f}".format(best.score), best.size,
          '{0:.4f}'.format(best.container.training_time), '{0:.4f}'.format(best.container.prediction_time)]
         for index,best in enumerate(sorted(best_scores, reverse=True))]
print(tabulate(table, headers='Classifier Score Training-Size Training-Time Prediction-Time'.split(), tablefmt='rst'))

SIZE = 300
knn_300 = containers['KNeighborsClassifier'][SIZE]
logistic_300 = containers['LogisticRegression'][SIZE]
forests_300 = containers['RandomForestClassifier'][SIZE]
decimals = '{0:.2f}'.format

table = [['LogisticRegression/KNeighborsClassifier', 
          decimals(logistic_300.training_time/knn_300.training_time)],
         ['RandomForestClassifier/KNeighborsClassifier',
          decimals(forests_300.training_time/knn_300.training_time)],
         ['RandomForestClassifier/LogisticRegression',
          decimals(forests_300.training_time/logistic_300.training_time)]]
print(tabulate(table, headers='Classifiers Ratio'.split(), tablefmt='rst'))

table = [['KNeighborsClassifier/LogisticRegression', 
          decimals(knn_300.prediction_time/logistic_300.prediction_time)],
         ['KNeighborsClassifier/RandomForestClassifier',
          decimals(knn_300.prediction_time/forests_300.prediction_time)],
         ['RandomForestClassifier/LogisticRegression',
          decimals(forests_300.prediction_time/logistic_300.prediction_time)]]
print(tabulate(table, headers='Classifiers Ratio'.split(), tablefmt='rst'))

table = [['LogisticRegression/KNeighborsClassifier', 
          decimals(logistic_300.f1_test/knn_300.f1_test)],
         ['KNeighborsClassifier/RandomForestClassifier',
          decimals(knn_300.f1_test/forests_300.f1_test)],
         ['LogisticRegression/RandomForestClassifier',
          decimals(logistic_300.f1_test/forests_300.f1_test)]]
print(tabulate(table, headers='Classifiers Ratio'.split(), tablefmt='rst'))

def plot_scores(which_f1='test'):
    sizes = sorted(containers['LogisticRegression'].keys())
    figure = plot.figure()
    axe = figure.gca()
    color_map = {'LogisticRegression': 'b',
                 'KNeighborsClassifier': 'r',
                 'RandomForestClassifier': 'm'}
    for model in containers:
        scores = [getattr(containers[model][size], 'f1_{0}'.format(which_f1)) for size in sizes]
        axe.plot(sizes, scores, label=model, color=color_map[model])
    axe.legend(loc='lower right')
    axe.set_title("{0} Set F1 Scores by Training-Set Size".format(which_f1.capitalize()))
    axe.set_xlabel('Training Set Size')
    axe.set_ylabel('F1 Score')
    axe.set_ylim([0, 1.0])
    filename = 'f1_scores_{0}'.format(which_f1)
    print_image_directive(filename, figure)
    return

plot_scores('training')

import pudb; pudb.set_trace()
plot_scores()

def plot_test_train(model):
    sizes = sorted(containers['LogisticRegression'].keys())
    figure = plot.figure()
    axe = figure.gca()
    color_map = {'LogisticRegression': 'b',
                 'KNeighborsClassifier': 'r',
                 'RandomForestClassifier': 'm'}
    test_scores = [containers[model][size].f1_test for size in sizes]
    train_scores = [containers[model][size].f1_training for size in sizes]
    axe.plot(sizes, test_scores, label="Test", color=color_map[model])
    axe.plot(sizes, train_scores, '--', label="Train", color=color_map[model])
    axe.legend(loc='lower right')
    axe.set_title("{0} F1 Scores by Training-Set Size".format(model))
    axe.set_xlabel('Training Set Size')
    axe.set_ylabel('F1 Score')
    axe.set_ylim([0, 1.0])
    filename = 'f1_scores_{0}'.format(model)
    print_image_directive(filename, figure)
    return

plot_test_train('LogisticRegression')

plot_test_train('KNeighborsClassifier')

plot_test_train('RandomForestClassifier')