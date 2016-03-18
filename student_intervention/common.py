# python standard library
from collections import namedtuple
import os
from distutils.util import strtobool

# third party
import pandas

student_data = pandas.read_csv("student-data.csv")
RANDOM_STATE = 100
SCALE = os.environ.get("SCALE", '85%')
STYLE = 'whitegrid'

def print_image_directive(filename, figure, scale='95%', print_only=False):
    """
    saves and prints the rst image directive

    :param:

     - `filename`: filename to save the image (without 'figures/' or file extension)
     - `figure`: matplotlib figure to save the image
     - `scale: percent scale for the image
     - `print_only`: assume the figure exists, print directive only
    :postcondition: figure saved, rst image directive output
    """
    path = os.path.join('figures/', filename)
    if not print_only:
        figure.savefig(path + '.svg')
        figure.savefig(path + '.pdf')
    print(".. image:: {0}.*".format(path))
    print("   :align: center")
    print("   :scale: {0}".format(scale))

feature_map = {"school": "student's school",
               "sex": "student's sex",
               "age": "student's age",
               "address": "student's home address type",
               "famsize": "family size",
               "Pstatus": "parent's cohabitation status",
               "Medu": "mother's education",
               "Fedu": "father's education",
               "Mjob": "mother's job",
               "Fjob": "father's job",
               "reason": "reason to choose this school",
               "guardian": "student's guardian",
               "traveltime": "home to school travel time",
               "studytime": "weekly study time",
               "failures": "number of past class failures",
               "schoolsup": "extra educational support",
               "famsup": "family educational support",
               "paid": "extra paid classes within the course subject (Math or Portuguese)",
               "activities": "extra-curricular activities",
               "nursery": "attended nursery school",
               "higher": "wants to take higher education",
               "internet": "Internet access at home",
               "romantic": "within a romantic relationship",
               "famrel": "quality of family relationships",
               "freetime": "free time after school",
               "goout": "going out with friends",
               "Dalc": "workday alcohol consumption",
               "Walc": "weekend alcohol consumption",
               "health": "current health status",
               "absences": "number of school absences",
               "passed": "did the student pass the final exam"}
    
TrainTestData = namedtuple('TrainTestData', 'X_train X_test y_train y_test'.split())
TrainTestDataOne = namedtuple('TrainTestDataOne', 'X_train X_test y_train y_test'.split())
train_test_path = 'pickles/train_test_data.pkl'
