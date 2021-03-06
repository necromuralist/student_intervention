
# third party
import matplotlib.pyplot as plot
import numpy
import pandas as pd
import seaborn

# this code
from common import print_image_directive, STYLE

from common import student_data

seaborn.set_style(STYLE)

n_students = student_data.shape[0]
n_features = student_data.shape[1] - 1
n_passed = sum(student_data.passed.map({'no': 0, 'yes': 1}))
n_failed = n_students - n_passed
grad_rate = n_passed/float(n_students)

assert n_students == student_data.passed.count()
assert n_features == len(student_data.columns[student_data.columns != 'passed'])
assert n_passed == len(student_data[student_data.passed == 'yes'].passed)
print("   Total number of students, {}".format(n_students))
print("   Number of students who passed, {}".format(n_passed))
print("   Number of students who failed, {}".format(n_failed))
print("   Number of features, {}".format(n_features))
print("   Graduation rate of the class, {:.2f}%".format(100 * grad_rate))

passing_rates = student_data.passed.value_counts()/student_data.passed.count()
filename = 'passing_count'
figure = plot.figure()
axe = figure.gca()
axe.set_ylabel('proportion')
axe.set_xlabel("Student Passed")
axe.set_title("Proportion of Passing Students")
grid = seaborn.barplot(x=passing_rates.index, y=passing_rates.values, ax=axe)
print_image_directive(filename, figure)