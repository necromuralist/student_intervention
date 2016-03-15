

1. Classification vs Regression
-------------------------------

Your goal is to identify students who might need early intervention - which type of supervised machine learning problem is this,
classification or regression? Why?

Identifying students who might need early intervention is a *classification* problem as you are sorting students into classes (*needs intervention*, *doesn't need intervention*) rather than trying to predict a quantitative value.

.. '
                                                                                                                               
2. Exploring the Data
---------------------




Now, can you find out the following facts about the dataset? - Total number of students - Number of students who passed - Number of students who failed - Graduation rate of the class (%) - Number of features.

.. csv-table:: Summary Statistics
   :header: Static, Value

   Total number of students, 395
   Number of students who passed, 265
   Number of students who failed, 130
   Number of features, 31
   Graduation rate of the class, 67.09%



Passed
~~~~~~

The target variable is 'passed'.


.. image:: figures/passing_count.*
   :align: center
   :scale: 95%



As noted above, 67% of the students passed.
