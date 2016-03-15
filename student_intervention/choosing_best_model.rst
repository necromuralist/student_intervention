5. Choosing the Best Model
--------------------------





Based on the previous experiments I chose *Logistic Regression* as the classifier to use. Given the data available, all three models have approximately the same F1 score (on the test data) but the Logistic Regression classifier is the fastest for both training and prediction when compared to *K-Nearest Neighbor* and *Random Forests*. In addition, the Logistic Regression classifier offers readily interpretable coefficients and L1 regression to sparsify the data, allowing us to see the most important of the variables when deciding who will pass their final exam.

Logistic Regression works by estimating the probability that the target feature is 1 given the input features. It does this using the 'sigmoid' function which creates an S-shaped curve which goes to 0 at negative infinity and 1 at positive infinity:

.. math::

   P(y=1|x) = \frac{1}{1+e^{-w^Tx}}\\

Here *x* is a vector of feature data and *w* is the vector of weights that the Logistic Regression algorithm finds. The output of this function when there is one feature with a weight of 1 looks like this.


.. image:: figures/sigmoid_function.*
   :align: center
   :scale: 95%



When *x* is greater than 0 the probability is greater than 0.5 the output is classified as a 1, otherwise it is classified as 0.












Fitting The Model
~~~~~~~~~~~~~~~~~

The Logistic Regression model was fit using sklearn's `GridSearchCV` with 10 folds for the cross-validation. The parameters tested were the penalty-type ('l1' or 'l2') and 'C' the inverse of the regularization strength (the smaller the number the larger the penalty).

The *penalties* refer to regularization penalties that shrink or eliminate variable-coefficients in the model. The 'l1' penalty refers to *lasso regularization* which causes some of the feature-variable coefficients to go to 0 if they don't contribute as much to the model as the other, more significant variables. The 'l2' penalty refers to *ridge regression* which shrinks the coefficients but never pushes them all the way to 0. Since *lasso regularization*  simplifies the model it might seem that it would be preferable, but it will only outperform *ridge regression* if there are differences in how much the variables contribute. If all the variables contribute equally to the model, then *ridge regression* will outperform the *lasso* (James G. et al., 2013).

'C' is the inverse of the regularization strength. The stronger the regularization, the smaller the coefficients and for *lasso* the fewer the coefficients. Shrinking the coefficients too much or eliminating too many variables can weaken the model so how much regularization is determined here using cross-validation.

This is the outcome of fitting the Logistic Regression classifier using Grid Search with the parameters given.






Parameters
``````````

===========  =======
Parameter    Value
===========  =======
penalty      l1
C            0.09
===========  =======



These are the parameters for the model that had the highest F1 score. The best scorer used 'l1', or lasso regularization so the model should be sparse. The amount of regularization used was fairly high (since 'C' is the inverse of the amount of regularization, the smaller it is the more regularization there is). 



Coefficients
````````````

==========  ===============================  =============  ======
Variable    Description                        Coefficient    Odds
==========  ===============================  =============  ======
Medu        mother's education                        0.14    1.16
age         student's age                             0.03    1.03
famrel      quality of family relationships           0.03    1.03
Fedu        father's education                        0.01    1.01
absences    number of school absences                -0.02    0.98
goout       going out with friends                   -0.07    0.93
failures    number of past class failures            -0.46    0.63
==========  ===============================  =============  ======



These are the variables that remained in the best model after the regularization was applied, sorted by their coefficient-values. The coefficients are log-odds so calculating :math:`e^{coefficient}` gives you the increase in odds that the student will graduate for every unit-increase for that variable.

Positive Contributions
++++++++++++++++++++++

Medu
####

The predictor with the greatest positive effect in the model was the amount of education the student's mother received. According to the read-me file there are 5 levels.

.. '

.. csv-table:: Mother's Education
   :header: Level, Meaning

   0,None
   1,4th Grade
   2,5th to 9th Grade
   3,Secondary Education
   4,Higher Education

.. '





.. image:: figures/mothers_education.*
   :align: center
   :scale: 95%



Looking at a plot of all the data, as the mother's education-level goes up the proportion of those that pass goes up relative to those that don't pass.

age
###

This is the age of the student. It wasn't immediately obvious why this would be a factor, assuming that the students are all in the same grade, but a plot of the data showed that the ages range from 15 to 22, with the oldest students not passing the final exam.

.. '

.. image:: figures/student_age.*
   :align: center
   :scale: 95%



famrel
######

According to the readme file ``famrel`` is a rating of the quality of family relations ranging form 1 (very bad) to 5 (excellent).


.. image:: figures/family_relations.*
   :align: center
   :scale: 95%



The plot seems to show that relations of 4 and 5 improve the likelihood of passing.

Fedu
####

This is the student's father's education level. The levels are the same as `Medu`.


.. image:: figures/fathers_education.*
   :align: center
   :scale: 95%



.. note:: Weekend alcohol consumption is included but has a coefficient of 0 so it is neither a negative nor positive contributor to passing.

Negative Contributions
++++++++++++++++++++++

These are variables that decrease the odds of a student passing as their values increase.

absences
########

This is a straightforward count of the number of absences there were.


.. image:: figures/absences.*
   :align: center
   :scale: 95%



The differences are not large, but at around 10 absences the distributions seem to cross over with the non-passing line raised above the passing line.

goout
#####

This is how often the student goes out with friends. There are 5 levels from 1 (very low) to 5 (very high).


.. image:: figures/going_out.*
   :align: center
   :scale: 95%



It looks like a level greater than 3 has a negative effect on passing.

failures
########

This is the most negative variable and represents the number of past class failures for a student. It ranges from 0 to 4 with 4 meaning 4 or more failures.


.. image:: figures/past_failures.*
   :align: center
   :scale: 95%



It looks like no student failed 4 or more classes before taking the final exam and 1 or more failures had a negative impact on whether a student passed.


F1 score
````````

0.86


The best F1 score for the Logistic Regression classifier was 0.83, which is a slight improvement over the default Logistic Regression classifier used earlier which had an f1 of approximately 0.79 for the test set when trained with 300 training instances.