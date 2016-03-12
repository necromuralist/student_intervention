
Project 2: Supervised Learning
==============================

Building a Student Intervention System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Note to self:* this is the source of the data:
https://archive.ics.uci.edu/ml/datasets/Student+Performance.

1. Classification vs Regression
-------------------------------

Your goal is to identify students who might need early intervention -
which type of supervised machine learning problem is this,
classification or regression? Why?

Identifying students who might need early intervention is a
classification problem as you are sorting students into classes (*needs
intervention*, *doesn't need intervention*) rather than trying to
predict a quantitative value.

2. Exploring the Data
---------------------

Let's go ahead and read in the student dataset first.

*To execute a code cell, click inside it and press **Shift+Enter**.*

.. code:: python

    # Import libraries
    import numpy
    import pandas as pd
    
    # my imports
    import matplotlib.pyplot as plot
    import seaborn
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import f1_score, make_scorer

.. code:: python

    %matplotlib inline

.. code:: python

    # Read student data
    student_data = pd.read_csv("student-data.csv")
    print("Student data read successfully!")
    # Note: The last column 'passed' is the target/label, all other are feature columns


.. parsed-literal::

    Student data read successfully!


Now, can you find out the following facts about the dataset? - Total
number of students - Number of students who passed - Number of students
who failed - Graduation rate of the class (%) - Number of features

*Use the code block below to compute these values. Instructions/steps
are marked using **TODO**\ s.*

.. code:: python

    n_students = student_data.shape[0]
    n_features = student_data.shape[1]
    n_passed = sum(student_data.passed.map({'no': 0, 'yes': 1}))
    n_failed = n_students - n_passed
    grad_rate = n_passed/float(n_students)
    print "Total number of students: {}".format(n_students)
    print "Number of students who passed: {}".format(n_passed)
    print "Number of students who failed: {}".format(n_failed)
    print "Number of features: {}".format(n_features)
    print "Graduation rate of the class: {:.2f}%".format(grad_rate)



.. parsed-literal::

    Total number of students: 395
    Number of students who passed: 265
    Number of students who failed: 130
    Number of features: 31
    Graduation rate of the class: 0.67%


.. code:: python

    types = student_data.dtypes
    categoricals = [column for column in types.index if types.loc[column] == object]
    numericals = [column for column in types.index if column not in categoricals]

.. code:: python

    print("Categorical Variables: {0}".format(len(categoricals)))
    print("Numeric Variables: {0}".format(len(numericals)))


.. parsed-literal::

    Categorical Variables: 18
    Numeric Variables: 13


.. code:: python

    for categorical in categoricals:
        print('{0}\t{1}'.format(categorical, ','.join(student_data[categorical].unique())))


.. parsed-literal::

    school	GP,MS
    sex	F,M
    address	U,R
    famsize	GT3,LE3
    Pstatus	A,T
    Mjob	at_home,health,other,services,teacher
    Fjob	teacher,other,services,health,at_home
    reason	course,other,home,reputation
    guardian	mother,father,other
    schoolsup	yes,no
    famsup	no,yes
    paid	no,yes
    activities	no,yes
    nursery	yes,no
    higher	yes,no
    internet	no,yes
    romantic	no,yes
    passed	no,yes


.. code:: python

    categorical_data = student_data[categoricals]
    for categorical in categoricals:
        grid = seaborn.FacetGrid(categorical_data, col='passed')
        grid = grid.map(seaborn.countplot, categorical)
        grid.fig.suptitle('passed vs {0}'.format(categorical))




.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f462aea50>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f463e80d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46539b90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f468f7a90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46796bd0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46a30050>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46a41190>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46b5d350>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46c63450>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46d9ced0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f46f23d90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f470cc350>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f471e0290>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f472daa50>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f4777c150>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f47556390>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f476feb10>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f47a50810>


Surprisingly, Females were less likely to pass than males.
``family size`` seems to influence passing, as does parental
cohabitation, whether parents worked jobs other than services, health,
teacher, or at home, reason for taking the course, whether they were
paid, whether they had internet access at home.

.. code:: python

    figure = plot.figure(figsize=(10,8))
    axe = figure.gca()
    axe.set_title('numeric variables')
    lines = seaborn.boxplot(x=student_data[numericals], ax=axe)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f47a50350>


.. parsed-literal::

    /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/seaborn/categorical.py:2125: UserWarning: The boxplot API has been changed. Attempting to adjust your arguments for the new API (which might not work). Please update your code. See the version 0.6 release notes for more info.
      warnings.warn(msg, UserWarning)


.. code:: python

    numerical_data = student_data[numericals]
    figure = plot .figure(figsize=(10,8))
    axe = figure.gca()
    axe = numerical_data.plot(kind='kde', ax=axe)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f45aff3d0>


.. code:: python

    from pandas.tools.plotting import parallel_coordinates
    numerical_data['passed'] = student_data['passed']
    figure = plot.figure(figsize=(10,10))
    axe = figure.gca()
    subplot = parallel_coordinates(numerical_data, 'passed', ax=axe)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f45bfd6d0>


.. parsed-literal::

    /home/charon/.virtualenvs/machinelearning/lib/python2.7/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app


.. code:: python

    passed = student_data.passed.value_counts()/student_data.shape[0]
    print(passed)


.. parsed-literal::

    yes    0.670886
    no     0.329114
    Name: passed, dtype: float64


.. code:: python

    grid = seaborn.FacetGrid(student_data, col='passed', size=8)
    grid = grid.map_dataframe(lambda data, color: seaborn.heatmap(data.corr(), linewidths=0))




.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f47a9afd0>


.. parsed-literal::

    /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):


The highest corellations appear to be Dalc (workday alcohol consumption)
and Walc (weekend alcohol consumption), along with Medu (mother's
education) and Fedu (father's education).

.. code:: python

    figure = plot.figure(figsize=(10,8))
    axe = figure.gca()
    axe.set_ylabel('proportion')
    axe.set_title("Count of Passing Students")
    grid = seaborn.countplot(student_data.passed, ax=axe)




.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1f477bf350>


3. Preparing the Data
---------------------

In this section, we will prepare the data for modeling, training and
testing.

Identify feature and target columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is often the case that the data you obtain contains non-numeric
features. This can be a problem, as most machine learning algorithms
expect numeric data to perform computations with.

Let's first separate our data into feature and target columns, and see
if any features are non-numeric. **Note**: For this dataset, the last
column (``'passed'``) is the target or label we are trying to predict.

.. code:: python

    # Extract feature (X) and target (y) columns
    feature_cols = list(student_data.columns[:-1])  # all columns but last are features
    target_col = student_data.columns[-1]  # last column is the target/label
    print "Feature column(s):-\n{}".format(feature_cols)
    print "Target column: {}".format(target_col)
    
    X_all = student_data[feature_cols]  # feature values for all students
    y_all = student_data[target_col]  # corresponding targets/labels
    print "\nFeature values:-"
    print X_all.head()  # print the first 5 rows


.. parsed-literal::

    Feature column(s):-
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    Target column: passed
    
    Feature values:-
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   
    
        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   
    
      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  
    
    [5 rows x 30 columns]


Preprocess feature columns
~~~~~~~~~~~~~~~~~~~~~~~~~~

As you can see, there are several non-numeric columns that need to be
converted! Many of them are simply ``yes``/``no``, e.g. ``internet``.
These can be reasonably converted into ``1``/``0`` (binary) values.

Other columns, like ``Mjob`` and ``Fjob``, have more than two values,
and are known as *categorical variables*. The recommended way to handle
such a column is to create as many columns as possible values (e.g.
``Fjob_teacher``, ``Fjob_other``, ``Fjob_services``, etc.), and assign a
``1`` to one of them and ``0`` to all others.

These generated columns are sometimes called *dummy variables*, and we
will use the
```pandas.get_dummies()`` <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies>`__
function to perform this transformation.

.. code:: python

    # Preprocess feature columns
    def preprocess_features(X):
        outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    
        # Check each column
        for col, col_data in X.iteritems():
            # If data type is non-numeric, try to replace all yes/no values with 1/0
            if col_data.dtype == object:
                col_data = col_data.replace(['yes', 'no'], [1, 0])
            # Note: This should change the data type for yes/no columns to int
    
            # If still non-numeric, convert to one or more dummy variables
            if col_data.dtype == object:
                col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
    
            outX = outX.join(col_data)  # collect column(s) in output dataframe
    
        return outX
    
    X_all = preprocess_features(X_all)
    print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))



.. parsed-literal::

    Processed feature columns (48):-
    ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


Split data into training and test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, we have converted all *categorical* features into numeric
values. In this next step, we split the data (both features and
corresponding labels) into training and test sets.

.. code:: python

    # First, decide how many training vs test samples you want
    num_all = student_data.shape[0]  # same as len(student_data)
    num_train = 300  # about 75% of the data
    num_test = num_all - num_train
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=num_test,
                                                        train_size=num_train)
    
    
    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])
    # Note: If you need a validation set, extract it from within training data


.. parsed-literal::

    Training set: 300 samples
    Test set: 95 samples


4. Training and Evaluating Models
---------------------------------

Choose 3 supervised learning models that are available in scikit-learn,
and appropriate for this problem. For each model:

-  What are the general applications of this model? What are its
   strengths and weaknesses?
-  Given what you know about the data so far, why did you choose this
   model to apply?
-  Fit this model to the training data, try to predict labels (for both
   training and test sets), and measure the F1 score. Repeat this
   process with different training set sizes (100, 200, 300), keeping
   test set constant.

Produce a table showing training time, prediction time, F1 score on
training set and F1 score on test set, for each training set size.

Note: You need to produce 3 such tables - one for each model.

LogisticRegression
~~~~~~~~~~~~~~~~~~

The first supervised learning model that I've chosen is
``Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>``\ \_.
Logistic Regression uses numeric data to predict binary categorical
values, matching our inputs (after transformation) and outputs here. It
is a linear classification model and so does best when the data is
linearly separable, although it can be made to work as long as the
features are pairwise-separable (Alpaydin, 2010). Logistic Regression
has the advantage of being computationally cheap, reasonable to
implement, and is interpretable but has the disadvantage that it is
prone to underfitting (Harrington, 2012).

Logistic Regression uses the log-likelihood of the model to decide how
good it is and tries to improve it by choosing weights that maximize the
log-likelihood (Witten & Eibe, 2005). Logistic Regression calculates the
probability that a target-feature is 1 using the
``logistic (sigmoid) function`` (Alpaydin, 2010).

.. code:: python

    %%latex
    P(y=1|x) = sigmoid(W^Tx + w_0)
    = \frac{1}{1 + e^{-(W^Tx + w_0)}}



.. parsed-literal::

    <IPython.core.display.Latex object>


The sklearn implementation also supports regularization and thus can be
used for feature selection.

Random Forests
~~~~~~~~~~~~~~

The second learning model that I will use will be
``Random Forests<http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>``\ \_.
This is an ensemble learner that combines predictions from multiple
decision trees, each trained on a separate data set.

Decision Trees have several advantages, including the fact that they are
easily interpretable, can sometimes fit complex data more easily than
linear models, and don't require dummy variable. They are, however,
generally not as accurate (James G. et al., 2013).

The idea behind using ensemble learners is that any particular model has
a bias built into it based on its assumptions - when the assumptions are
wrong it will perform poorly. You can improve performance by combining
base-learners each of which has a different bias so that (ideally) no
instance of the data will cause a majority of the learners to perform
poorly, even if each performs poorly in some instances. For combining of
models to work, there has to be enough diversity that they don't all
fail on the same data (Alpaydin 2010).

The first way to introduce diversity is through *bagging (boostrap
aggregation)* where each tree (base-learner) is given a data set that is
constructed by re-sampling (with replacement) from the training-data.

The next way that diversity is introduced is by using a random samples
of features whenever a split is made, rather than choosing the best
split from all the features (the number of features used is near the
square-root of the number of total features). By keeping the number of
features small it reduces the likelihood that more influential features
will dominate the splitting early on, causing the trees to be too
similar (Gareth G. et al., 2013). This use of sub-sets of features in
splitting is what makes it a random-forest (rather than just bagged
trees).

Predictions are made by having each tree make a prediction and then the
average of the predictions is used for the final prediction for the
entire forest. Using these methods improves the performance over using
an individual tree, but the ensemble is no longer interpretable the way
a tree would be.

K-Nearest Neighbors
~~~~~~~~~~~~~~~~~~~

My final predictor will use *K-Nearest Neighbors (KNN)* classification.
It is a fairly straight-forward method that doesn't build a model in the
sense that the other two methods do. Instead KNN stores the training
data and when asked to make a prediction finds the *k* number of points
in the training data that are 'closest' to the input and calculates the
probability of a classification (say *y=1*) as the fraction of the
k-neighbors that are of that class. Thus if k=4 and three of the chosen
neighbors are classified as *1* then the predicted class will be *1*,
because the majority of the neighbors were 1.

Unlike Logistic Regression, KNN doesn't require linear separability and
unlike some other methods also makes no assumption about the
distribution of the data (it is *non-parametric*). This makes it better
in some cases, but how accurate it is depends on the choice of *k*. If
*k* is too small it will tend to overfit the training data and if *k* is
too large, it will become too rigid. Besides the difficulty in choosing
*k*, because it is non-parametric it's not possible to inspect the model
to decide which features are important. Additionally, since it's
non-parametric, KNN needs more data to be accurate.

.. code:: python

    # Train a model
    import time
    
    def train_classifier(clf, X_train, y_train):
        print "Training {}...".format(clf.__class__.__name__)
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        print "Done!\nTraining time (secs): {:.3f}".format(end - start)
    
    # TODO: Choose a model, import it and instantiate an object
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    
    # Fit model to training data
    train_classifier(clf, X_train, y_train)  # note: using entire training set here
    print clf  # you can inspect the learned model by printing it


.. parsed-literal::

    Training DecisionTreeClassifier...
    Done!
    Training time (secs): 0.004
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')


.. code:: python

    # Predict on training set and compute F1 score
    from sklearn.metrics import f1_score
    
    def predict_labels(clf, features, target):
        print "Predicting labels using {}...".format(clf.__class__.__name__)
        start = time.time()
        y_pred = clf.predict(features)
        end = time.time()
        print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
        return f1_score(target.values, y_pred, pos_label='yes')
    
    train_f1_score = predict_labels(clf, X_train, y_train)
    print "F1 score for training set: {}".format(train_f1_score)


.. parsed-literal::

    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for training set: 1.0


.. code:: python

    # Predict on test data
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))


.. parsed-literal::

    Predicting labels using DecisionTreeClassifier...
    Done!
    Prediction time (secs): 0.001
    F1 score for test set: 0.588235294118


.. code:: python

    # Train and predict using different training set sizes
    def train_predict(clf, X_train, y_train, X_test, y_test):
        print "------------------------------------------"
        print "Training set size: {}".format(len(X_train))
        train_classifier(clf, X_train, y_train)
        print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
        print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))


.. code:: python

    class Classifier(object):
        """
        Trains, predicts, evaluates classifier using f1 score
        """
        def __init__(self, classifier, x_train, y_train, x_test, y_test, delim='\t'):
            """
            :param:
             - `classifier`: sklearn classifier object
             - `x_train`: feature training data
             - `y_train`: target training data
             - `x_test`: feature test data
             - `y_test`: target test data
             - `delim`: separator for the table row
            """
            self.clf = classifier
            self._classifier = None
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
            self._f1_train = None
            self._f1_test = None
            self.delim = delim
            self._table_row = None
            self._training_time = None
            self._prediction_time = None
            return
    
        @property
        def f1_train(self):
            """
            :return: F1 score using training data
            """
            if self._f1_train is None:
                predictions, time_ = self.predict(self.x_train)
                self._f1_train = self.f1_score(predictions, self.y_train)
            return self._f1_train
    
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
                start = time.time()
                self._classifier = self.clf.fit(self.x_train, self.y_train)
                self._training_time = time.time() - start
            return self._training_time
            
        @property
        def classifier(self):
            """
            :return: trained classifier
            """
            if self._classifier is None:
                start = time.time()
                self._classifier = self.clf.fit(self.x_train, self.y_train)
                self._training_time = time.time() - start
            return self._classifier
    
        def f1_score(self, predictions, target):
            """
            :param:
             - `predictions`: predicted values for model
             - `target`: actual outcomes from data
            :return: f1 score for predictions
            """
            return f1_score(target.values, predictions, pos_label='yes')
    
        def predict(self, features):
            """
            :param:
             - `features`: array of feature data
            :return: predicted values, time to execute
            """
            start = time.time()
            predictions = self.classifier.predict(features)
            elapsed = time.time() - start
            return predictions, elapsed
    
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
                                                                                       self.f1_train,
                                                                                       self.f1_test)])
            return self._table_row

.. code:: python

    def train_and_predict(clf):
        scores = []
        for size in range(100, 400, 100):
            x_train_subset, y_train_subset = X_train[:size], y_train[:size]
            classifier = Classifier(clf, x_train_subset, y_train_subset,
                                    X_test, y_test, delim='\t\t')
            # train_time, train_score, test_time, test_score = classifier.train_and_predict()
            # print('\t\t\t'.join([str(size)] + ['{0:.2f}'.format(item) for item in (classifier.training_time,
            #                                                                        train_score,
            #                                                                        test_time,
            #                                                                        test_score)]))
            print(classifier.table_row)
            scores.append((classifier.f1_test, size))
        return max(scores)    

.. code:: python

    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    
    classifiers = [LogisticRegression(), tree.DecisionTreeClassifier(), RandomForestClassifier(),
                   svm.SVC(), GaussianNB(), SGDClassifier(), AdaBoostClassifier(),
                   GradientBoostingClassifier(), KNeighborsClassifier()]
    best_scores = []
    line_width = 80
    for classifier in classifiers:
        print('')
        print(classifier.__class__.__name__)
        print("=" * line_width)
        print("Size\t\tTime(t)\t\tTime(p)\t\tTrain F1\tTest F1")
        print('-' * line_width)
        best_score, best_size = train_and_predict(classifier)
        print("-" * line_width)
        print("best score: {0:.2f}, best_size: {1}".format(best_score, best_size))
        best_scores.append((best_score,classifier.__class__.__name__, best_size))
    print("=" * line_width)
    print('')
    print("Ranked by Score")
    print('~' * line_width)
    from tabulate import tabulate
    table = [[score[1], score[0], score[-1]] for index,score in enumerate(sorted(best_scores, reverse=True))]
    print(tabulate(table, headers='Classifier score training-size'.split()))


.. parsed-literal::

    
    --------------------------------------------------------------------------------
    best score: 0.80, best_size: 200
    
    KNeighborsClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0008		0.0016		0.8176		0.8054
    200		0.0009		0.0022		0.8664		0.8082
    300		0.0009		0.0028		0.8604		0.8243
    --------------------------------------------------------------------------------
    best score: 0.82, best_size: 300
    ================================================================================
    
    Ranked by Score
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Classifier                     score    training-size
    --------------------------  --------  ---------------
    KNeighborsClassifier        0.824324              300
    LogisticRegression          0.814286              200
    SVC                         0.805195              300
    SGDClassifier               0.805031              300
    GradientBoostingClassifier  0.8                   200
    RandomForestClassifier      0.787879              300
    AdaBoostClassifier          0.787879              100
    GaussianNB                  0.779412              200
    DecisionTreeClassifier      0.761905              100
    
    300		0.1404		0.0006		0.9663		0.7538
    200		0.1142		0.0006		0.9858		0.8000
    --------------------------------------------------------------------------------
    best score: 0.79, best_size: 100
    
    GradientBoostingClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0823		0.0006		1.0000		0.7704
    300		0.1106		0.0060		0.8505		0.7852
    200		0.1708		0.0077		0.9034		0.7801
    200		0.0009		0.0002		0.4286		0.4138
    300		0.0010		0.0002		0.8024		0.8050
    --------------------------------------------------------------------------------
    best score: 0.81, best_size: 300
    
    AdaBoostClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0952		0.0061		0.9859		0.7879
    --------------------------------------------------------------------------------
    best score: 0.79, best_size: 300
    
    SVC
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0019		0.0011		0.8485		0.7975
    200		0.0046		0.0017		0.8598		0.8052
    300		0.0092		0.0023		0.8529		0.8052
    --------------------------------------------------------------------------------
    best score: 0.81, best_size: 300
    
    GaussianNB
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0008		0.0004		0.5053		0.3373
    200		0.0009		0.0004		0.8333		0.7794
    300		0.0009		0.0003		0.8009		0.7786
    --------------------------------------------------------------------------------
    best score: 0.78, best_size: 200
    
    SGDClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0006		0.0002		0.7287		0.7414
    200		0.0250		0.0014		0.9929		0.7681
    300		0.0233		0.0013		0.9975		0.7879
    LogisticRegression
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0023		0.0003		0.8725		0.8000
    200		0.0039		0.0004		0.8366		0.8143
    300		0.0046		0.0003		0.8273		0.7971
    --------------------------------------------------------------------------------
    best score: 0.81, best_size: 200
    
    DecisionTreeClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0013		0.0003		1.0000		0.7619
    200		0.0022		0.0003		1.0000		0.6504
    300		0.0029		0.0003		1.0000		0.6218
    --------------------------------------------------------------------------------
    best score: 0.76, best_size: 100
    
    RandomForestClassifier
    ================================================================================
    Size		Time(t)		Time(p)		Train F1	Test F1
    --------------------------------------------------------------------------------
    100		0.0227		0.0013		1.0000		0.7737

5. Choosing the Best Model
--------------------------

-  Based on the experiments you performed earlier, in 1-2 paragraphs
   explain to the board of supervisors what single model you chose as
   the best model. Which model is generally the most appropriate based
   on the available data, limited resources, cost, and performance?
-  In 1-2 paragraphs explain to the board of supervisors in layman's
   terms how the final model chosen is supposed to work (for example if
   you chose a Decision Tree or Support Vector Machine, how does it make
   a prediction).
-  Fine-tune the model. Use Gridsearch with at least one important
   parameter tuned and with at least 3 settings. Use the entire training
   set for this.
-  What is the model's final F1 score?

.. code:: python

    y_train_numeric = y_train.replace('yes no'.split(), [1, 0])
    y_test_numeric = y_test.replace('yes no'.split(), [1, 0])

.. code:: python

    class LRClassifier(object):
        """
        holds the LogisticRegression classifier
        """
        def __init__(self, c_range, score_function=f1_score, n_jobs=-1, folds=10,
                     training_features=X_train, training_targets=y_train_numeric,
                     test_features=X_test, test_targets=y_test_numeric):
            """
            :param:
             - `c_range`: range of 'C' values for grid search
             - `score_function`: function to maximize
             - `n_jobs`: number of parallel jobs for the grid search
             - `folds`: number of cross validation folds to use
             - `training_features`: array of training feature-data
             - `training_targets`: array of training target-values
             - `test_features`: array of testing feature-data
             - `test_targets`: array of testing target-data
            """
            self.c_range = c_range
            self.n_jobs = n_jobs
            self.folds = folds
            self.score_function = score_function
            self.training_features = training_features
            self.training_targets = training_targets
            self.test_features = test_features
            self.test_targets = test_targets
            
            self._scorer = None
            self._model = None
            self._grid = None
            self._parameters = None
            return
    
        @property
        def parameters(self):
            """
            :return: dict of grid search parameters
            """
            if self._parameters is None:
                self._parameters = {'penalty': ('l1', 'l2'),
                                    'C': self.c_range}
            return self._parameters
        
        @property
        def scorer(self):
            """
            :return: scorer for the grid search
            """
            if self._scorer is None:
                self._scorer = make_scorer(self.score_function)
            return self._scorer
    
        @property
        def model(self):
            """
            :return: LogisticRegression object
            """
            if self._model is None:
                self._model = LogisticRegression()
            return self._model
    
        @property
        def grid(self):
            """
            :return: GridSearchCV object with best model
            """
            if self._grid is None:
                self._grid = GridSearchCV(self.model,
                                          param_grid=self.parameters,
                                          scoring=self.scorer,
                                          cv=self.folds,
                                          n_jobs=self.n_jobs)
                self._grid.fit(self.training_features, self.training_targets)
            return self._grid
    
        def print_columns(self):
            """
            prints non-zero coefficients in descending order
            """
            coefficients = self.grid.best_estimator_.coef_[0]
            sorted_coefficients = sorted((column for column in coefficients), reverse=True)
            for coefficient in sorted_coefficients:
                if abs(coefficient) > 0:
                    index = numpy.where(coefficients == coefficient)[0][0]
                    print(X_test.columns[index], coefficient)
            return
    
        def print_best(self):
            print('Parameters')
            print(self.grid.best_params_)
            print('\nF1 Score')
            print(self.grid.score(self.test_features, self.test_targets))
            print('\ncoefficients')
            self.print_columns()

.. code:: python

    grid_01 = LRClassifier(numpy.arange(.01, 1.1, .01))
    grid_01.print_best()



.. parsed-literal::

    {'penalty': 'l1', 'C': 0.080000000000000002}
    
    F1 Score
    0.797297297297
    
    coefficients
    ('age', 0.04631275424685484)
    ('Medu', 0.04122577352087383)
    ('famrel', 0.02989095441447533)
    ('absences', -0.020337195557860503)
    ('failures', -0.6111689390192977)
    Parameters

