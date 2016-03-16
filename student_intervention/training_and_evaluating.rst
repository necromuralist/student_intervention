Training and Evaluating Models
------------------------------








Logistic Regression
~~~~~~~~~~~~~~~~~~~

The first model I chose was `Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_. Logistic regression is a linear classification model that is useful when the target variable is categorical and the feature variables are either numeric or categorical (Peng C., et al, 2002), although the categorical variables have to be converted to numeric values prior to use. Logistic Regression has the advantage of being computationally cheap, reasonable to implement, and is interpretable but has the disadvantage that it is prone to underfitting (Harrington, 2012).

I chose this model for thre primary reasons:

  * Its coefficients are interpretable so besides predicting whether a student is at risk, factors that are most influential in determining students who are at risk can be identified
  * It suports ridge regression, including lasso regression which might help reduce the number of variables in the data set, both to aid interpretation and improve performance
  * It is well understood/well studied

Since it is a linear model it performs best when the data is linearly separable, which makes it a complement to Random Forests, the next model I chose.
  
Random Forests
~~~~~~~~~~~~~~

`Random Forests <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_ are ensemble learners that combine predictions from multiple decision trees.

Decision Trees have several advantages including:

   * they are interpretable
   * they can sometimes fit complex data more easily than linear models
   * they don't require dummy variables. 

.. '
   
They are, however, generally not as accurate (James G. et al., 2013). Random Forests overcome the failings of the individual Decision Trees using two methods:

 *  training each tree on a separate data set that is created by re-sampling with replacement from the original training set ( *bagging*) which improves the accuracy of the forest over that of an individual tree. 
 *  each tree in the forest uses a random sub-set of the features when deciding on a split so that there is sufficient diversity in the forest to improve reliability 

The predicted output for the features is created by taking an average of the predictions given by the trees.

Random forests, like decision trees, can be used for either classification or regression, but the trade-off for their improved accuracy is that the ensemble is not as directly interpretable as the individual decision trees are. Decision Trees, and thus Random Forests, don't assume linear separability and can perform better than linear models when the data isn't linearly separable, but may not do as well in the cases where the linear model is appropriate (James G. et al., 2013), thus, given the number of variables I thought that it might be more effective on this data set in the event that Logistic Regression cannot model the data well.

K-Nearest Neighbors
~~~~~~~~~~~~~~~~~~~

My final predictor used `K-Nearest Neighbors (KNN) <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_ classification. It is a fairly straight-forward method that doesn't build a model in the sense that the other two methods do. Instead KNN stores the training data and when asked to make a prediction finds the *k* number of points in the training data that are 'closest' to the input and calculates then
probability of a classification (say *y=1*) as the fraction of the k-neighbors that are of that class. Thus if k=4 and three of the chosen neighbors are classified as *1* then the predicted class will be *1*, because the majority of the neighbors were 1.

Because *KNN* doesn't build a model the way the other two classifiers do, it has the quickest training time but trades this for the longest prediction times.

Unlike Logistic Regression, KNN doesn't require linear separability and unlike some other methods also makes no assumption about the distribution of the data (it is *non-parametric*). This makes KNN more flexible, but how accurate it is depends on the choice of *k*. If *k* is too small it will tend to overfit the training data and if *k* is too large, it will become too rigid. Besides the difficulty in choosing *k*, because it is non-parametric it's not possible to inspect the model to decide which features are important and it needs more data to be accurate (James G., et al., 2013).

I thought that KNN might be a good non-parametric alternative to Logistic Regression since the data comes from students attending two schools in Portugal which might make the instances more similar than dissimilar, the 'nearest neighbor' method was conceptually appropriate and it is different enough in approach that it might improve on the other two methods should the separation of the classes be unusually difficult.

.. '

Performance Comparisons
~~~~~~~~~~~~~~~~~~~~~~~

To compare the models each was fit using their default parameters on the same sub-sets of the data. The sub-sets were made of the first 100, 200, and 300 observations of the data. Times are in seconds and the 'best' scores are based on their F1 scores, the weighted average of their prediction and recall scores.

Since the running times are based on my machine's performance as much as that of the classifiers' all times are the minimum value from 100 repetitions (following the advice in the python `timeit <https://docs.python.org/2/library/timeit.html>`_ documentation). The f1-scores are the median values for the 100 repetitions.










.. csv-table:: LogisticRegression
   :header: Size,Time (train),Time (predict),Train F1,Test F1

   100,0.0011,0.0001,0.8702,0.6720
   200,0.0025,0.0001,0.8333,0.7910
   300,0.0029,0.0001,0.8075,0.8120
Best score and size of data-set that gave the best test score:
 * best score: 0.81
 * best size: 300

.. csv-table:: RandomForestClassifier
   :header: Size,Time (train),Time (predict),Train F1,Test F1

   100,0.0198,0.0011,0.9841,0.6557
   200,0.0208,0.0011,0.9887,0.7087
   300,0.0218,0.0012,0.9949,0.7714
Best score and size of data-set that gave the best test score:
 * best score: 0.77
 * best size: 300

.. csv-table:: KNeighborsClassifier
   :header: Size,Time (train),Time (predict),Train F1,Test F1

   100,0.0004,0.0013,0.8244,0.7519
   200,0.0005,0.0019,0.8099,0.7536
   300,0.0006,0.0027,0.8491,0.7945
Best score and size of data-set that gave the best test score:
 * best score: 0.79
 * best size: 300



Summation
+++++++++


======================  =======  ===============  ===============  =================
Classifier                Score    Training-Size    Training-Time    Prediction-Time
======================  =======  ===============  ===============  =================
LogisticRegression         0.81              300           0.0029             0.0001
KNeighborsClassifier       0.79              300           0.0006             0.0027
RandomForestClassifier     0.77              300           0.0218             0.0012
======================  =======  ===============  ===============  =================


It looks like all three did about equally well on the test-sets.

As expected, KNN had the shortest training time and the longest prediction time. Since the values are so small, I'll look at the ratios of the times next instead of the absolute times.

.. '



Training Times
``````````````

===========================================  =======
Classifiers                                    Ratio
===========================================  =======
LogisticRegression/KNeighborsClassifier         4.45
RandomForestClassifier/KNeighborsClassifier    33.75
RandomForestClassifier/LogisticRegression       7.59
===========================================  =======


The Random Forest classifier was 5-10 times slower than the Logistic Regression classifier, which was itself about 5 times slower than the KNN classifier when training the models.

Prediction Times
````````````````

===========================================  =======
Classifiers                                    Ratio
===========================================  =======
KNeighborsClassifier/LogisticRegression        20.23
KNeighborsClassifier/RandomForestClassifier     2.29
RandomForestClassifier/LogisticRegression       8.84
===========================================  =======


It looks like the Logistic Regression classifier was significantly faster than either the Random Forest classifier or the K-Nearest Neighbors classifier - about 20 times faster than KNN and 5-10 times faster than the Random Forest classifier.

F1 Prediction Scores (Test Set)
```````````````````````````````

===========================================  =======
Classifiers                                    Ratio
===========================================  =======
LogisticRegression/KNeighborsClassifier         1.02
KNeighborsClassifier/RandomForestClassifier     1.03
LogisticRegression/RandomForestClassifier       1.05
===========================================  =======


The three models seem to have been comparable once the training data reached 300 instances.

F1 Scores
~~~~~~~~~

Although I printed the tables for the F1 scores I will plot them here to take a closer look at them. The training-set sizes for the plots ranged from 10 to 300, increasing in steps of 10.




Training Set
++++++++++++


.. image:: figures/f1_scores_training.*
   :align: center
   :scale: 85%


The Random Forest did well on the training set from the start, while the K-nearest Neighbor classifier  and the Logistic Regression classifier were erratic until just over 100 training instances. Neither K-Nearest Neighbors nor Logistic Regression was able to do as well on the training set as Random Forests did, suggesting that they are underfitting the data.

Test Set
++++++++


.. image:: figures/f1_scores_test.*
   :align: center
   :scale: 85%


All three classifiers did comparably once the training set was increased to 300 instances, but the Random Forest Classifier shows larger fluctuations in the F1 score than the other classifiers, while the Logistic Regression classifier seemed to be the most stable, and performed the best for most of the instance-counts.




Comparing Test vs Train Scores by Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: figures/f1_scores_LogisticRegression.*
   :align: center
   :scale: 85%


The training and testing sets for the Logistic Regression seem to be converging around 0.8, suggesting the model is underfitting and may not be complex enough for the data. Oddly, the test score is better than  the training score after about 250 training instances. Looking at the table above, the differences are fractional and might just be that the larger training set has proportionally more difficult instances than the test-set.


.. image:: figures/f1_scores_KNeighborsClassifier.*
   :align: center
   :scale: 85%


The K-Nearest Neighbors classifier seems to perform comparably to the Logistic Regression classifier, although the two curves haven't converged yet, suggesting that it might be improved with more data, although it will still underfit the data.


.. image:: figures/f1_scores_RandomForestClassifier.*
   :align: center
   :scale: 85%


The Random Forest classifier doesn't do better with the test data than the other two classifiers but is much better with the training data, suggesting that it is currently overfitting, and might be improved with more data.
