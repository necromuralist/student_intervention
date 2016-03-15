Project 2: Building a Student Intervention System
=================================================




A pdf version is available :download:`here <student_intervention.pdf>`
and the repository for the source of this document is `here
<https://github.com/necromuralist/student_intervention>`_.



This is Project Two from Udacity's Machine Learning Nanodegree program. It uses the `UCI Student Performance data-set <https://archive.ics.uci.edu/ml/datasets/Student+Performance>`_ to model the factors that predict the likelihood that a student will pass his or her final exam. The project begins with an exploration of the data to understand the feature and target variables, followed by the selection of three machine-learning algorithms to potentially model the student-performance. Finally one of the models is chosen and the optimal parameters chosen.

.. toctree::
   :maxdepth: 1
 
   Exploring the Data <exploring_data>
   Preparing the Data <preparing_data>
   Training and Evaluating Models <training_and_evaluating>
   Choosing the Best Model <choosing_best_model>
   Citations <citations>
..     Software References <software>
    
.. <<name='html_only', echo=False, results='sphinx', wrap=False>>=
.. HTML_ONLY = strtobool(os.environ.get("HTML_ONLY", 'on'))
.. if HTML_ONLY:
..    create_toctree()
.. @
