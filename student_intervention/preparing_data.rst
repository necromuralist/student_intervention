3. Preparing the Data
---------------------


In this section, we will prepare the data for modeling, training and testing.

Identify feature and target columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The target (as noted previously) is the 'passed' column. Here I'll list the feature columns to get an idea of what's there.



.. csv-table:: Features
   :header: Variable, Description, Data Values
   :delim: ;

   Dalc;workday alcohol consumption;1, 2, 3, 4, 5
   Fedu;father's education;0, 1, 2, 3, 4
   Fjob;father's job;at_home, health, other, services, teacher
   Medu;mother's education;0, 1, 2, 3, 4
   Mjob;mother's job;at_home, health, other, services, teacher
   Pstatus;parent's cohabitation status;A, T
   Walc;weekend alcohol consumption;1, 2, 3, 4, 5
   absences;number of school absences;0 ... 75
   activities;extra-curricular activities;no, yes
   address;student's home address type;R, U
   age;student's age;15, 16, 17, 18, 19, 20, 21, 22
   failures;number of past class failures;0, 1, 2, 3
   famrel;quality of family relationships;1, 2, 3, 4, 5
   famsize;family size;GT3, LE3
   famsup;family educational support;no, yes
   freetime;free time after school;1, 2, 3, 4, 5
   goout;going out with friends;1, 2, 3, 4, 5
   guardian;student's guardian;father, mother, other
   health;current health status;1, 2, 3, 4, 5
   higher;wants to take higher education;no, yes
   internet;Internet access at home;no, yes
   nursery;attended nursery school;no, yes
   paid;extra paid classes within the course subject (Math or Portuguese);no, yes
   reason;reason to choose this school;course, home, other, reputation
   romantic;within a romantic relationship;no, yes
   school;student's school;GP, MS
   schoolsup;extra educational support;no, yes
   sex;student's sex;F, M
   studytime;weekly study time;1, 2, 3, 4
   traveltime;home to school travel time;1, 2, 3, 4


Preprocess feature columns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some Machine Learning algorithms (e.g. Logistic Regression) require numeric data so the columns with string-data need to be transformed. The columns in this data-set that had 'yes' or 'no' values had the values converted to 1 and 0 respectively. Those columns that had other kinds of categorical data were transformed into dummy-columns.




Split data into training and test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next the data was converted into training and testing sets.






.. csv-table:: Training and Testing Data
   :header: Set, Count

   Training Features,300
   Test Features,95

