
# python standard library
import pickle

# third party
import pandas
from sklearn.cross_validation import train_test_split

# this code
from common import (student_data, feature_map, TrainTestData,
                    train_test_path, RANDOM_STATE)

# Extract feature (X) and target (y) columns
feature_columns = list(student_data.columns[:-1])  # all columns but last are features
target_column = student_data.columns[-1]  # last column is the target/label
X_all = student_data[feature_columns]  # feature values for all students
y_all = student_data[target_column]  # corresponding targets/labels

assert len(y_all) == 395, "Expected: 395 Actual: {1}".format(len(y_all))

for column_name in sorted(feature_columns):
    column = X_all[column_name] 
    dtype = column.dtype
    examples = (', '.join(sorted(column.unique())) if dtype == object else
                ', '.join([str(item) for item in sorted(column.unique())]) if len(column.unique()) < 10
                else
                "{0} ... {1}".format(column.min(), column.max()))
    print('   {0};{1};{2}'.format(column_name,
                                      feature_map[column_name],
                                      examples))

def preprocess_features(X):
    """
    Converts categorical data to numeric
    :param:
     - `X`: dataframe of data
    :return: data with yes/no changed to 1/0, others changed to dummies
    """
    outX = pandas.DataFrame(index=X.index)

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pandas.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe
    return outX

X_all = preprocess_features(X_all)
y_all = y_all.replace(['yes', 'no'], [1, 0])
assert len(y_all) == 395, "Expected: 395 Actual: {0}".format(len(y_all))

original_columns = len(feature_columns)
with_dummies = len(X_all.columns)
print("   * Original Feature Columns: {0}".format(original_columns))
print("   * With Dummies: {0}".format(with_dummies))
print("\nWith dummy variables there are now {0} more columns in the feature data.".format(with_dummies - original_columns))

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
assert num_all == 395, "Expected: 395 Actual: {0}".format(num_all)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                    test_size=num_test,
                                                    train_size=num_train,
                                                    random_state=RANDOM_STATE)
assert len(y_train) == 300
assert len(y_test) == 95

data = TrainTestData(X_train = X_train,
                     X_test = X_test,
                     y_train = y_train,
                     y_test = y_test)
with open(train_test_path, 'wb') as pickler:
    pickle.dump(data, pickler)

print("   Training Instances,{0}".format(X_train.shape[0]))
print("   Test Instances,{0}".format(X_test.shape[0]))