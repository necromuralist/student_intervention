
# third party
import pandas

# this code
from common import student_data, feature_map

# Extract feature (X) and target (y) columns
feature_columns = list(student_data.columns[:-1])  # all columns but last are features
target_column = student_data.columns[-1]  # last column is the target/label
X_all = student_data[feature_columns]  # feature values for all students
y_all = student_data[target_column]  # corresponding targets/labels

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

for column in X_all.columns:
    if '_' in column:
        print(" * {0}".format(column))

for value in X_all.passed.unique():
    print('   {0},{1}'.format(value, len(X_all[X_all.passed==value])))