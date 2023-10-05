# -------------------------------------------------------------------------
# AUTHOR: Tommy James
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
db = []
test1 = []

# reading the training data in a csv file
# --> add your Python code here
# reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        db.append(row)
        # print(row)

rows = len(db)-1
cols = len(db[0])-2
# print(f'rows: {rows}\ncols:{cols}')

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
X = [[0 for i in range(cols)] for j in range(rows)]
# print(X)

feature_vals = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3,
    'Hot': 1,
    'Mild': 2,
    'Cool': 3,
    'High': 1,
    'Normal': 2,
    'Weak': 1,
    'Strong': 2,
    'Yes': 1,
    'No': 2
}


def trans_feats(db, rows, cols, X, feature_vals):
    for i in range(rows):
        for j in range(cols):
            if db[i+1][j+1] in feature_vals:
                X[i][j] = feature_vals[db[i+1][j+1]]
        #         print(X[i][j], end=' ')
        # print()

    # print(f'\nX: {len(X)} x {len(X[0])}')
    return X


trans_feats(db, rows, cols, X, feature_vals)
# print('------------------')

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
Y = [0 for i in range(rows)]
# print(Y)
# print(len(Y))

# transforms classes to numbers and adds to vector Y


def trans_class(db, rows, cols, feature_vals, Y):
    # print(f'rows: {rows}\ncols:{cols}')
    # print('----------------')
    for i in range(rows):
        if db[i+1][cols+1] in feature_vals:
            Y[i] = feature_vals[db[i+1][cols+1]]
            # print(Y[i], end=' ')

    # print(f'\nY: {len(Y)}')
    return Y


trans_class(db, rows, cols, feature_vals, Y)

# fitting the naive bayes to the data
# X = trans_feats(db, rows, cols, X, feature_vals)
# Y = trans_class(db, rows, cols, feature_vals, Y)
clf = GaussianNB()
clf.fit(X, Y)

# print('\n----------------')

# reading the test data in a csv file
# --> add your Python code here
# print('TESTING')
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        test1.append(row)
        # print(row)

rows = len(test1)-1
cols = len(test1[0])-2
# print(f'rows: {rows}\ncols:{cols}')
# print('----------------')

# printing the header os the solution
# --> add your Python code here
for i in range(len(test1[0])):
    print(f'\t{test1[0][i]}', end=' ')
print('\tConfidence')

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# --> add your Python code here
# Sunny = 1, Overcast = 2, Rain = 3
# Hot = 1, Mild = 2, Cool = 3
# High = 1, Normal = 2
# Weak = 1, Strong = 2
# Yes = 1, No = 2

print('\tD15 \tSunny \t\tHot \t\tNormal \t\tWeak \tYes \t\t', end='')
print(round(max(clf.predict_proba([[1, 1, 2, 1]])[0]), 2))  # D15
# print(round(max(clf.predict_proba([[1, 1, 2, 2]])[0]), 2))  # D16
# print(round(max(clf.predict_proba([[1, 3, 1, 1]])[0]), 2))  # D17
# print(round(max(clf.predict_proba([[1, 1, 2, 2]])[0]), 2))  # D18
# print(round(max(clf.predict_proba([[2, 3, 1, 1]])[0]), 2))  # D19
# print(round(max(clf.predict_proba([[2, 3, 1, 2]])[0]), 2))  # D20
print('\tD21 \tRain \t\tMild \t\tNormal \t\tStrong \tYes \t\t', end='')
print(round(max(clf.predict_proba([[3, 2, 2, 2]])[0]), 2))  # D21
print('\tD22 \tRain \t\tHot \t\tNormal \t\tStrong \tYes \t\t', end='')
print(round(max(clf.predict_proba([[3, 1, 2, 2]])[0]), 2))  # D22
# print(round(max(clf.predict_proba([[3, 1, 1, 1]])[0]), 2))  # D23
# print(round(max(clf.predict_proba([[3, 2, 1, 2]])[0]), 2))  # D24
