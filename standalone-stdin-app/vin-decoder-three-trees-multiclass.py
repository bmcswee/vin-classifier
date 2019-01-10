# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 08:26:56 2018

@author: Brendan, Nathan and Seth
"""

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

print("Loading Data Set")


# load data
dataFolder = "..\\data-cleaner\\clean-data\\"
df = pd.read_csv(dataFolder + "dataset-fullsplit-make-model.csv")

# function to drop labels with numbers of samples too low for training
def drop_low_samples(data, label_name, drop_below):
    dfcount = pd.DataFrame(df[label_name].value_counts())
    dfcount.reset_index(inplace=True)
    dfcount.columns = [label_name, 'counts']
    dfwithcount = df.merge(dfcount, on=label_name)
    dfwithcount = dfwithcount[dfwithcount.counts >= drop_below]
    return dfwithcount.drop(['counts'], axis=1)

##############################################################################
# YEAR

print("Training Year Decision Tree Classifier")

# Drop all years with 5 or fewer available training examples
year_df = drop_low_samples(data=df, label_name='year', drop_below=5)

# split data into test & train sets
X = year_df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = year_df['year']
le = LabelEncoder().fit(y) 
year_labels = le.transform(y)
year_classes = list(le.classes_)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]

# train decision tree
yearclf = tree.DecisionTreeClassifier()
yearclf = yearclf.fit(X_train, y_train)

##############################################################################
# MAKE

print("Training Make Decision Tree Classifier")

# Drop all makers with 5 or fewer available training examples
make_df = drop_low_samples(data=df, label_name='make', drop_below=5)

# split data into test & train sets
X = make_df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = make_df['make']
le = LabelEncoder().fit(y) 
make_labels = le.transform(y)
make_classes = list(le.classes_)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = make_labels[train_index], make_labels[test_index]

# train decision tree
makeclf = tree.DecisionTreeClassifier()
makeclf = makeclf.fit(X_train, y_train)

##############################################################################
# MODEL

print("Training Model Decision Tree Classifier")

# Drop all models with 5 or fewer available training examples
model_df = drop_low_samples(data=df, label_name='model', drop_below=5)

# split data into test & train sets
X = model_df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = model_df['model']
le = LabelEncoder().fit(y) 
model_labels = le.transform(y)
model_classes = list(le.classes_)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = model_labels[train_index], model_labels[test_index]

# train decision tree
modelclf = tree.DecisionTreeClassifier()
modelclf = modelclf.fit(X_train, y_train)

##############################################################################
# Stdin reader and predictor

ex = 0

while ex == 0:

    s = input ('Enter VIN Number (type q to exit program): ')

    if s == 'q':
        ex = -1
    else:
        if s.isalnum() == True and len(s) == 17: # Ensure user input is valid VIN
            # Encode inputted VIN number to match expected input of trained model
            s = s.upper()
            arr = []

            for c in s:
                try:
                    vinchar_int = int(c)
                except ValueError:
                    vinchar_int = ord(c) - ord('A') + 10
                # End of try-catch

                arr.append(vinchar_int)
            # End of for loop

            # Predict vehicle year, make, and model using respective decision trees
            year_pred = yearclf.predict([arr])
            make_pred = makeclf.predict([arr])
            model_pred = modelclf.predict([arr])

            # Map numerical labels produced by decision tree to human-readable text labels
            final_year = year_pred[0]
            final_make = make_classes[make_pred[0]]
            final_model = model_classes[model_pred[0]]
            print(final_year, final_make, final_model)
        else:
            print('Invalid VIN number, please try again.')
        # End of if statement
    # End of if statement
# End of while loop

# that's all