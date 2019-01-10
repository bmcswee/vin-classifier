# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 08:26:56 2018

@author: Brendan, Nathan and Seth
"""

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split

print("Loading Data Set")

# IMPORTANT: Path to Graphviz is system specific! Should be changed to match your environment
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load data
dataFolder = "..\\data-cleaner\\clean-data\\"
df = pd.read_csv(dataFolder + "dataset-fullsplit-make-model.csv")

# function to drop labels with numbers of samples too low for training
def drop_low_samples(data, label_name, drop_below):
    dfcount = pd.DataFrame(data[label_name].value_counts())
    dfcount.reset_index(inplace=True)
    dfcount.columns = [label_name, 'counts']
    dfwithcount = data.merge(dfcount, on=label_name)
    dfwithcount = dfwithcount[dfwithcount.counts >= drop_below]
    return dfwithcount.drop(['counts'], axis=1)


##############################################################################
# Train YEAR and MAKE and MODEL together

print("Training Year, Make, and Model Decision Tree Classifier")

# Drop all makers with 5 or fewer available training examples
make_df = drop_low_samples(data=df, label_name='make', drop_below=10)

# split data into test & train sets
X = make_df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = make_df.drop(['makemodel', 'vinregex', 'vinchar1', 'vinchar2'
                  , 'vinchar3', 'vinchar4', 'vinchar5', 'vinchar6', 'vinchar7'
                  , 'vinchar8', 'vinchar9', 'vinchar10', 'vinchar11', 'vinchar12'
                  , 'vinchar13', 'vinchar14', 'vinchar15', 'vinchar16', 'vinchar17'], axis=1)

labels = y.apply(LabelEncoder().fit_transform)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

y_test_labels = y_test.apply(LabelEncoder().fit_transform)
y_train_labels = y_train.apply(LabelEncoder().fit_transform)

# train decision tree
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, y_train_labels)

ex = 0

while ex == 0:

    s = input ('Enter VIN Number (type q to exit program): ')

    if s == 'q':
        ex = -1
    else:
        if s.isalnum() == True and len(s) == 17:
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

            # Predict the inputed vin number
            pred = treeclf.predict([arr])
            final = -1

            # Get correct labels out and print them
            for index, row in enumerate(y_train_labels.values):
                if row[0] == pred[0][0]:
                    if row[1] == pred[0][1]:
                        if row[2] == pred[0][2]:
                            final = index
                            break
                        # End of if statement
                    # End of if statement
                # End of if statement
            # End of for loop
                
            if final != -1:
                r = y_train.iloc[[final]]
                print(r.values[0][0], r.values[0][1], r.values[0][2])
            else:
                print('Unable to find')
            # End of if statement
        else:
            print('Invalid VIN number, please try again.')
        # End of if statement
    # End of if statement
# End of while loop

