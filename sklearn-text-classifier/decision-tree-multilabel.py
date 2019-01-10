# -*- coding: utf-8 -*-
"""
Train ONE single decision tree to predict vehicle year, make, and model all together.
Then output the big decision tree to a graph. Unfortunately, it's a huge pain to get
GraphViz working. See: https://stackoverflow.com/questions/40632486/dot-exe-not-found-in-path-pydot-on-python-windows-7
ABOVE LINK WORKS ON WIN10 AS WELL PER MY TESTING

Created on Wed Nov 28 08:26:56 2018

@author: Brendan
"""
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
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

# Drop all makers with 5 or fewer available training examples
make_df = drop_low_samples(data=df, label_name='make', drop_below=10)

# split data into test & train sets
X = make_df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = make_df.drop(['makemodel', 'vinregex', 'vinchar1', 'vinchar2'
                  , 'vinchar3', 'vinchar4', 'vinchar5', 'vinchar6', 'vinchar7'
                  , 'vinchar8', 'vinchar9', 'vinchar10', 'vinchar11', 'vinchar12'
                  , 'vinchar13', 'vinchar14', 'vinchar15', 'vinchar16', 'vinchar17'], axis=1)

labels = y.apply(LabelEncoder().fit_transform)

# SKLearn's Stratified shuffle split DOES NOT WORK for multilabel. Random split must be used
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

y_test_labels = y_test.apply(LabelEncoder().fit_transform)
y_train_labels = y_train.apply(LabelEncoder().fit_transform)


# train decision tree
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, y_train_labels)

# test decision tree
train_predictions = treeclf.predict(X_test)

# SKLearn's accuracy score implementation DOES NOT WORK with multilabel.
# Instead, a Hamming loss can be calculated manually.
# see: https://stackoverflow.com/questions/38697982/python-scikit-learn-multi-class-multi-label-performance-metrics
import numpy as np
hamming = np.sum(np.not_equal(y_test_labels, train_predictions))/float(y_test_labels.size)
print ("Hamming losses:")
print(hamming)

classnamestrings = [str(i) for i in labels]
# make a graph and write it to the disk
#  class name labels ARE NOT SUPPORTED for multilabel per the sklearn docs
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(treeclf, out_file=dot_data,
                     feature_names=X.columns.values,
                     class_names=classnamestrings,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
# ensure that Graphviz is installed on the machine, or this will fail!
graph[0].write_pdf("multilabel-tree.pdf")
    
# that's all
