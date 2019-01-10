# -*- coding: utf-8 -*-
"""
Train three decision trees (year, make, model) and output each decision tree to a graph. Unfortunately, it's a huge pain to get
GraphViz working. See: https://stackoverflow.com/questions/40632486/dot-exe-not-found-in-path-pydot-on-python-windows-7
ABOVE LINK WORKS ON WIN10 AS WELL PER MY TESTING

Created on Wed Nov 28 08:26:56 2018

@author: Brendan
"""
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import os
from usm_ml_utils import drop_low_samples
# IMPORTANT: Path to Graphviz is system specific! Should be changed to match your environment
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load data
dataFolder = "..\\data-cleaner\\clean-data\\"
df = pd.read_csv(dataFolder + "dataset-fullsplit-make-model.csv")

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
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, y_train)

# test decision tree
train_predictions = treeclf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("VEHICLE YEAR Accuracy: {:.4%}".format(acc))

classnamestrings = [str(i) for i in year_classes]
# make a graph and write it to the disk
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
graph[0].write_pdf("year-tree.pdf")


##############################################################################
# Do the same thing, but for vehicle MAKE

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
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, y_train)

# test decision tree
train_predictions = treeclf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("VEHICLE MAKE Accuracy: {:.4%}".format(acc))

classnamestrings = [str(i) for i in make_classes]
# make a graph and write it to the disk
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
graph[0].write_pdf("make-tree.pdf")


##############################################################################
# Do the same thing, but for vehicle MODEL

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
treeclf = tree.DecisionTreeClassifier()
treeclf = treeclf.fit(X_train, y_train)

# test decision tree
train_predictions = treeclf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("VEHICLE MODEL Accuracy: {:.4%}".format(acc))

classnamestrings = [str(i) for i in model_classes]
# make a graph and write it to the disk
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
graph[0].write_pdf("model-tree.pdf")
    
# that's all
