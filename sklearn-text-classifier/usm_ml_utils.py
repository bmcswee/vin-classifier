# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:01:28 2018

@author: Brendan
"""
import pandas as pd

# function to drop labels with numbers of samples too low for training
def drop_low_samples(data, label_name, drop_below):
    dfcount = pd.DataFrame(data[label_name].value_counts())
    dfcount.reset_index(inplace=True)
    dfcount.columns = [label_name, 'counts']
    dfwithcount = data.merge(dfcount, on=label_name)
    dfwithcount = dfwithcount[dfwithcount.counts >= drop_below]
    return dfwithcount.drop(['counts'], axis=1)