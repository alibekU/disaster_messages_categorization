'''
    support_functions.py - additional custom functions that can be accessed from both training and flask app files
    Author: Alibek Utyubayev
'''

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
import scipy.stats as ss

def tokenize(text):
    '''
        tokenize() - function that tokenizes a given English text. Will be used in ML pipeline to prepare text.
        Input:
            text -  a string to tokenize
        Output:
            clean_tokens - a list of cleaned tokens
    '''
    # url pattern
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # detect urls and replace them with some placeholders
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # break down text into separate words - tokens
    tokens = word_tokenize(text)
    # initialize a lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize each token, strip it from whitespaces and convert to lower case
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # then append each "clean" token into output list
        clean_tokens.append(clean_tok)

    return clean_tokens

def cramers_v(x, y):
    '''
        cramers_v() - function to compute cramers V coefficient which measures how two arrays with categorical data are associated.
        Credits to https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        Input:
            x,y - two arrays of same length with categorical values
        Output:
            Cramers V - a number from 0 to 1, where 0 means no correlation, 1 - full correlatin (association)
    '''
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    res= np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))) 
    return round(res,2)

def create_heatmap_data(arr, func):
    '''
        fill_matrix_efficiently() - function to create an n x n matrix of form 
            func(a,a)  func(a,b)  func(c,c) ...
            func(b,a)  func(b,b)  func(b,c) ...
            func(c,a)  func(c,b)  func(c,c) ...
            ...        ...        ...       ...
        where a,b,c are the first 3 elements or columns of a Pandas dataframe 'arr' that has n elements/columns,
        and func is some function that is applied to the elements/columns of arr pairwise 

        Credits to https://stackoverflow.com/questions/57167806/most-effective-method-to-create-and-fill-a-numpy-matrix-from-an-array
        Input:
            arr - a Pandas dataframe of n elements or n columns 
            func - a function that takes 2 elements or columns of arr and returns some result
        Output:
            res - a n x n matrix of the abovementioned form
    '''
    column_names = arr.columns
    # the 'n' - number of elements/columns in the arr, the output matrix will have n x n elements
    number_elements = arr.shape[1]
    # compute indices of elements in lower triangle
    rows,cols=np.tril_indices(number_elements)
    values = []
    for left, right in zip(rows, cols):
        values.append(func(arr[column_names[left]], arr[column_names[right]] ))
    # convert arr to a numpy
    #arr = np.array(arr)
    # compute arr values for those indices
    #left,right=arr[rows],arr[cols]
    # do the stuff
    #values=func(left,right)
    # preallocate result
    res=np.zeros((number_elements, number_elements))
    # assign lower triangle
    res[rows,cols]=values
    # assing upper triangle
    res[cols,rows]=values

    return res
