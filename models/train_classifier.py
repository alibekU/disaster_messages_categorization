'''
    train_classifier.py - ML training pipeline for Disaster Messages Categorization web app.
    Code to load messages data from sqlite database to create a multi-label classifiation model to categorize new messages from the web app.
    Author: Alibek Utyubayev.

    Usage:
        Need to pass following arguments as sys.argv to the program:
            database_filepath -  a string with a filepath to a sqllite database file where the data is stored
            model_filepath - a string with a filepath to a Pickle file where ML model, that was trained on the data and is ready to classify new messages, will be stored
'''

# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, print(cv_ada.best_params_)
from scipy.stats import gmean
import re
import pickle
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import sys


def load_data(database_filepath):
    '''
        load_data() - function that creates a Pandas dataframe by reading data from sqlite database
        Input:
            database_filepath -  a string with a filepath to a sqllite database file where the data is stored
        Output:
            X - a Pandas Series for training and testing data - messages in English
            Y - a Pandas dataframe of training and test labels - categories of the messages, each category is a column which contains 1 if a message (row) is in the category and 0 otherwise
            categories - a list of strings - names of the categories
    '''
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Messages', engine)

    X = df['message']
    Y = df[df.columns[4:]]
    categories = Y.columns
    return X, Y, categories

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


def build_model():
    '''
        build_model() - function that creates a Pandas dataframe frim given CSV files
        Input:
            None as model's parameters were tuned and optimized in a Jupyter Notebook 'ML Pipeline Preparation' and here we simply recreate the best model
        Output:
            model - a sklearn Pipeline object, model to be trained on existing data and predict categories for the new data
    '''
    model = Pipeline(steps=[('vector',
                 CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf',
                 MultiOutputClassifier(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced',
                                                                                                          max_depth=1),
                                                                    learning_rate=0.5)))])
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
        evaluate_model() - function that evaluates an sklearn model
        Input:
            model -  a string with a filepath to a CSV file with messages to train ML model
            X_test - a string with a filepath to a CSV file with labels for the messages to train ML model
            y_test - 
            category_names -
        Output:
            None, prints out the report
    '''
    # get the model prediction
    y_pred = model.predict(X_test)
    f1_array = []
    # for each category of labels (each column of Y) 
    #   1 print out classification_report
    #   2 compute f1 score for the positive class (value = 1)
    # then compute an average of all the f1 scores (global_f1_score)
    for index, column in enumerate(category_names):
        prediction = y_pred[:,index]
        actual = y_test.iloc[:,index]
        print('Category = {}'.format(column))
        print(classification_report(actual, prediction))
        #print(precision_recall_fscore_support(actual, prediction, average='binary'))
        f1 = f1_score(actual, prediction)
        f1_array.append(f1)
    global_f1_score = np.mean(f1_array)
    print('\n')
    print('--- Arithmetic mean of all f1 scores for each category = {}'.format(round(global_f1_score,2)))
    return None


def save_model(model, model_filepath):
    '''
        save_model() - function that saves a classification model into a Pickle file for later use in web app
        Input:
            model -  a sklearn Pipeline object - model that was trained on existing data and can predict categories for the new data
            model_filepath - a string with a filepath to a Pickle file where ML model, that was trained on the data and is ready to classify new messages, will be stored
        Output:
            None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return None


def main():
    '''
        main() - function that controls ML pipeline
        Input:
            None the function, but need input as system arguments to the program
        Output:
            None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()