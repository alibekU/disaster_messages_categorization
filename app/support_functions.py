'''
    support_functions.py - additional custom functions that can be accessed from both training and flask app files
    Author: Alibek Utyubayev
'''

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

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