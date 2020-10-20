'''
    process_data.py - ETL processes for Disaster Messages Categorization web app.
    1. Extracts and transforms data from csv files and then load in sqlite database for ML training pipeline later.
    2. Computes and saves data for the web app visualizations, so that web app only extracts the data and does not compute it everytime.
    Author: Alibek Utyubayev.

    Usage:
        Need to pass following arguments as sys.argv to the program:
            messages_filepath -  a string with a filepath to a CSV file with messages to train ML model
            categories_filepath - a string with a filepath to a CSV file with labels for the messages to train ML model
            database_filepath - a string with a filepath to a sqllite database file where data should be stored. If it does not exist, then a new one will be created 
'''

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys
import numpy as np
import scipy.stats as ss


def load_data(messages_filepath, categories_filepath):
    '''
        load_data() - function that creates a Pandas dataframe from given CSV files
        Input:
            messages_filepath -  a string with a filepath to a CSV file with messages to train ML model
            categories_filepath - a string with a filepath to a CSV file with labels for the messages to train ML model
        Output:
            df - a Pandas dataframe with messages and categories, where each category is a column which contains 1 if a message (row) is in the category and 0 otherwise
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

      # merge datasets
    df = pd.merge(messages, categories, on='id')
   
    # create a dataframe of the 36 individual category columns
    categories_new = categories['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories_new.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [word[0:-2] for word in row]

    # rename the columns of `categories`
    categories_new.columns = category_colnames

    for column in categories_new:
        # set each value to be the last character of the string
        categories_new[column] = categories_new[column].str[-1]
    
        # convert column from string to numeric
        categories_new[column] = pd.to_numeric(categories_new[column])

        # convert values greater than 1 to 1 as we have classification task
        categories_new[column] = categories_new[column].apply(lambda x: 1 if x > 1 else x)

    # drop 'child_alone' category as it has all zeroes
    categories_new.drop(columns=['child_alone'], inplace=True)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_new], axis=1)

    return df


def clean_data(df):
    '''
        clean_data() - function that cleanes a Pandas dataframe for future storage and analysis
        Input:
            df - a Pandas dataframe with messages and categories (labels)
        Output:
            df - a cleaned Pandas dataframe with messages and categories
    '''
    # replace nulls in the 'original' column with '' as the rows still have message in English and we do not need to drop them
    df['original'] = df['original'].fillna('')
    #drop nulls
    df = df.dropna()
    # drop duplicates
    df = df.drop_duplicates(subset=['message'])

    return df

def save_data(df, table_name, database_filename):
    '''
        save_data() - a function that saves a Pandas dataframe into sqlite database
        Input:
            df -  a Pandas dataframe with data to save
            table_name - a string with the name of the table where df will be stored
            database_filename - a string with a filepath to a sqllite database file where data should be stored. If it does not exist, then a new one will be created 
        Output:
            None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace')  

 # finding correlation between classes
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

# building a matrix with data for heatmap
def create_heatmap_data(arr, func):
    '''
        fill_matrix_efficiently() - function to efficiently create an n x n matrix of form 
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
    # compute correlations for the lower triangle part
    for left, right in zip(rows, cols):
        values.append(func(arr[column_names[left]], arr[column_names[right]] ))
    # preallocate result
    res=np.zeros((number_elements, number_elements))
    # assign lower triangle
    res[rows,cols]=values
    # assing upper triangle
    res[cols,rows]=values

    return res


def main():
    '''
        main() - function that performs an ETL process on messages and categories data 
                and saves data for data visuals in DB for future use by the web app
        Input:
            None the function, but need input as system arguments to the program
        Output:
            None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        # get the messages and categories from CSV files
        df = load_data(messages_filepath, categories_filepath)

        # transform the messages and categories data
        print('Cleaning data...')
        df = clean_data(df)
        
        # load the messages and categories data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, "Messages", database_filepath)

        # --- Compute and save data for visuals to the DB to avoid computing it everytime a page is loaded ---
        print('Saving data for the web app visuals...\n    DATABASE: {}'.format(database_filepath))
        # get the genre distribution
        genre_counts = df.groupby('genre').count()['message']
        # save the genre distribution
        save_data(genre_counts, "Genre_Counts", database_filepath)

        # get the labels - Y dataframe - for visuals
        Y = df[df.columns[4:]]
        # get the names of the categories
        category_names = Y.columns.to_list()
       
        # get the category distribution data
        sums_categories = Y.sum(axis=0).sort_values()
        #save the category distribution data
        save_data(sums_categories, "Category_Counts", database_filepath)
        
        # create a matrix of correlations btw categories for heatmap graph
        corr = create_heatmap_data(Y, cramers_v)
        # create a dataframe from the matrix
        corr_df = pd.DataFrame(data=corr, index=category_names, columns=category_names)
        #save the heatmap correalation data
        save_data(corr_df, "Category_Correlations", database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster_response.db')


if __name__ == '__main__':
    main()