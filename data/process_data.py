'''
    process_data.py - ETL processes for Disaster Messages Categorization web app.
    Code to extract and transform data from csv files and then load in sqlite database for ML training pipeline later.
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

def save_data(df, database_filename):
    '''
        save_data() - function that saves a Pandas dataframe into sqlite database
        Input:
            df -  a Pandas dataframe with data to save
            database_filename - a string with a filepath to a sqllite database file where data should be stored. If it does not exist, then a new one will be created 
        Output:
            None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
    '''
        main() - function that performs an ETL process on messages and categories data
        Input:
            None the function, but need input as system arguments to the program
        Output:
            None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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