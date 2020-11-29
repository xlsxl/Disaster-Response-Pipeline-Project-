# import librariries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Function loads datas from given filepaths, merges them into single data frame.
    
    Input --> messages_filepath  : includes the file path as string (example : "disaster_messages.csv")
              categories_filepath: includes the file path as string (example : "disaster_categories.csv")
    
    Output --> df: a single data frame
    
    """
    
    # read datas
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datas
    df = pd.merge(messages, categories, on = "id" )
    
    return df


def clean_data(df):
    
    """
    Function takes df as a dataframe and returns df after cleaning.
    
    Cleaning operation:
        1.Splits categories into seperate columns
        2.Change values of categories columns to numbers 0 or 1
        3.Drop duplicates
        4.Cleaning of column "related"
    
    """
   
    # 1. Split categories into separate category columns
    
    # create a dataframe from 36 individual category columns
    categories = df.categories.str.split(";", expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values.tolist()

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [each.split("-")[0] for each in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # 2.Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand = True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop("categories", axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)  
    
    # 3.Drop duplicates
    df.drop_duplicates(inplace = True)
    
    # 4.Values of 2 at column "related" will be replaced with 0; because they are not related to any categories.
    df.related = [0 if value == 2 else value for value in df.related]
    
    return df

def save_data(df, database_filename):
    
    """
    Function saves data as an sql database.
    
    input -->   df: dataframe
                database_filename: name for database 
        
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    #table = database_filename.replace(".db","_table")
    df.to_sql("table", engine, index=False, if_exists='replace')
    
    
def main():
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
              'DisasterResponse.db')


if __name__ == '__main__':
    main()