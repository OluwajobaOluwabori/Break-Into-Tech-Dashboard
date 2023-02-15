# # Project: Investigate how developers are transitioning into tech
# ## Introduction
# ### Dataset Description
# 
# 
# 
# To proceed with this analysis, I would be answering three questions which are:
# - The formal education of current developers and this showed that the highest level of formal education for most developers is Bachelor's degree.
# - The different methods developers use to learn how to code. We saw that majority learnt how to code outside school and the top online platforms used are Udemy, Coursera and Codeacademy.
# - Lastly, we will look at the top tools and technologies developers are working with and what they would love to work with in the future. This will help bring you up to date on current tools in your desired role and help you decide what tool to learn as well.

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
import plotly.colors
from collections import OrderedDict
import requests

def chart():

    #read in the data and
    df = pd.read_csv("survey_results_public.csv")


    #Extracting the columns we need
    df_new = df.iloc[:,np.r_[0,5,6,7,8,11,19:37,39:43,58]].reset_index(drop=True)



    
    # ##### There are 73,268 rows and 79 columns (i.e this data consists of 73,268 developers responses to 79 questions/sub-questions.                        The columns/questions we will be working with are printed below

    
    # ## Data Cleaning
    # 
    # Since most of our analysis is going to be career based, we will remove the null rows w.r.t to DevType column. And obtain the unique responses for the multi-choice questions.


    # Handling missing data
    # Drop rows with no response to ''Which of the following describes your current job?''
    df_new.dropna(subset =['DevType'],inplace=True)

    df_new=df_new.reset_index(drop=True)

    #df_new.dtypes


    #Total number of response to the question 'Which of the following describes your current job?' (DevType) is 61302


    def unique_answers(column):
        '''
        
        This function obtains the unique answers(provided options) for any given question.
        df - dataframe
        
        INPUT 
            
            column - the column name you want to obtain the unique answers from.
            
        OUTPUT
        
            Unique_ans - A set containing all the possible (unique) answers in the column
        '''
        df1=df_new.dropna(subset = [column])
        Unique_vals = set()
        df1[column]=df1[column].str.split(';')
        for row in df1[column]:
            for role in row:
                Unique_vals.add(role)
        return(Unique_vals)




    '''

    Calling the function to obtain the possible(unique) answers for the necessary columns.

    Current & Next will always be the same.

    '''
    Unique_roles = unique_answers('DevType')
    Unique_current_languages = unique_answers('LanguageHaveWorkedWith')
    Unique_next_languages = unique_answers('LanguageWantToWorkWith')
    Unique_current_db = unique_answers('DatabaseHaveWorkedWith')
    Unique_next_db = unique_answers('DatabaseWantToWorkWith')
    Unique_current_platform = unique_answers('PlatformHaveWorkedWith')
    Unique_next_platform = unique_answers('PlatformWantToWorkWith')
    Unique_current_Webframe = unique_answers('WebframeHaveWorkedWith')
    Unique_next_Webframe = unique_answers('WebframeWantToWorkWith')
        



    def split_multiple_answers(column):
        '''
        
        This function splits the column with multiple answers and convert each cell into a string.
        df - dataframe

        INPUT 
            
            column - the column name with multiple answers that needs to be splitted
            
        OUTPUT
        
            Splitted_col - A set containing all unique answers in the different rows
            
        '''
        df = df_new.dropna(subset = [column])
        df[column] = df[column].str.split(';')
        
        return(df[column])


    df_new['DevType'] = split_multiple_answers('DevType')


    '''
    Using Pandas melt function to convert list-like column elements to separate rows 

    1. Split DevType column list values (columns with individual list values are created).

    2. Merge the new columns with the rest of the data set.

    3. Drop the old names list column and then transform the new columns into separate rows using the melt function.

    4. Additional column ‘variable’ containing the ids of the numeric columns is seen. This column is dropped and empty values are removed.

    '''

    df_clean=df_new.DevType.apply(pd.Series) \
    .merge(df_new, right_index = True, left_index = True) \
    .drop(["DevType"], axis = 1) \
    .melt(id_vars = df.iloc[:,np.r_[0,5,6,7,8,19:37,39:43,58]], value_name = "DevType")\
    .dropna(subset=['DevType'])\
    .drop("variable", axis=1) 

    
    # ##### Responses to the question 'Which of the following describes your current job? Please select all that apply.' is now splitted into different rows according to the respective ids.
    # 


    # ##  Exploratory Data Analysis




    # #### Let's find the proportion of each role
    # 
    # #### From the code below, the majority of respondents are Developers 

    # %%
    DevType_count=(df_clean['DevType'].value_counts()/df_new['DevType'].shape[0]).mul(100).reset_index().rename(columns={'index':'Role','DevType':'count'})
    DevType_count.columns=['Role','count']
    DevType_count

    # A horizontal bar chart showing the proportion

    plt.figure(figsize=(8,8))
    plt.barh(DevType_count['Role'],DevType_count['count'])
    plt.title("Which of the following describes your current job?")
    #plt.show()

    
    import plotly.graph_objects as go
    fig = go.Figure([go.Bar(x=DevType_count['Role'], y=DevType_count['count'])])
    #fig.update_layout(title_text='January 2013 Sales Report')
    layout = dict(title = 'Change in Rural Population <br> (Percent of Total Population)')   
    #fig.show()
    return fig
