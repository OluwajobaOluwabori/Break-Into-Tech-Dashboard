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
import copy
from itertools import cycle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
import plotly.colors
from collections import OrderedDict
import requests
from itertools import cycle
import plotly.express as px

roles_default = ['Data or business analyst',
    'Data scientist or machine learning specialist',
    'Developer, back-end'#,
    # 'Developer, front-end',
    # 'Developer, full-stack',
    # 'DevOps specialist',
    # 'Designer',
    # 'Developer, embedded applications or devices'
    ]


def chart(roles=roles_default):
    
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
    # Unique_current_db = unique_answers('DatabaseHaveWorkedWith')
    # Unique_next_db = unique_answers('DatabaseWantToWorkWith')
    # Unique_current_platform = unique_answers('PlatformHaveWorkedWith')
    # Unique_next_platform = unique_answers('PlatformWantToWorkWith')
    # Unique_current_Webframe = unique_answers('WebframeHaveWorkedWith')
    # Unique_next_Webframe = unique_answers('WebframeWantToWorkWith')
        



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
 
   
    df=df_clean.groupby(['DevType'])['WorkExp'].mean().sort_values()
    labels = df.index
    import plotly.graph_objects as go
    fig1 = go.Figure()
    dic={}
    count=0
    for index,label in enumerate(labels):
        dic[label]=index
        count+=1
        k=[df[df.index == label].values][0]
        fig1.add_traces(go.Bar(x=[label], y=k,text=k,
                textposition='auto',showlegend=False,marker=dict(
            #size=16,
            cmax=20,
            cmin=10,
            color=k,

            colorscale='Picnic'
        )))
        fig1.update_traces(texttemplate='%{text:.2s}')

    buttonsList=[dict(label='All',
                method="update",
                args=[{"visible": [True]*count},
                    {"title": "Tools used by role: All"}])]

    test=[False]*count
    for i in range(0,count):
        test[i]=True
        buttonsList.append(dict(label=labels[i],
                method="update",
                args=[{"visible": test},
                    {"title": "Tools used by role: All"}]))
        test=[False]*count
    fig1.update_layout(updatemenus=[dict( type='dropdown', active=0, buttons=list(buttonsList))], 
                       xaxis_tickangle=-45,
                       font_color="rgb(237, 216, 216)",
                       #hoverlabel_bgcolor="rgb(68, 112, 118)"
                       paper_bgcolor='rgb(68, 112, 118)',
                       plot_bgcolor= 'rgb(68, 112, 118)')
    #fig.show()
    figures = [fig1]
    #figures.append(fig1)
    #Calling the function to obtain the possible(unique) answers for the necessary columns

    df_clean['LanguageHaveWorkedWith']= split_multiple_answers('LanguageHaveWorkedWith')
    df_clean['LanguageWantToWorkWith']=split_multiple_answers('LanguageWantToWorkWith')
    df_clean['DatabaseHaveWorkedWith']=split_multiple_answers('DatabaseHaveWorkedWith')
    df_clean['DatabaseWantToWorkWith']=split_multiple_answers('DatabaseWantToWorkWith')
    df_clean['PlatformHaveWorkedWith']=split_multiple_answers('PlatformHaveWorkedWith')
    df_clean['PlatformWantToWorkWith']=split_multiple_answers('PlatformWantToWorkWith')
    df_clean['WebframeHaveWorkedWith']=split_multiple_answers('WebframeHaveWorkedWith')
    df_clean['WebframeWantToWorkWith']=split_multiple_answers('WebframeWantToWorkWith')
    columns=[['LanguageHaveWorkedWith','LanguageWantToWorkWith'],['DatabaseHaveWorkedWith','DatabaseWantToWorkWith'],['PlatformHaveWorkedWith','PlatformWantToWorkWith']]

    for col1,col2 in columns:
        df1 = df_clean[col1].apply(pd.Series) \
            .merge(df_clean, right_index = True, left_index = True) \
            .drop([col1], axis = 1) \
            .melt(id_vars = df_new.loc[:,df_new.columns!=col1], value_name = col1)\
            .dropna(subset=[col1])\
            .drop("variable", axis=1) 



        df2 = df_clean[col2].apply(pd.Series) \
            .merge(df_clean, right_index = True, left_index = True) \
            .drop([col2], axis = 1) \
            .melt(id_vars = df_new.loc[:,df_new.columns!=col2], value_name = col2)\
            .dropna(subset=[col1])\
            .drop("variable", axis=1) 


        daata=pd.DataFrame()
        daata1=pd.DataFrame()
        roles=['Data or business analyst',
        'Data scientist or machine learning specialist',
        'Developer, back-end',
        'Developer, front-end',
        'Developer, full-stack',
        'DevOps specialist',
        'Designer',
        'Developer, embedded applications or devices']
        for role in roles:
            g = df2.groupby(['DevType'])[col2].value_counts().mul(100).unstack().fillna(0)/(df_clean.dropna(subset=[col2])[df_clean['DevType'] ==role]).shape[0]
            daata1['Tools']=g.T[role].sort_values().tail(2).index
            daata1['Category']=g.T[role].sort_values().tail(2).index.name
            daata1['Category']=daata1['Category']
            daata1['role']=role
            daata1['figure']=g.T[role].sort_values().tail(2).values
            daata=daata.append(daata1)
            daata2=pd.DataFrame()
            f = df1.groupby(['DevType'])[col1].value_counts().mul(100).unstack().fillna(0)/(df_clean.dropna(subset=[col1])[df_clean['DevType'] ==role]).shape[0]
            daata2['Tools']=f.T[role].sort_values().tail(2).index
            daata2['Category']=f.T[role].sort_values().tail(2).index.name
            daata2['Category']=daata2['Category']
            daata2['role']=role
            daata2['figure']=f.T[role].sort_values().tail(2).values
            daata=daata.append(daata2)
        
        def generateYAxis(df, x_axis_column, filter_value, y_axis_value):
        ###
            # to avoid cases where there are inconsistencies in the Y axis data (e.g each data group having differing total number of values in the y-axis), we normalize it and fill in the empty values with None
            # filter_value - must be a list
            # filtery_axis_value_value - must be a list
            # the reason for the above 2 restrictions is so that the function can also work for cases where we have multiple entries for a certain category
            ###
            unique_xaxis = df[x_axis_column].unique().tolist()
            yaxis_length = len(unique_xaxis)
            output = [None] * yaxis_length # length of y axis should be the same as that of xaxis unique values
            #get appropriate position for y axis value
            if len(y_axis_value) == len(unique_xaxis):
                return y_axis_value
            filter_value_index = unique_xaxis.index(filter_value[0]) # assume that filter_value contains only 1 element
            output[filter_value_index] = y_axis_value[0]

            return output


        def createDataframeBasedOnColumn(df, column_name, column_value):
        ###
                # returns a new dataframe created based off a certain value of the supplied column
            ###
            return df[df[column_name] == column_value]

        def splitDataframeByColumn(df, column_name):
        ###
            # create new dataframe(s) with rows that have the same value in 'column_name' from the current dataframe
        ###
            data_frames = {}
            unique_column_data = df[column_name].unique().tolist()
            for column_value in unique_column_data:
                data_frames[column_value] = createDataframeBasedOnColumn(df, column_name, column_value)
            return data_frames

        def createGroups(df, grouping_column):
        ###
            # Creates a list out of the column we'd like to group our data by
        ###
         return df[grouping_column].tolist()

        def createDataframesFromGroups(df, grouping_column, groups):
        ###
            # this function creates dataframes for each of the groups we want our chart to be grouped by. In this case we are grouping based on Tools ('PostgreSQL', 'MySQL', 'MongoDB', 'Redis')
            ###
            group_data_frames = {}
            for group in groups:
                group_data_frames[group] = df[df[grouping_column] == group]
            return group_data_frames


        def computeVisibilityList(options_list, total_traces_drawn, trace_indexes):
        ###
            ## dynamically computes visibility for each trace based on the number of options that would be in the dropdown and the total number of traces drawn 
        ###

            buttons_list = [ dict(label="All", method="update", args = [{"visible": [True] * total_traces_drawn, "title": "All"}]) ] ### Initialize button list 

            visibility_list = [False] * total_traces_drawn 

            for option in options_list:
                ## get the start and end index for the traces drawn for each of the options
                trace_positions = trace_indexes[option]
                ### loop through the visibility_list for the current option and set to True only the indexed that are between the start(inclusive) and end(inclusive) indexes for the traces drawn for this option 
                for index in range(trace_positions["start"], trace_positions["end"] + 1):
                 visibility_list[index] = True
                button = dict(label=option, method="update", args = [{"visible": copy.deepcopy(visibility_list), "title": option}]) ## deepcopy just cos of paranoia
                buttons_list.append(button)
                visibility_list = [False] * total_traces_drawn # reset list to avoid unnecessary erros

            return buttons_list


        def addButtonsToFigure(figure, data_frame, total_traces_count, trace_indexes):
            roles = data_frames.keys()
            figure.update_layout(
                updatemenus = [
                    dict(
                        type='dropdown',
                        active = 0,
                        buttons=list(computeVisibilityList(roles, total_traces_count, trace_indexes))
                    )
                ]
            )


        ###
        ########################################################################### START OF PROGRAM #########################################################################################
        ###
        df=daata

        palette = cycle(px.colors.qualitative.Plotly)
        colors = {c:next(palette) for c in df['Tools']}

        fig = go.Figure()

        ### Create Bar Charts
        data_frames = splitDataframeByColumn(df, 'role');
        total_traces_count = 0
        trace_indexes = {}

        for role, data_frame in data_frames.items():
            role_trace_start_and_end_index = { "start": 0, "end": 0, "start_is_recorded": False }
            groups = createGroups(data_frame, 'Tools')
            group_data_frames = createDataframesFromGroups(data_frame, 'Tools', groups)
            categories = data_frame['Category'].unique().tolist()
            for group, group_frame in group_data_frames.items():
                if role_trace_start_and_end_index["start_is_recorded"] == False:
                    role_trace_start_and_end_index["start_is_recorded"] = True
                    role_trace_start_and_end_index["start"] = total_traces_count
                total_traces_count += 1
                y_axis = generateYAxis(data_frame, 'Category', group_frame['Category'].tolist(), group_frame['figure'].tolist())
                fig.add_trace(go.Bar(x=categories, y=y_axis, name="{} - {}".format(role, group), legendgroup=role, text=role,marker_color=colors[group_frame['Tools'].to_list()[0]]))

            role_trace_start_and_end_index["end"] = total_traces_count - 1
            trace_indexes[role] = role_trace_start_and_end_index

        fig.update_layout(barmode="group",
                          legend_font_size= 9,
                          font_color="white",
                          #xaxis_color="white",
                          #yaxis_color="white",
                          paper_bgcolor='rgb(68, 112, 118)',
                          legend_font_color='white',
                          plot_bgcolor= 'rgb(68, 112, 118)',
                          legend=dict(x=-1.2,y=0.7,#bgcolor='LightSteelBlue', bordercolor='LightSteelBlue',

                        xanchor = 'auto'),updatemenus=list([
    dict( x = -0.5,y=1)]))
        ### Add buttons
        addButtonsToFigure(fig, data_frame, total_traces_count, trace_indexes)
        #fig.show()
        figures.append(fig)
        
   

    return  figures

#chart()
