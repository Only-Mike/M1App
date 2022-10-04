import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

#Import datasets
hr = pd.read_csv("data_dictionary.csv")
hr1 = pd.read_csv("employee_survey_data.csv")
hr2 = pd.read_csv("general_data.csv")
hr3 = pd.read_csv("manager_survey_data.csv")

# Merge datasets
mergehr = pd.merge(hr1, hr2)
hr_df = pd.merge(mergehr, hr3)

#Replacing value with fillna with the rounded up to the closest integer
#We know this from the EDA Stuff 

hr_df.NumCompaniesWorked = hr_df.NumCompaniesWorked.fillna(3)
hr_df.EnvironmentSatisfaction = hr_df.EnvironmentSatisfaction.fillna(3)
hr_df.JobSatisfaction = hr_df.JobSatisfaction.fillna(3)
hr_df.WorkLifeBalance = hr_df.WorkLifeBalance.fillna(3)
hr_df.TotalWorkingYears = hr_df.TotalWorkingYears.fillna(11) 




st.set_page_config(page_title='Streamlit EDA', layout = 'wide')
st.header('EDA in HR dataset')

st.text('First we import all the datasets')
code = '''
hr = pd.read_csv("data_dictionary.csv")
hr1 = pd.read_csv("employee_survey_data.csv")
hr2 = pd.read_csv("general_data.csv")
hr3 = pd.read_csv("manager_survey_data.csv")'''

st.code(code, language='python')

st.text('Then we merge them with the pandas.merge function')
code = '''
mergehr = pd.merge(hr1, hr2)
hr_df = pd.merge(mergehr, hr3)
'''
st.code(code, language='python')

st.text('Then we fill out na within the dataset with the closest mean integer')

code = '''
hr_df.NumCompaniesWorked = hr_df.NumCompaniesWorked.fillna(3)
hr_df.EnvironmentSatisfaction = hr_df.EnvironmentSatisfaction.fillna(3)
hr_df.JobSatisfaction = hr_df.JobSatisfaction.fillna(3)
hr_df.WorkLifeBalance = hr_df.WorkLifeBalance.fillna(3)
hr_df.TotalWorkingYears = hr_df.TotalWorkingYears.fillna(11) '''

st.code(code, language='python')
