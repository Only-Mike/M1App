import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

#Set page layout/header
st.set_page_config(page_title='Streamlit Cluster', layout = 'wide')
st.header(' Graphs showing clusters in HR data')

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

# Now to the UML Part

#Clean and get the data ready
hr_df.Attrition.replace(('Yes', 'No'), (1, 0), inplace=True)
hr_df.Gender.replace(('Female', 'Male'), (1, 0), inplace=True)
hr_df.drop(['BusinessTravel', 'Department', 'Over18', 'StandardHours', 'EmployeeCount', 'EducationField', 'JobRole', 'MaritalStatus'], axis = "columns", inplace=True)

#Scale and fit data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

hr_df_scaled = scaler.fit_transform(hr_df)
from sklearn.preprocessing import MinMaxScaler
scaler_min_max = MinMaxScaler()
hr_df_to_cluster_minmax = scaler_min_max.fit_transform(hr_df)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_reduced_pca = pca.fit_transform(hr_df_scaled)
print(pca.explained_variance_ratio_)

import altair as alt
vis_data = pd.DataFrame(data_reduced_pca)
vis_data['Attrition'] = hr_df['Attrition']
vis_data['MonthlyIncome'] = hr_df['MonthlyIncome']
vis_data.columns = ['x', 'y', 'Attrition', 'MonthlyIncome']

st.header('PCA Graph')
c = alt.Chart(vis_data).mark_circle(size = 60).encode(
    x='x', y='y', color = 'Gender', tooltip=['Attrition', 'MonthlyIncome'])

st.altair_chart(c, use_container_width=False)

st.header('UMAP Graphs')
st.subheader('Because of compat problems with UMAP, we have put in pictures on streamlit instead.')
st.text('On the x-axis we find the Gender values and on the y-axis we find the Monthly income ')
st.text('The two graphs done by the UMAP model clearly shows clustering in regards to both Attrition and Gender.')

from PIL import Image
image1 = Image.open('AttritionUMAP.png')
image2 = Image.open('GenderUMAP.png')
image3 = Image.open('PerformanceRating.png')

st.image(image1, caption="UMAP Cluster - Colored by Attrition", output_format="auto")

st.image(image2, caption="UMAP Cluster - Colored by gender", output_format="auto")

st.image(image3, caption="UMAP Cluster - Colored by Performance rating", output_format="auto")

st.text('The two graphs done by the UMAP model clearly shows clustering in regards to both Attrition, Gender and PerformanceRating.')
st.text('For some reason the PerformanceRating only shows the employees given the performance rating 3 and 4')

