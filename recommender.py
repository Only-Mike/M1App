import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(page_title='Streamlit HR recommender', layout = 'wide')

@st.experimental_singleton
def read_process_data():
    # Import datasets
    hr = pd.read_csv("data_dictionary.csv")
    hr1 = pd.read_csv("employee_survey_data.csv")
    hr2 = pd.read_csv("general_data.csv")
    hr3 = pd.read_csv("manager_survey_data.csv")

    # Merge datasets
    mergehr = pd.merge(hr1, hr2)
    hr_df = pd.merge(mergehr, hr3)
    hr_df.info() #Controlling if they have merged correctly


    #Replacing value with fillna with the rounded up to the closest integer 

    hr_df.NumCompaniesWorked = hr_df.NumCompaniesWorked.fillna(3)
    hr_df.EnvironmentSatisfaction = hr_df.EnvironmentSatisfaction.fillna(3)
    hr_df.JobSatisfaction = hr_df.JobSatisfaction.fillna(3)
    hr_df.WorkLifeBalance = hr_df.WorkLifeBalance.fillna(3)
    hr_df.TotalWorkingYears = hr_df.TotalWorkingYears.fillna(11) 


    # encode ids
    le_WorkLifeBalance = LabelEncoder()
    le_Education = LabelEncoder()

    hr_df['WorkLifeBalance'] = le_WorkLifeBalance.fit_transform(hr_df['Work_Life_Balance'])
    hr_df['Education'] = le_Education.fit_transform(hr_df['education1'])


    # construct matrix
    ones = np.ones(len(hr_df), np.uint32)
    matrix = ss.coo_matrix((ones, (hr_df['WorkLifeBalance'], hr_df['Education'])))

    # decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_WorkLifeBalances = svd.fit_transform(matrix)
    matrix_Educations = svd.fit_transform(matrix)


    # distance-matrix
    cosine_distance_matrix_matrix_Educations = cosine_distances(matrix_Educations)
  

    return hr_df, le_WorkLifeBalance, le_Education, matrix, svd, matrix_WorkLifeBalance, matrix_Education, cosine_distance_matrix_WorkLifeBalance

hr_df, le_WorkLifeBalance, le_Education, matrix, svd, matrix_WorkLifeBalance, matrix_Education, cosine_distance_matrix_WorkLifeBalance = read_process_data()


def similar_WorkLifeBalance(WorkLifeBalance, n):
  """
  this function performs WorkLifeBalance similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_WorkLifeBalance.transform(['WorkLifeBalance'])[0]
  sim_WorkLifeBalance = le_WorkLifeBalance.inverse_transform(np.argsort(cosine_distance_matrix_WorkLifeBalance[ix,:])[:n+1])
  return sim_WorkLifeBalance[1:]

st.title('Streamlit Recommender')
    
WorkLifeBalance = st.selectbox('Select education1', hr_df.education1.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Something - click!'):
    st.write(similar_WorkLifeBalance(WorkLifeBalance, n_recs_c))


def similar_WorkLifeBalance_Education(Work_Life_Balance, n):
  u_id = le_Education.transform([Work_Life_Balance])[0]
  Education_ids = hr_df[hr_df.Education == u_id]['WorkLifeBalance'].unique()
  Education_vector_hr_df = np.mean(matrix_WorkLifeBalance[Education_ids], axis=0)
  closest_for_user = cosine_distances(Education_vector_hr_df.reshape(1,5), matrix_WorkLifeBalance)
  sim_WorkLifeBalance = le_WorkLifeBalance.inverse_transform(np.argsort(closest_for_user[0])[:n])
  return sim_WorkLifeBalance

one_user = st.selectbox('Select Education', hr_df.Work_Life_Balance.unique())
if one_user:
    st.write(hr_df[hr_df.Education == one_user]['education1'].unique())

n_recs_u = st.slider('How many recs? for Education', 1, 20, 2)

if st.button('Recommend Education - click!'):
    similar_cities = similar_WorkLifeBalance_Education(one_user, n_recs_u)
    st.write(similar_cities)

