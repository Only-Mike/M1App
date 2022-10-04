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

    hr_df['WorkLifeBalanceID'] = le_WorkLifeBalance.fit_transform(hr_df['WorkLifeBalance'])
    hr_df['EducationID'] = le_Education.fit_transform(hr_df['Education'])


    # construct matrix
    ones = np.ones(len(hr_df), np.uint32)
    matrix = ss.coo_matrix((ones, (hr_df['WorkLifeBalanceID'], hr_df['EducationID'])))

    # decomposition
    svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
    matrix_WorkLifeBalances = svd.fit_transform(matrix)
    matrix_Educations = svd.fit_transform(matrix.T)


    # distance-matrix
    cosine_distance_matrix_Educations = cosine_distances(matrix_Educations)
  

    return hr_df, le_WorkLifeBalance, le_Education, matrix, svd, matrix_WorkLifeBalances, matrix_Educations, cosine_distance_matrix_Educations

hr_df, le_WorkLifeBalance, le_Education, matrix, svd, matrix_WorkLifeBalances, matrix_Educations, cosine_distance_matrix_Educations = read_process_data()


def similar_Education(WorkLifeBalance, n):
  """
  this function performs WorkLifeBalance similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_Education.transform(['Education'])[0]
  sim_WorkLifeBalances = le_Education.inverse_transform(np.argsort(cosine_distance_matrix_Educations[ix,:])[:n+1])
  return sim_WorkLifeBalances[1:]

st.title('Streamlit Recommender')
st.subheader('NB: It seems like the encoder cant recognise the "Education" value for some reason')

One_Education = st.selectbox('Select Education', hr_df.Education.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Something - click!'):
    st.write(similar_Education(One_Education, n_recs_c))


def similar_WorkLifeBalance_Education(WorkLifeBalance, n):
  u_id = le_WorkLifeBalance.transform([WorkLifeBalance])[0]
  Education_ids = hr_df[hr_df.WorkLifeBalanceID == u_id]['EducationID'].unique()
  Education_vector_hr_df = np.mean(matrix_Educations[Education_ids], axis=0)
  closest_for_user = cosine_distances(Education_vector_hr_df.reshape(1,5), matrix_Educations)
  sim_Educations = le_Education.inverse_transform(np.argsort(closest_for_user[0])[:n])
  return sim_Educations

one_user = st.selectbox('Select Education', hr_df.WorkLifeBalance.unique())
if one_user:
    st.write(hr_df[hr_df.Education == one_user]['WorkLifeBalance'].unique())

n_recs_u = st.slider('How many recs? for Education', 1, 20, 2)

if st.button('Recommend Education - click!'):
    similar_cities = similar_WorkLifeBalance_Education(one_user, n_recs_u)
    st.write(similar_cities)

