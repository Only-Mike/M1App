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
    le_MonthlyIncome = LabelEncoder()
    le_Age = LabelEncoder()

    hr_df['MonthlyIncome'] = le_MonthlyIncome.fit_transform(hr_df['MonthlyIncome'])
    hr_df['Age'] = le_Age.fit_transform(hr_df['Age'])


    # construct matrix
    ones = np.ones(len(hr_df), np.uint32)
    matrix = ss.coo_matrix((ones, (hr_df['MonthlyIncome'], hr_df['Age'])))

    # decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_MonthlyIncome = svd.fit_transform(matrix)
    matrix_Age = svd.fit_transform(matrix.T)


    # distance-matrix
    cosine_distance_matrix_MonthlyIncome = cosine_distances(matrix_MonthlyIncome)
  

    return hr_df, le_MonthlyIncome, le_Age, matrix, svd, matrix_MonthlyIncome, matrix_Age, cosine_distance_matrix_MonthlyIncome

hr_df, le_MonthlyIncome, le_Age, matrix, svd, matrix_MonthlyIncome, matrix_Age, cosine_distance_matrix_MonthlyIncome = read_process_data()


def similar_MonthlyIncome(MonthlyIncome, n):
  """
  this function performs MOnthly Income similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_MonthlyIncome.transform([MonthlyIncome])[0]
  sim_MonthlyIncome = le_MonthlyIncome.inverse_transform(np.argsort(cosine_distance_matrix_MonthlyIncome[ix,:])[:n+1])
  return sim_MonthlyIncome[1:]

st.title('Streamlit Recommender')
    
monthly_income = st.selectbox('Select Monthly Income', hr_df.MonthlyIncome.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Something - click!'):
    st.write(similar_MonthlyIncome(monthly_income, n_recs_c))


def similar_Monthly_Age(Age, n):
  u_id = le_Age.transform([Age])[0]
  Age_ids = hr_df[hr_df.Age == u_id]['MonthlyIncome'].unique()
  Age_vector_hr_df = np.mean(matrix_MonthlyIncome[Age_ids], axis=0)
  closest_for_user = cosine_distances(Age_vector_hr_df.reshape(1,5), matrix_MonthlyIncome)
  sim_MonthlyIncome = le_MonthlyIncome.inverse_transform(np.argsort(closest_for_user[0])[:n])
  return sim_MonthlyIncome

one_user = st.selectbox('Select Age', hr_df.Age.unique())
if one_user:
    st.write(hr_df[hr_df.Age == one_user]['Age'].unique())

n_recs_u = st.slider('How many recs? for Age', 1, 20, 2)

if st.button('Recommend Age - click!'):
    similar_cities = similar_Monthly_Age(one_user, n_recs_u)
    st.write(similar_cities)

