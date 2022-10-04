import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# Cluster 2 ❄️")
    st.sidebar.markdown("# Cluster 2 ❄️")


page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

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
    le_Department = LabelEncoder()
    le_EducationField = LabelEncoder()

    hr_df['Department'] = le_Department.fit_transform(hr_df['Department'])
    hr_df['EducationField'] = le_EducationField.fit_transform(hr_df['EducationField'])


    # construct matrix
    ones = np.ones(len(hr_df), np.uint32)
    matrix = ss.coo_matrix((ones, (hr_df['Department'], hr_df['EducationField'])))

    # decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_Department = svd.fit_transform(matrix)
    matrix_EducationField = svd.fit_transform(matrix)


    # distance-matrix
    cosine_distance_matrix_Department = cosine_distances(matrix_Department)
  

    return hr_df, le_Department, le_EducationField, matrix, svd, matrix_Department, matrix_EducationField, cosine_distance_matrix_Department

hr_df, le_Department, le_EducationField, matrix, svd, matrix_Department, matrix_EducationField, cosine_distance_matrix_Department = read_process_data()


def similar_Department(Department, n):
  """
  this function performs Department similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_Department.transform([Department])[0]
  sim_Department = le_Department.inverse_transform(np.argsort(cosine_distance_matrix_Department[ix,:])[:n+1])
  return sim_Department[1:]

st.title('Streamlit Recommender')
    
department = st.selectbox('Select Department', hr_df.Department.unique())
n_recs_c = st.slider('How many recs?', 1, 20, 2)

if st.button('Recommend Something - click!'):
    st.write(similar_Department(department, n_recs_c))


def similar_Department_EducationField(EducationField, n):
  u_id = le_EducationField.transform([EducationField])[0]
  EducationField_ids = hr_df[hr_df.EducationField == u_id]['Department'].unique()
  EducationField_vector_hr_df = np.mean(matrix_Department[EducationField_ids], axis=0)
  closest_for_user = cosine_distances(EducationField_vector_hr_df.reshape(1,5), matrix_Department)
  sim_Department = le_Department.inverse_transform(np.argsort(closest_for_user[0])[:n])
  return sim_Department

one_user = st.selectbox('Select EducationField', hr_df.EducationField.unique())
if one_user:
    st.write(hr_df[hr_df.EducationField == one_user]['EducationField'].unique())

n_recs_u = st.slider('How many recs? for EducationField', 1, 20, 2)

if st.button('Recommend EducationField - click!'):
    similar_cities = similar_Department_EducationField(one_user, n_recs_u)
    st.write(similar_cities)

