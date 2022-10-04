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


st.set_page_config(page_title='Streamlit Cluster', layout = 'wide')
st.header(' Graphs showing clusters in HR data')
st.write('hello world')

hr = pd.read_csv("data_dictionary.csv")
hr1 = pd.read_csv("employee_survey_data.csv")
hr2 = pd.read_csv("general_data.csv")
hr3 = pd.read_csv("manager_survey_data.csv")

