import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import itertools
import imblearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


st.set_page_config(page_title='Streamlit Cluster', layout = 'wide')
st.header(' Graphs showing clusters in HR data')
st.write('hello world')



