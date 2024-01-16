import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app header
st.title("Jobs and Salaries in Data Science")

# Sidebar header
st.sidebar.header('Menu')

# Sidebar radio
radio_selected_option = st.sidebar.radio("Pick one", ["Dataset", "Basic Statics", "Missing Values", "Some Exploration"])

# Read dataset (this my dataframe)
df = pd.read_csv('jobs_in_data.csv')

# Show widgets according to selected option from sidebar radio
if radio_selected_option == "Dataset":
    # Select box to show dataset according to options
    dataset_select_box = st.selectbox("Pick one", ["First 5 rows of the dataset", "Whole dataset"])

    # Check which option the user selected and perform corresponding actions.
    if dataset_select_box == "First 5 rows of the dataset":
        # If the user chose "First 5 rows of the dataset," display a subheader and show the first 5 rows of the dataset.
        st.subheader("First 5 rows of the dataset:")
        st.write(df.head())
    elif dataset_select_box == "Whole dataset":
        # If the user chose "Whole dataset," display a subheader and show
        st.subheader("Whole dataset:")
        st.write(df)
