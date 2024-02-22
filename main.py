import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Streamlit app header
st.title("Jobs and Salaries in Data Science")

# Sidebar header
st.sidebar.header('Menu')

# Sidebar radio
radio_selected_option = st.sidebar.radio("Pick one", ["Dataset", "Missing Values", "Basic Statics", "Some Exploration",
                                                      "Some Charts"])

# Read dataset (this my dataframe)
df = pd.read_csv('jobs_in_data.csv')

# Copy of dataframe
copy_df = df.copy()

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

elif radio_selected_option == "Missing Values":
    # Check for missing values
    st.subheader("Missing values in the dataset:")
    st.table(df.isnull().sum())
    # Display the first 16 rows of the DataFrame containing null values using Streamlit expander
    with st.expander("See first 16 rows that contain null values"):
        st.write(df.head(16))

    # Remove null values from the DataFrame using dropna and display the first 16 rows
    with st.expander("Remove null values using dropna"):
        st.write(copy_df.dropna().head(16))

    # Replace null values in specific columns with the mean of the respective column or default values
    with st.expander("Replace null values with mean of the column using fillna"):
        # Fill null values in 'salary' column with the mean value of the column
        copy_df['salary'].fillna(copy_df['salary'].mean(), inplace=True)

        # Fill null values in 'salary_currency' column with "Unknown"
        copy_df['salary_currency'].fillna("Unknown", inplace=True)

        # Fill null values in 'work_setting' column with "Unknown"
        copy_df['work_setting'].fillna("Unknown", inplace=True)

        # Display the first 16 rows of the DataFrame after null value replacement
        st.write(copy_df.head(16))

elif radio_selected_option == "Basic Statics":
    # Basic statistics of numeric columns
    st.subheader("Basic statistics of numeric columns:")
    st.table(df.describe())

elif radio_selected_option == "Some Exploration":
    # Create two columns
    col1, col2 = st.columns(2)
    # First column
    with col1:
        # Explore job titles and their frequencies
        st.subheader("Job Title Frequencies:")
        st.write(df['job_title'].value_counts())
        # Explore experience levels and their frequencies
        st.subheader("Experience Level Frequencies:")
        st.write(df['experience_level'].value_counts())

    # Second column
    with col2:
        # Explore job categories and their frequencies
        st.subheader("Job Category Frequencies:")
        st.write(df['job_category'].value_counts())
        # Explore employment types and their frequencies
        st.subheader("Employment Type Frequencies:")
        st.write(df['employment_type'].value_counts())

    # Explore company sizes and their frequencies
    st.subheader("Company Size Frequencies:")
    st.write(df['company_size'].value_counts())

elif radio_selected_option == "Some Charts":
    # Pie chart: Explore company sizes and their frequencies
    st.subheader("Pie Chart: Company Size Frequencies")
    employment_type_counts = df['company_size'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(employment_type_counts, labels=employment_type_counts.index, autopct='%1.1f%%',
            colors=['lightcoral', 'lightgreen', 'lightskyblue'])
    plt.title("Company Size Frequencies")
    st.pyplot(plt)
    # Bar chart: Explore job categories and their frequencies
    st.subheader("Bar Chart: Job Category Frequencies")
    job_category_counts = df['job_category'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(job_category_counts.index, job_category_counts.values, color='skyblue')
    plt.title("Job Category Frequencies")
    plt.xlabel("Job Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    # Line chart: Explore experience level and their frequencies
    st.subheader("Line Chart: Experience Level Frequencies")
    plt.figure(figsize=(10, 6))
    experience_level_counts = df['experience_level'].value_counts()
    plt.plot(experience_level_counts.values[::-1], experience_level_counts.index[::-1], marker='o', linestyle='-')
    plt.title("Experience Level Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Experience Level")
    st.pyplot(plt)
    # Another chart with linear regression
    # Selecting only the required columns
    df = df[['work_year', 'salary_in_usd']].dropna()

    # Grouping data by work year and calculating average salary for each year
    avg_salary_by_year = df.groupby('work_year')['salary_in_usd'].mean().reset_index()

    # Creating feature (work year) and target variable (average salary)
    X = avg_salary_by_year['work_year'].values.reshape(-1, 1)
    y = avg_salary_by_year['salary_in_usd'].values

    # Fitting linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predicting average salary for future years
    future_years = np.arange(df['work_year'].min(), df['work_year'].max() + 5).reshape(-1, 1)
    predicted_salaries = model.predict(future_years)

    # Calculating R-squared (R2) score
    r2 = r2_score(y, model.predict(X))
    st.write("R-squared (R2) Score:", r2)

    # Plotting Salary Growth Analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual')
    plt.plot(future_years, predicted_salaries, color='red', linestyle='--', label='Predicted')
    plt.title('Salary Growth Analysis')
    plt.xlabel('Work Year')
    plt.ylabel('Average Salary (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
