import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from io import BytesIO
import base64
import seaborn as sns
import matplotlib.pyplot as plt

# Function to preprocess the data
def preprocess_data(input_file, imputer_strategy_numerical='median', imputer_strategy_categorical='most_frequent', scaler=True):
    # Read the input file
    if input_file.name.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    elif input_file.name.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Input file format not supported. Only .xlsx and .csv files are supported.")

    st.subheader("Preview of the dataset:")
    st.write(df.head())
    
    # Display data summary
    st.subheader("Dataset Summary:")
    st.write(df.describe())

    # Display data summary
    st.subheader("Dataset Missing Values Preview:")
    st.write(df.isnull().sum())

    # Define preprocessing steps for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', get_numerical_imputer(imputer_strategy_numerical)),
        ('scaler', StandardScaler()) if scaler else ('passthrough', 'passthrough')
    ])

    # Define preprocessing steps for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputer_strategy_categorical)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Define preprocessing for all columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, df.select_dtypes(include='number').columns),
            ('cat', categorical_transformer, df.select_dtypes(exclude='number').columns)
        ])

    # Fit the preprocessor to the data and transform the input file
    transformed_data = preprocessor.fit_transform(df)

    # Convert transformed data to DataFrame
    preprocessed_df = pd.DataFrame(transformed_data, columns=df.columns)

    # Save the preprocessed data to a bytes buffer
    output_buffer = BytesIO()
    preprocessed_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)

    return output_buffer

# Function to get the appropriate imputer based on user choice for numerical columns
def get_numerical_imputer(strategy):
    if strategy == 'mean':
        return SimpleImputer(strategy=strategy)
    elif strategy == 'median':
        return SimpleImputer(strategy=strategy)
    elif strategy == 'most_frequent':
        return SimpleImputer(strategy=strategy)
    elif strategy == 'knn':
        return KNNImputer()
    else:
        raise ValueError("Invalid imputation strategy")

# Function to plot distribution using Seaborn's distplot (KDE plot)
def plot_distplot(data, columns):
    st.subheader("Distribution Plot (Seaborn - distplot)")
    sns.set(style="whitegrid")
    for column in columns:
        plt.figure(figsize=(8, 6))
        sns.distplot(data[column], kde=True, hist=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        st.pyplot()

# Function to plot distribution using Matplotlib's histogram
def plot_histogram(data, columns):
    st.subheader("Distribution Plot (Matplotlib - Histogram)")
    for column in columns:
        plt.figure(figsize=(8, 6))
        plt.hist(data[column], bins=20, alpha=0.7)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot()

# Function to perform Min-Max Scaling
def min_max_scaling(data, columns):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to perform Standardization
def standardization(data, columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to perform Robust Scaling
def robust_scaling(data, columns):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to perform Unit Vector Scaling
def unit_vector_scaling(data, columns):
    scaler = Normalizer()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to perform Max Abs Scaling
def max_abs_scaling(data, columns):
    scaler = MaxAbsScaler()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to perform Quantile Transformation
def quantile_transformation(data, columns):
    scaler = QuantileTransformer()
    scaled_data = scaler.fit_transform(data[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    return scaled_df

# Function to describe the dataset
def describe_dataset(df):
    """
    Generate a point-wise description of the dataset.
    """
    st.subheader("Dataset Description")
    
    # Display the number of rows and columns
    st.write(f"1. The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Display the column names
    st.write("2. The columns in the dataset are as follows:")
    st.write(", ".join(df.columns.tolist()))
    
    # Display the data types of columns
    st.write("3. The data types of columns are as follows:")
    st.write(df.dtypes)
    
    # Display the first few rows of the dataset
    st.write("4. The first few rows of the dataset are as follows:")
    st.write(df.head())
    
    # Display summary statistics
    st.write("5. Summary statistics for numerical columns:")
    st.write(df.describe())

def main():
    st.title("DataBits - The Data Preprocessing Tool")

    # Sidebar options
    st.sidebar.header("Options")
    operation = st.sidebar.radio("Select Operation", ["Missing Values Imputation", "Data Normalization", "Dataset Description"])

    if operation == "Missing Values Imputation":
        st.subheader("Missing Values Imputation")
        uploaded_file = st.file_uploader("Upload File", type=["xlsx", "csv"])
        if uploaded_file is not None:
            imputer_strategy_numerical = st.selectbox("Select imputation strategy for numerical columns:", ['mean', 'median', 'most_frequent', 'knn'])
            scaler = st.checkbox("Scale numeric features?")
            imputer_strategy_categorical = st.selectbox("Select imputation strategy for categorical columns:", ['most_frequent'])
            if st.button('Preprocess Data'):
                preprocessed_data = preprocess_data(uploaded_file, imputer_strategy_numerical, imputer_strategy_categorical, scaler)
                if preprocessed_data:
                    csv = preprocessed_data.getvalue()
                    b64 = base64.b64encode(csv).decode()  # B64 encoding
                    href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Download Preprocessed Data</a>'
                    st.markdown(href, unsafe_allow_html=True)

    elif operation == "Data Normalization":
        st.subheader("Data Normalization")
        uploaded_file = st.file_uploader("Upload File", type=["xlsx", "csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)  # Assume data is numeric for normalization
            numerical_columns = data.select_dtypes(include=['int', 'float']).columns.tolist()
            selected_columns = st.multiselect("Select columns to plot:", numerical_columns)
            plot_distribution = st.checkbox("Plot Distribution")
            if plot_distribution:
                plot_type = st.radio("Select plot type:", ("Histogram", "Distplot (KDE)"))
                if plot_type == "Histogram":
                    plot_histogram(data, selected_columns)
                elif plot_type == "Distplot (KDE)":
                    plot_distplot(data, selected_columns)
            
            st.subheader("Normalization Techniques:")
            min_max_checkbox = st.checkbox("Min-Max Scaling (Normalization)")
            standardization_checkbox = st.checkbox("Standardization (Z-score normalization)")
            robust_scaling_checkbox = st.checkbox("Robust Scaling")
            unit_vector_scaling_checkbox = st.checkbox("Unit Vector Scaling")
            max_abs_scaling_checkbox = st.checkbox("Max Abs Scaling")
            quantile_transformation_checkbox = st.checkbox("Quantile Transformation")

            if st.button("Apply Normalization"):
                if min_max_checkbox:
                    data[selected_columns] = min_max_scaling(data, selected_columns)
                if standardization_checkbox:
                    data[selected_columns] = standardization(data, selected_columns)
                if robust_scaling_checkbox:
                    data[selected_columns] = robust_scaling(data, selected_columns)
                if unit_vector_scaling_checkbox:
                    data[selected_columns] = unit_vector_scaling(data, selected_columns)
                if max_abs_scaling_checkbox:
                    data[selected_columns] = max_abs_scaling(data, selected_columns)
                if quantile_transformation_checkbox:
                    data[selected_columns] = quantile_transformation(data, selected_columns)
                
                if plot_distribution:
                    if plot_type == "Histogram":
                        plot_histogram(data, selected_columns)
                    elif plot_type == "Distplot (KDE)":
                        plot_distplot(data, selected_columns)

    elif operation == "Dataset Description":
        st.subheader("Dataset Description")
        uploaded_file = st.file_uploader("Upload File", type=["xlsx", "csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)  # Assume data is numeric for normalization
            describe_dataset(df)

if __name__ == "__main__":
    main()
