import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 🏷 **Title of Dashboard**
st.title("📢 Anomaly Detection Machine Learning")

# Load dataset
DATA_PATH = "machine_failure_dataset.csv"

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Ensure data integrity (Checking first few rows)
    st.write("🔍 Data Preview:", df.head())

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Ensure all machine types exist after encoding
    machine_types = ["Drill", "Mill", "Lathe"]
    df = pd.get_dummies(df, columns=["Machine_Type"], drop_first=False)

    for m in machine_types:
        col_name = f"Machine_Type_{m}"
        if col_name not in df.columns:
            df[col_name] = 0  # Add missing columns if not present
    
    return df

df = load_and_preprocess_data()

# Function to classify failure risk
def classify_failure_risk(df):
    df["Failure_Status"] = df["Failure_Risk"].map({0: "No Failure", 1: "Failure Risk"})
    return df

df = classify_failure_risk(df)

# Function to detect anomalies using Isolation Forest
def detect_anomalies(df, feature_columns):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(df[feature_columns])
    df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})
    return df

df = detect_anomalies(df, ["Temperature", "Vibration", "Power_Usage", "Humidity"])

# 🏷 **Sidebar Heading Updated**
st.sidebar.header("📊 Data")

# Sidebar user selection
selected_data = st.sidebar.selectbox("📌 Select Data Type:", df.columns[1:])

st.sidebar.header("📊 Select Chart Type")
chart_type = st.sidebar.selectbox("Choose chart type:", ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart"])

st.sidebar.header("🔍 Select Anomaly Detection Method")
model_type = st.sidebar.selectbox("Choose Model:", ["Isolation Forest", "Logistic Regression", "Random Forest"])

st.sidebar.header("⚙ Select Machine Type for Anomalies")
machine_type = st.sidebar.selectbox("Select Machine:", ["Drill", "Mill", "Lathe"])

# Ensure selected machine type exists
if f"Machine_Type_{machine_type}" in df.columns:
    machine_df = df[df[f"Machine_Type_{machine_type}"] == 1]
else:
    st.warning(f"No data available for {machine_type}")
    machine_df = pd.DataFrame()

# 📌 **Verify Data Matches Graph**
st.write("✅ Dataset Shape:", df.shape)
st.write("🔍 Checking First 5 Rows After Processing:", df.head())

# Layout with two charts dynamically
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 {selected_data} Over Time")
    if chart_type == "Line Chart":
        fig = px.line(df, x=df.index, y=selected_data, title=f"{selected_data} Over Time")
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=df.index, y=selected_data, title=f"{selected_data} Over Time")
    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=df.index, y=selected_data, title=f"{selected_data} Over Time")
    else:
        fig = px.pie(df, names="Failure_Status", title="Failure Risk Proportion")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📍 Anomaly Detection")
    fig = px.scatter(df, x=df.index, y=selected_data, color="Anomaly", title=f"Anomalies in {selected_data}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📌 Failure Risk Analysis")
fig = px.bar(df, x=df.index, y=selected_data, color="Failure_Status",
             title="Failure Risk Highlighted", color_discrete_map={"No Failure": "blue", "Failure Risk": "red"})
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
corr_matrix = df.select_dtypes(include=['number']).corr()
fig = px.imshow(corr_matrix, labels=dict(color="Correlation"), color_continuous_scale="RdBu_r", title="Sensor Correlation")
st.plotly_chart(fig, use_container_width=True)

# Pairplot & Histogram for Selected Machine Anomalies
if not machine_df.empty:
    st.subheader(f"📊 Pairplot for {machine_type} Anomalies")

    sns.set(style="ticks")
    num_cols = machine_df.select_dtypes(include=['number']).columns
    df_pairplot = machine_df[num_cols]

    pairplot_fig = sns.pairplot(df_pairplot, hue='Failure_Risk', palette="coolwarm")
    pairplot_fig.savefig("pairplot.png")
    st.image("pairplot.png", use_container_width=True)

    # Histogram of Anomalies for Machine Type
    st.subheader(f"📊 Anomalies in {machine_type} Under Different Conditions")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    features = ["Temperature", "Humidity", "Vibration", "Power_Usage"]
    colors = {0: "blue", 1: "red"}

    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        for risk in [0, 1]:
            subset = machine_df[machine_df["Failure_Risk"] == risk]
            sns.histplot(subset[feature], bins=20, kde=True, color=colors[risk], ax=ax, label=f"Risk {risk}")
        ax.set_title(f"{feature} Distribution")
        ax.legend()

    plt.tight_layout()
    plt.savefig("anomaly_histograms.png")
    st.image("anomaly_histograms.png", use_container_width=True)

st.write("✅ Interactive Machine Failure Analysis Dashboard")  
