import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="K-Means Clustering",
    layout="wide"
)

st.title("K-Means Clustering Visualization")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(
        include=['int64','float64']
    ).columns

    x_col = st.selectbox(
    "Select Feature 1",
    numerical_cols,
    index=0
)

    y_col = st.selectbox(
    "Select Feature 2",
    numerical_cols,
    index=1 if len(numerical_cols) > 1 else 0
)

    n_clusters = st.slider(
        "Number of Clusters",
        2,
        10,
        5
    )

    X = df[[x_col, y_col]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    df['Cluster'] = labels

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Scatter Plot
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df['Cluster']
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("K-Means Clusters")

    st.pyplot(fig)

    # Cluster Count
    st.subheader("Cluster Distribution")

    cluster_count = df['Cluster'].value_counts()

    fig2, ax2 = plt.subplots()

    ax2.bar(
        cluster_count.index.astype(str),
        cluster_count.values
    )

    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)
