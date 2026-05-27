import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# IRIS SPECIES CLASSIFICATION - DATA MINING FINAL PROJECT
# Universidad de la Costa
# ============================================================

st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# 1. Load dataset
# -----------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    df["species_id"] = iris.target
    df["species"] = df["species_id"].map({
        0: "Iris setosa",
        1: "Iris versicolor",
        2: "Iris virginica"
    })
    return df, iris


df, iris = load_data()

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species_id"]

# -----------------------------
# 2. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------
# 3. Model training
# -----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=4
    )
    model.fit(X_train, y_train)
    return model


model = train_model(X_train, y_train)

# -----------------------------
# 4. Model evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# -----------------------------
# Dashboard title
# -----------------------------
st.title("🌸 Iris Species Classification")
st.subheader("Final Project - Data Mining")

st.markdown("""
This dashboard presents a complete machine learning workflow for classifying Iris flowers into three species:
**Iris setosa**, **Iris versicolor**, and **Iris virginica**.

The model uses four numerical variables:

- Sepal length
- Sepal width
- Petal length
- Petal width
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Input flower measurements")

sepal_length = st.sidebar.slider(
    "Sepal length (cm)",
    float(df["sepal_length"].min()),
    float(df["sepal_length"].max()),
    float(df["sepal_length"].mean())
)

sepal_width = st.sidebar.slider(
    "Sepal width (cm)",
    float(df["sepal_width"].min()),
    float(df["sepal_width"].max()),
    float(df["sepal_width"].mean())
)

petal_length = st.sidebar.slider(
    "Petal length (cm)",
    float(df["petal_length"].min()),
    float(df["petal_length"].max()),
    float(df["petal_length"].mean())
)

petal_width = st.sidebar.slider(
    "Petal width (cm)",
    float(df["petal_width"].min()),
    float(df["petal_width"].max()),
    float(df["petal_width"].mean())
)

new_sample = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

prediction_id = model.predict(new_sample)[0]
prediction_name = {
    0: "Iris setosa",
    1: "Iris versicolor",
    2: "Iris virginica"
}[prediction_id]

prediction_probability = model.predict_proba(new_sample)[0]
confidence = np.max(prediction_probability)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Model metrics",
    "Prediction panel",
    "3D visualization",
    "Data exploration"
])

# -----------------------------
# Tab 1: Metrics
# -----------------------------
with tab1:
    st.header("Model performance metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    st.markdown("""
    The model was trained using a **Random Forest Classifier**. This algorithm was selected because it performs well in classification problems,
    handles non-linear relationships, and is robust with small datasets such as Iris.
    """)

    st.subheader("Classification report")
    report = classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=iris.target_names,
        columns=iris.target_names
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(x="Predicted species", y="Actual species", color="Count"),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------
# Tab 2: Prediction panel
# -----------------------------
with tab2:
    st.header("Predict a new Iris flower")

    st.markdown("""
    Use the sliders in the left panel to enter the measurements of a new flower.
    The trained model will predict the most likely Iris species.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input values")
        st.dataframe(new_sample)

    with col2:
        st.subheader("Prediction result")
        st.success(f"Predicted species: **{prediction_name}**")
        st.info(f"Model confidence: **{confidence:.2%}**")

        probability_df = pd.DataFrame({
            "Species": ["Iris setosa", "Iris versicolor", "Iris virginica"],
            "Probability": prediction_probability
        })

        fig_prob = px.bar(
            probability_df,
            x="Species",
            y="Probability",
            title="Prediction probabilities",
            text_auto=".2%"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# -----------------------------
# Tab 3: 3D scatter plot
# -----------------------------
with tab3:
    st.header("3D scatter plot with new sample")

    st.markdown("""
    The 3D chart shows the position of the new sample compared with the original Iris dataset.
    This helps visualize whether the predicted species is consistent with the distribution of known flowers.
    """)

    fig_3d = px.scatter_3d(
        df,
        x="petal_length",
        y="petal_width",
        z="sepal_length",
        color="species",
        symbol="species",
        title="Iris dataset and new predicted sample",
        labels={
            "petal_length": "Petal length",
            "petal_width": "Petal width",
            "sepal_length": "Sepal length"
        }
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=[petal_length],
            y=[petal_width],
            z=[sepal_length],
            mode="markers",
            marker=dict(size=10, color="black", symbol="diamond"),
            name=f"New sample: {prediction_name}"
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------
# Tab 4: Data exploration
# -----------------------------
with tab4:
    st.header("Exploratory data analysis")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10))

    st.subheader("Statistical summary")
    st.dataframe(df.describe())

    st.subheader("Histograms by feature")
    selected_feature = st.selectbox(
        "Select a feature",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    fig_hist = px.histogram(
        df,
        x=selected_feature,
        color="species",
        marginal="box",
        title=f"Distribution of {selected_feature} by species"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Scatter matrix")
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        color="species",
        title="Scatter matrix of Iris features"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
**Team members:**  
- José Miguel Carmona González  
- Camilo Martinez

**Course:** Data Mining  
**University:** Universidad de la Costa
""")
