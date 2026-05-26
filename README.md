# Iris Species Classification

## Final Project - Data Mining

This project develops a complete data mining workflow to classify Iris flowers into three species:

- Iris setosa
- Iris versicolor
- Iris virginica

The classification is based on four numerical variables:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Objective

The objective of this project is to train a classification model capable of predicting the species of an Iris flower using its physical measurements.  
The project also includes an interactive dashboard developed with Streamlit to present the workflow, metrics, predictions and visualizations.

## Methodology

The project follows a basic data mining pipeline:

1. **Data understanding**  
   The Iris dataset is loaded and explored to identify its features, target classes and general structure.

2. **Data preprocessing**  
   The dataset is divided into independent variables and target labels.  
   A train/test split is applied to evaluate the model with unseen data.

3. **Model training**  
   A Random Forest Classifier is used because it is robust, easy to interpret and suitable for classification tasks with structured data.

4. **Model evaluation**  
   The model is evaluated using the following metrics:

   - Accuracy
   - Precision
   - Recall
   - F1 Score

5. **Dashboard development**  
   A Streamlit dashboard is created to display model metrics, exploratory visualizations and an interactive prediction panel.

## Model Used

The selected model is **Random Forest Classifier**.

This model was selected because:

- It performs well in classification problems.
- It can handle non-linear relationships.
- It is robust with small datasets.
- It provides reliable results without requiring complex preprocessing.

## Dashboard Features

The Streamlit dashboard includes:

- Model metrics: Accuracy, Precision, Recall and F1 Score.
- Classification report.
- Confusion matrix.
- Interactive prediction panel.
- 3D scatter plot showing the new sample compared with the original dataset.
- Histograms by feature.
- Scatter matrix for exploratory analysis.

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_LINK
cd YOUR_REPOSITORY_FOLDER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run Proyect.py
```

## Project Structure

```text
.
├── Proyect.py
├── requirements.txt
└── README.md
```

## Deliverables

- Streamlit dashboard.
- GitHub repository.
- Video presentation link.
- README file with instructions.
- Team members' names.

## Team Members

- José Miguel Carmona González
- Camilo Martinez

## Course Information

**Course:** Data Mining  
**University:** Universidad de la Costa  
**Project:** Iris Species Classification
