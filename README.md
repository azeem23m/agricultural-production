# Agricultural Production Optimization with Dash

## Overview
This project is an interactive web application built using **Dash** that leverages machine learning models (Random Forest, SVM, XGBoost) and clustering algorithms (KMeans) to optimize agricultural production. The app visualizes key insights, evaluates classification models, and explores the data using clustering techniques. 

It is designed to help agricultural businesses make data-driven decisions and improve efficiency by analyzing agricultural data.

## Features
- **Data Visualizations:**
  - Demonstration relationship between features and each other.
  - Visualizion the distribution of each feature.

- **Classification Models:**
  - Applied classification models to predict crop by its growth conditions.
  - Comparison of **Random Forest**, **SVM**, and **XGBoost** classifiers by Precision, Recall and F1 Score.

- **Clustering Models:**
  - **KMeans** clustering with visualization using PCA to reduce dimensionality to 3D.
  - Analysis and description each cluster by extracting key characteristics.

- **Interactive Interface:**
  - Select models, view metrics, and interact with visualizations.
## Live Demo
[Website](https://azeem23m.pythonanywhere.com/)

## Local Installation & Usage

To set up and run the project locally:
   ```bash
   git clone https://github.com/azeem23m/agricultural-production
   pip install requirements.txt
   cd agricultural-production/src
   python main.py
   ```
Open <http://127.0.0.1:8050/> in your web browser
