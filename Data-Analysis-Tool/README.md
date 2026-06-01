# Data Analysis Tool

## Overview

Data Analysis Tool is an object-oriented Python package developed to simplify data preprocessing, exploratory data analysis, and interactive visualization tasks. The package provides a collection of utilities for data inspection, missing value treatment, duplicate removal, outlier detection, feature transformation, correlation analysis, and visualization. It is designed to help users efficiently prepare and explore tabular datasets within Python and Google Colab environments.

---

## Features

### Data Inspection

* Dataset loading and management
* Dataset summary generation
* Column information analysis
* Missing value identification

### Data Cleaning

* Missing value imputation using mean, median, mode, or constant values
* Duplicate row detection and removal
* Outlier detection using the Interquartile Range (IQR) method
* Row and column deletion utilities

### Data Visualization

* Histograms
* Bar charts
* Pie charts
* Scatter plots
* Violin plots
* Correlation heatmaps

### Feature Engineering

* Numerical data normalization

  * Min-Max Scaling
  * Standard Scaling
  * Robust Scaling
* Categorical data encoding

  * One-Hot Encoding
  * Ordinal Encoding

### Statistical Analysis

* Numerical correlation analysis
* Categorical correlation analysis using Cramer's V
* Unified association heatmaps

---

## Project Structure

```text
Data-Analysis-Tool/
│
├── data_analysis/
│   ├── __init__.py
│   └── core.py
│
├── pyproject.toml
├── README.md
├── .gitignore
└── data_analysis_tool.ipynb
```

---



