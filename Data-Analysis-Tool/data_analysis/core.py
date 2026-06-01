import pandas as pd
import numpy as np
import io

from google.colab import files

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder
)

from scipy.stats import chi2_contingency


# =====================================================
# PlottingMethods Class
# =====================================================

class PlottingMethods:

    def plot_histogram(self, x, data, title="Histogram"):

        fig = px.histogram(
            data,
            x=x,
            title=title
        )

        fig.show()

    def plot_bar_chart(self, x, y, data, title="Bar Chart"):

        fig = px.bar(
            data,
            x=x,
            y=y,
            title=title
        )

        fig.show()

    def plot_pie_chart(self, names, values, data, title="Pie Chart"):

        fig = px.pie(
            data,
            names=names,
            values=values,
            title=title
        )

        fig.show()


# =====================================================
# DataInspector Class
# =====================================================

class DataInspector:

    def __init__(self):

        self.df = None

    # -------------------------------------------------
    # Data Upload
    # -------------------------------------------------

    def upload_data(self):

        uploaded = files.upload()

        file_name = list(uploaded.keys())[0]

        self.df = pd.read_csv(
            io.BytesIO(uploaded[file_name]),
            na_values=["?", "NULL", "null", "n/a", "N/A"]
        )

        print("Dataset loaded successfully")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def get_summary(self):

        print("\nDataset Shape")
        print(self.df.shape)

        print("\nFirst 10 Rows")
        print(self.df.head(10))

        print("\nData Types")
        print(self.df.dtypes)

    def column_details(self):

        details = pd.DataFrame({
            "DataType": self.df.dtypes,
            "Missing Values": self.df.isnull().sum(),
            "Unique Values": self.df.nunique()
        })

        print(details)

    # -------------------------------------------------
    # Missing Values
    # -------------------------------------------------

    def show_missing_data(self):

        print(self.df.isnull().sum())

    def handle_missing_values(
        self,
        strategy="mean",
        constant_value=0
    ):

        numeric_cols = self.df.select_dtypes(
            include=np.number
        ).columns

        categorical_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns

        if strategy == "mean":

            for col in numeric_cols:

                self.df[col].fillna(
                    self.df[col].mean(),
                    inplace=True
                )

        elif strategy == "median":

            for col in numeric_cols:

                self.df[col].fillna(
                    self.df[col].median(),
                    inplace=True
                )

        elif strategy == "mode":

            for col in categorical_cols:

                self.df[col].fillna(
                    self.df[col].mode()[0],
                    inplace=True
                )

        elif strategy == "constant":

            self.df.fillna(
                constant_value,
                inplace=True
            )

        print("Missing values handled")

    # -------------------------------------------------
    # Duplicate Removal
    # -------------------------------------------------

    def remove_duplicates(self):

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        print(
            f"Removed {before - after} duplicate rows"
        )

    # -------------------------------------------------
    # Outlier Detection
    # -------------------------------------------------

    def handle_outliers(
        self,
        columns,
        find_and_delete=False
    ):

        for col in columns:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[col] < lower)
                |
                (self.df[col] > upper)
            ]

            print(f"\nOutliers in {col}")
            print(outliers)

            if find_and_delete:

                self.df = self.df[
                    (self.df[col] >= lower)
                    &
                    (self.df[col] <= upper)
                ]

        print("Outlier analysis complete")

    # -------------------------------------------------
    # Numerical Plots
    # -------------------------------------------------

    def plot_numerical(
        self,
        column_names
    ):

        for col in column_names:

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=[
                    "Violin Plot",
                    "Scatter Plot",
                    "Histogram"
                ]
            )

            fig.add_trace(
                go.Violin(
                    y=self.df[col]
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    y=self.df[col],
                    mode="markers"
                ),
                row=1,
                col=2
            )

            fig.add_trace(
                go.Histogram(
                    x=self.df[col]
                ),
                row=1,
                col=3
            )

            fig.update_layout(
                title=f"{col} Analysis"
            )

            fig.show()

    # -------------------------------------------------
    # Categorical Plots
    # -------------------------------------------------

    def plot_categorical(
        self,
        column_names
    ):

        for col in column_names:

            counts = self.df[col].value_counts()

            fig = px.bar(
                x=counts.index,
                y=counts.values,
                title=f"{col} Distribution"
            )

            fig.show()

    # -------------------------------------------------
    # Relationship Plot
    # -------------------------------------------------

    def plot_relationship(
        self,
        col1,
        col2
    ):

        type1 = pd.api.types.is_numeric_dtype(
            self.df[col1]
        )

        type2 = pd.api.types.is_numeric_dtype(
            self.df[col2]
        )

        if type1 and type2:

            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline="ols"
            )

        elif (not type1) and type2:

            fig = px.box(
                self.df,
                x=col1,
                y=col2,
                points="all"
            )

        else:

            grouped = self.df.groupby(
                [col1, col2]
            ).size().reset_index(name="Count")

            fig = px.bar(
                grouped,
                x=col1,
                y="Count",
                color=col2,
                barmode="group"
            )

        fig.show()

    # -------------------------------------------------
    # Numerical Normalization
    # -------------------------------------------------

    def extract_normalized_numeric_data(
        self,
        method="minmax"
    ):

        numeric_df = self.df.select_dtypes(
            include=np.number
        )

        if method == "minmax":

            scaler = MinMaxScaler()

        elif method == "standard":

            scaler = StandardScaler()

        else:

            scaler = RobustScaler()

        scaled = scaler.fit_transform(
            numeric_df
        )

        return pd.DataFrame(
            scaled,
            columns=numeric_df.columns
        )

    # -------------------------------------------------
    # Categorical Encoding
    # -------------------------------------------------

    def extract_normalized_categorical_data(
        self,
        method="onehot"
    ):

        cat_df = self.df.select_dtypes(
            exclude=np.number
        )

        if method == "onehot":

            return pd.get_dummies(cat_df)

        elif method == "ordinal":

            encoder = OrdinalEncoder()

            encoded = encoder.fit_transform(
                cat_df
            )

            return pd.DataFrame(
                encoded,
                columns=cat_df.columns
            )

    # -------------------------------------------------
    # Combined Dataset
    # -------------------------------------------------

    def create_normalized_data_df(self):

        numeric = self.extract_normalized_numeric_data()

        categorical = (
            self.extract_normalized_categorical_data()
        )

        final_df = pd.concat(
            [numeric, categorical],
            axis=1
        )

        return final_df

    # -------------------------------------------------
    # Numerical Correlation
    # -------------------------------------------------

    def plot_numerical_correlation(self):

        corr = self.df.corr(
            numeric_only=True
        )

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Numerical Correlation"
        )

        fig.show()

    # -------------------------------------------------
    # Cramer's V
    # -------------------------------------------------

    def cramers_v(self, x, y):

        confusion = pd.crosstab(x, y)

        chi2 = chi2_contingency(
            confusion
        )[0]

        n = confusion.sum().sum()

        phi2 = chi2 / n

        r, k = confusion.shape

        return np.sqrt(
            phi2 / min(k - 1, r - 1)
        )

    # -------------------------------------------------
    # Categorical Correlation
    # -------------------------------------------------

    def plot_categorical_correlation(self):

        cat_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns

        matrix = pd.DataFrame(
            index=cat_cols,
            columns=cat_cols
        )

        for col1 in cat_cols:

            for col2 in cat_cols:

                matrix.loc[col1, col2] = (
                    self.cramers_v(
                        self.df[col1],
                        self.df[col2]
                    )
                )

        matrix = matrix.astype(float)

        fig = px.imshow(
            matrix,
            text_auto=True,
            title="Categorical Correlation"
        )

        fig.show()

    # -------------------------------------------------
    # Unified Heatmap
    # -------------------------------------------------

    def plot_all_associations_heatmap(
        self,
        max_columns=15
    ):

        final_df = self.create_normalized_data_df()

        corr = final_df.corr().abs()

        column_scores = corr.sum()

        selected_columns = (
            column_scores
            .sort_values(ascending=False)
            .head(max_columns)
            .index
        )
    
        corr = final_df[selected_columns].corr()
    
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Association Heatmap"
        )
    
        fig.update_layout(
            width=900,
            height=900
        )
    
        fig.show()
