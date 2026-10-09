import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Statistical Analyst", layout="wide")

# ---------- BACKGROUND ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(120deg,#1f4037,#99f2c8);
}

h1,h2,h3{
color:black;
text-align:center;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("🤖 AI Statistical Analysis Dashboard")

st.write("Upload dataset and choose statistical analysis.")

# ---------- FILE UPLOAD ----------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ---------- VARIABLE SELECTION ----------
    st.sidebar.header("Analysis Settings")

    response = st.sidebar.selectbox("Select Response Variable", numeric_cols)

    predictors = st.sidebar.multiselect(
        "Select Predictor Variables",
        numeric_cols,
        default=[col for col in numeric_cols if col != response]
    )

    group_var = st.sidebar.selectbox("Grouping Variable (for t-test/ANOVA)", ["None"] + cat_cols)

    # ---------- ANALYSIS OPTIONS ----------
    analysis = st.sidebar.multiselect(
        "Select Analysis",
        [
            "Descriptive Statistics",
            "Correlation",
            "Regression",
            "T-Test",
            "ANOVA"
        ]
    )

    graphs = st.sidebar.multiselect(
        "Select Graphs",
        [
            "Histogram",
            "Boxplot",
            "Scatterplot",
            "Pairplot",
            "Correlation Heatmap"
        ]
    )

    # ---------- DESCRIPTIVE ----------
    if "Descriptive Statistics" in analysis:

        st.header("Descriptive Statistics")

        st.write(df.describe())

    # ---------- CORRELATION ----------
    if "Correlation" in analysis:

        st.header("Correlation Matrix")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm")

        st.pyplot(fig)

    # ---------- REGRESSION ----------
    if "Regression" in analysis and predictors:

        st.header("Regression Analysis")

        X = df[predictors]
        y = df[response]

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)

        st.write("R² Score:", r2)

        coef = pd.DataFrame({
            "Variable": predictors,
            "Coefficient": model.coef_
        })

        st.dataframe(coef)

        X2 = sm.add_constant(X)
        results = sm.OLS(y, X2).fit()

        st.text(results.summary())

    # ---------- T TEST ----------
    if "T-Test" in analysis and group_var != "None":

        st.header("Independent T-Test")

        groups = df[group_var].unique()

        if len(groups) == 2:

            g1 = df[df[group_var] == groups[0]][response]
            g2 = df[df[group_var] == groups[1]][response]

            t, p = stats.ttest_ind(g1, g2)

            st.write("t statistic:", t)
            st.write("p-value:", p)

            if p < 0.05:
                st.success("Significant difference between groups")
            else:
                st.warning("No significant difference")

        else:
            st.warning("T-test requires exactly 2 groups.")

    # ---------- ANOVA ----------
    if "ANOVA" in analysis and group_var != "None":

        st.header("One Way ANOVA")

        formula = f"{response} ~ C({group_var})"

        model = ols(formula, data=df).fit()

        anova_table = sm.stats.anova_lm(model, typ=2)

        st.dataframe(anova_table)

    # ---------- GRAPHS ----------
    st.header("Graphs")

    if "Histogram" in graphs:

        for col in numeric_cols:

            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True)
            plt.title(f"Histogram of {col}")
            st.pyplot(fig)

    if "Boxplot" in graphs and group_var != "None":

        for col in numeric_cols:

            fig, ax = plt.subplots()
            sns.boxplot(x=df[group_var], y=df[col])
            plt.title(f"Boxplot of {col}")
            st.pyplot(fig)

    if "Scatterplot" in graphs and predictors:

        for col in predictors:

            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col], y=df[response])
            plt.title(f"{response} vs {col}")
            st.pyplot(fig)

    if "Pairplot" in graphs:

        st.write("Pairplot")

        fig = sns.pairplot(df[numeric_cols])

        st.pyplot(fig)

    if "Correlation Heatmap" in graphs:

        fig, ax = plt.subplots()

        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="viridis")

        st.pyplot(fig)

else:

    st.info("Upload a dataset to start analysis.")
