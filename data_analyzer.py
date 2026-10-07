import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Advanced AI Stat Analyst", page_icon="📊", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px 5px 0 0; padding: 10px 20px; box-shadow: 0px 2px 5px rgba(0,0,0,0.05);}
    .stTabs [aria-selected="true"] {border-bottom: 3px solid #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Statistical Analysis & ML Dashboard")
st.markdown("Upload your dataset to explore both **Quantitative** and **Qualitative** data with interactive visuals and advanced machine learning models.")

# ---------- SIDEBAR & FILE UPLOAD ----------
with st.sidebar:
    st.header("1. Upload Data")
    file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if file:
        st.success("File uploaded successfully!")
        
if file:
    # Read data
    df = pd.read_csv(file)
    
    # Preprocessing: Missing Values
    with st.sidebar.expander("🛠️ Data Preprocessing", expanded=False):
        handle_na = st.radio("Handle Missing Values:", ["Keep", "Drop Rows with NA", "Fill with Mean/Mode"])
        if handle_na == "Drop Rows with NA":
            df = df.dropna()
        elif handle_na == "Fill with Mean/Mode":
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    # Separate columns by type
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗂️ Data Overview", 
        "📈 Interactive EDA", 
        "🧪 Statistical Tests", 
        "🤖 Predictive Modeling"
    ])

    # ================= TAB 1: DATA OVERVIEW =================
    with tab1:
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())
        
        st.dataframe(df, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Quantitative (Numerical) Summary")
            st.dataframe(df.describe())
        with col_b:
            st.subheader("Qualitative (Categorical) Summary")
            if cat_cols:
                st.dataframe(df[cat_cols].describe())
            else:
                st.info("No categorical columns found.")

    # ================= TAB 2: INTERACTIVE EDA =================
    with tab2:
        st.header("Exploratory Data Analysis")
        
        eda_type = st.radio("Select Analysis Type:", ["Quantitative (Numerical)", "Qualitative (Categorical)", "Correlation Matrix"], horizontal=True)
        
        if eda_type == "Quantitative (Numerical)" and num_cols:
            sel_num = st.selectbox("Select Variable to visualize:", num_cols)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(df, x=sel_num, marginal="box", title=f"Distribution of {sel_num}", color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                if len(num_cols) > 1:
                    sel_num2 = st.selectbox("Select Y-axis Variable for Scatter:", [c for c in num_cols if c != sel_num])
                    fig2 = px.scatter(df, x=sel_num, y=sel_num2, title=f"{sel_num} vs {sel_num2}", trendline="ols", color_discrete_sequence=['#e74c3c'])
                    st.plotly_chart(fig2, use_container_width=True)

        elif eda_type == "Qualitative (Categorical)" and cat_cols:
            sel_cat = st.selectbox("Select Categorical Variable:", cat_cols)
            fig = px.bar(df[sel_cat].value_counts().reset_index(), x='index', y=sel_cat, labels={'index': sel_cat, sel_cat: 'Count'}, title=f"Frequency of {sel_cat}", color='index')
            st.plotly_chart(fig, use_container_width=True)
            
        elif eda_type == "Correlation Matrix" and len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Numeric Variables Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 3: STATISTICAL TESTS =================
    with tab3:
        st.header("Hypothesis Testing")
        test_type = st.selectbox("Select Test:", ["T-Test (Num vs Cat)", "ANOVA (Num vs Cat)", "Chi-Square (Cat vs Cat)"])
        
        if test_type == "T-Test (Num vs Cat)" and num_cols and cat_cols:
            st.markdown("**Independent T-Test**: Compares the means of a numerical variable across exactly 2 categories.")
            t_num = st.selectbox("Select Numerical Variable (Y):", num_cols, key='t_num')
            t_cat = st.selectbox("Select Categorical Variable (Groups):", cat_cols, key='t_cat')
            
            groups = df[t_cat].dropna().unique()
            if len(groups) == 2:
                g1 = df[df[t_cat] == groups[0]][t_num].dropna()
                g2 = df[df[t_cat] == groups[1]][t_num].dropna()
                t_stat, p_val = stats.ttest_ind(g1, g2)
                st.write(f"**Group 1 ({groups[0]}):** Mean = {g1.mean():.4f}")
                st.write(f"**Group 2 ({groups[1]}):** Mean = {g2.mean():.4f}")
                st.metric("P-Value", f"{p_val:.5f}")
                if p_val < 0.05:
                    st.success("Result: Statistically Significant Difference (Reject Null Hypothesis)")
                else:
                    st.warning("Result: No Significant Difference (Fail to Reject Null Hypothesis)")
            else:
                st.error(f"T-Test requires exactly 2 groups. '{t_cat}' has {len(groups)} groups. Use ANOVA instead.")

        elif test_type == "ANOVA (Num vs Cat)" and num_cols and cat_cols:
            st.markdown("**One-Way ANOVA**: Compares the means of a numerical variable across 3 or more categories.")
            a_num = st.selectbox("Select Numerical Variable (Y):", num_cols, key='a_num')
            a_cat = st.selectbox("Select Categorical Variable (Groups):", cat_cols, key='a_cat')
            
            formula = f"{a_num} ~ C({a_cat})"
            try:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(anova_table)
                p_val_anova = anova_table['PR(>F)'][0]
                if p_val_anova < 0.05:
                    st.success("Result: Statistically Significant Difference between groups.")
                else:
                    st.warning("Result: No Significant Difference between groups.")
            except Exception as e:
                st.error("Error computing ANOVA. Ensure column names have no spaces/special characters.")

        elif test_type == "Chi-Square (Cat vs Cat)" and len(cat_cols) >= 2:
            st.markdown("**Chi-Square Test of Independence**: Tests if two categorical variables are related.")
            c1 = st.selectbox("Select Categorical Variable 1:", cat_cols, key='c1')
            c2 = st.selectbox("Select Categorical Variable 2:", [c for c in cat_cols if c != c1], key='c2')
            
            contingency = pd.crosstab(df[c1], df[c2])
            st.write("Contingency Table:")
            st.dataframe(contingency)
            
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            st.metric("P-Value", f"{p:.5f}")
            if p < 0.05:
                st.success("Result: Variables are significantly associated (Dependent).")
            else:
                st.warning("Result: Variables are not significantly associated (Independent).")

    # ================= TAB 4: PREDICTIVE MODELING =================
    with tab4:
        st.header("Advanced Predictive Modeling")
        st.markdown("Automatically encodes categorical variables (One-Hot Encoding) and builds models.")
        
        target = st.selectbox("Select Target Variable (Y):", df.columns)
        predictors = st.multiselect("Select Predictor Variables (X):", [c for c in df.columns if c != target])
        
        if target and predictors:
            if st.button("Train Model 🚀"):
                # Data Prep
                X = df[predictors].copy()
                y = df[target].copy()
                
                # Drop NAs for modeling
                valid_idx = X.dropna().index.intersection(y.dropna().index)
                X = X.loc[valid_idx]
                y = y.loc[valid_idx]
                
                # One-hot encode categorical predictors
                X = pd.get_dummies(X, drop_first=True)
                
                # Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Determine task type based on Target variable type
                if df[target].dtype in ['float64', 'int64'] and df[target].nunique() > 10:
                    # REGRESSION TASK
                    st.subheader("Linear Regression Model")
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    r2 = r2_score(y_test, preds)
                    st.metric("R² Score (Test Data)", f"{r2:.4f}")
                    
                    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}).sort_values(by="Coefficient", ascending=False)
                    st.write("**Feature Importance (Coefficients):**")
                    fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation='h', title="Feature Impact on Target")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # CLASSIFICATION TASK
                    st.subheader("Logistic Regression (Classification Model)")
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, preds)
                    st.metric("Accuracy (Test Data)", f"{acc:.2%}")
                    
                    st.write("**Confusion Matrix:**")
                    cm = confusion_matrix(y_test, preds)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", x=model.classes_, y=model.classes_, labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Please upload a CSV dataset in the sidebar to begin.")
