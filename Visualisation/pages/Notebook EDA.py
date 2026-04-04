import streamlit as st

# Page configuration
st.set_page_config(layout="wide", page_title="EDA - Transaction Analysis")

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
}
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 34px;
    margin-bottom: 10px;
}
.text-block {
    font-size: 16px;
    line-height: 1.6;
}
.dashlist ul{
    list-style: none;
    margin: 0;
    padding-left: 0;
}
.dashlist ul ul{
    margin-left: 26px;
    margin-top: 6px;
}
.dashlist li{
    margin: 6px 0;
}
.dashlist li::before{
    content: "- ";
}
</style>
""", unsafe_allow_html=True)

st.title("Exploratory Data Analysis Notebook")

st.markdown("""
<div class="section-title">Notebook Structure</div>
<div class="text-block dashlist">
<ul>
  <li>Imports</li>
  <li>Load and Overview of Datasets
    <ul>
      <li>Memory Usage Analysis</li>
    </ul>
  </li>
  <li>Data Cleaning
    <ul>
      <li>Missing Values</li>
      <li>Duplicate Values</li>
      <li>Outliers, Inconsistent, and Invalid Data</li>
      <li>Column Type Conversion</li>
      <li>Outlier Visualization and IQR Analysis</li>
      <li>Cleaned and Filtered Dataset</li>
    </ul>
  </li>
  <li>Analysis
    <ul>
      <li>Univariate Analysis
        <ul>
          <li>Analysis of Numerical Variables</li>
          <li>Analysis of Categorical Variables</li>
        </ul>
      </li>
      <li>Bivariate Analysis
        <ul>
          <li>Correlation Analysis</li>
          <li>Trend Analysis</li>
        </ul>
      </li>
      <li>Statistical Tests</li>
      <li>Price Elasticity</li>
      <li>Basket Intensity Proxy</li>
    </ul>
  </li>
  <li>Feature Engineering
    <ul>
      <li>ML Datasets</li>
    </ul>
  </li>
  <li>Final Verification
    <ul>
      <li>Filter to Remove Non-Relevant Columns</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

with open("Notebooks/exploratory_data_analysis.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)