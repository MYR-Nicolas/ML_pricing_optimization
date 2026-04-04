import streamlit as st

st.set_page_config(layout="wide", page_title="Model Training - Sales Analysis")

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

st.title("Model Training Notebook")

st.markdown("""
<div class="section-title">Notebook Structure</div>

<div class="text-block dashlist">
<ul>
  <li>Imports
    <ul>
      <li>Data Loading</li>
    </ul>
  </li>
  <li>Preprocessing
    <ul>
      <li>Dataset Splitting</li>
    </ul>
  </li>
  <li>Model Evaluation Tools
    <ul>
      <li>Metrics Analysis</li>
      <li>Resource Usage Analysis</li>
      <li>Learning Curves</li>
      <li>SHAP Interpretation</li>
    </ul>
  </li>
  <li>Training
    <ul>
      <li>Baseline</li>
      <li>Machine Learning
        <ul>
          <li>Dataset Benchmarking</li>
          <li>Best Model Selection</li>
          <li>Final Model Training</li>
          <li>Model Interpretation</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

with open("Notebooks/test_model.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)