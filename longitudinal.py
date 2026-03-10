import streamlit as st
import pandas as pd
import plotly.express as px

# Dashboard Configuration
st.set_page_config(page_title="Zimbabwe Infant HIV Dashboard", page_icon="🇿🇼", layout="wide")

st.title("🇿🇼 Infant HIV Testing & Outcomes Dashboard")

@st.cache_data
def load_data():
    # Load files
    df1 = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
    df2 = pd.read_csv('uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv')
    
    # Standardize Date of Birth
    df1['dob_dt'] = pd.to_datetime(df1['infant_date_of_birth'], dayfirst=True, errors='coerce')
    df2['dob_dt'] = pd.to_datetime(df2['infant_date_of_birth'], dayfirst=True, errors='coerce')
    
    # Create proxy ID for Dataset 2 (No Trace)
    df2['proxy_id'] = (df2['infant_date_of_birth'].astype(str) + "_" + 
                      df2['infant_sex'].astype(str) + "_" + 
                      df2['facility_id'].astype(str))
    
    return df1, df2

df1_raw, df2_raw = load_data()

# --- Sidebar Date Filter ---
st.sidebar.header("Reporting Period")
start_date = pd.to_datetime(st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-10-01")))
end_date = pd.to_datetime(st.sidebar.date_input("End Date", value=pd.to_datetime("2023-09-30")))

# --- Data Processing (Filtering for Cohort) ---
c1 = df1_raw[(df1_raw['dob_dt'] >= start_date) & (df1_raw['dob_dt'] <= end_date)].copy()
c2 = df2_raw[(df2_raw['dob_dt'] >= start_date) & (df2_raw['dob_dt'] <= end_date)].copy()

# 1. Dataset 1 (Traceable): Group by unique child_person_id
c1_grouped = c1.groupby('child_person_id').agg({
    'infant_hiv_test_result': lambda x: list(x.dropna().astype(str).str.upper()),
    'facility': 'first'
}).reset_index()
c1_grouped['is_tested'] = c1_grouped['infant_hiv_test_result'].apply(lambda x: len(x) > 0)
c1_grouped['test_count'] = c1_grouped['infant_hiv_test_result'].apply(len)

# 2. Dataset 2 (No Traceable): Group by proxy_id
c2_grouped = c2.groupby('proxy_id').agg({
    'infant_hiv_test_result': lambda x: list(x.dropna().astype(str).str.upper()),
    'facility': 'first'
}).reset_index()
c2_grouped['is_tested'] = True  # Based on dataset definition
c2_grouped['test_count'] = c2_grouped['infant_hiv_test_result'].apply(len)

# --- Calculation of Final Unique Metrics ---
count_trace = len(c1_grouped)
count_no_trace = len(c2_grouped)
total_infants = count_trace + count_no_trace

tested_trace = c1_grouped['is_tested'].sum()
tested_no_trace = len(c2_grouped)
total_tested = tested_trace + tested_no_trace
total_not_tested = total_infants - total_tested

# Outcomes Logic
def get_final_outcome(res_list):
    if not res_list: return "Unknown/Pending"
    res_str = " ".join(res_list)
    if "POSITIVE" in res_str: return "HIV Positive"
    if "NEGATIVE" in res_str: return "HIV Negative"
    return "Unknown/Pending"

c1_grouped['outcome'] = c1_grouped['infant_hiv_test_result'].apply(get_final_outcome)
c2_grouped['outcome'] = c2_grouped['infant_hiv_test_result'].apply(get_final_outcome)

o_pos = (c1_grouped['outcome'] == "HIV Positive").sum() + (c2_grouped['outcome'] == "HIV Positive").sum()
o_neg = (c1_grouped['outcome'] == "HIV Negative").sum() + (c2_grouped['outcome'] == "HIV Negative").sum()
o_unk = total_infants - (o_pos + o_neg)

# Test Distribution
all_test_counts = list(c1_grouped[c1_grouped['is_tested']]['test_count']) + list(c2_grouped['test_count'])
t_counts = pd.Series(all_test_counts).value_counts()
t1, t2, t3plus = t_counts.get(1, 0), t_counts.get(2, 0), t_counts[t_counts.index >= 3].sum()

# --- Visuals ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cohort Overview: Trace Status")
    fig_cohort = px.pie(values=[count_trace, count_no_trace], 
                        names=['With Mother Trace', 'Without Mother Trace'],
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_cohort, use_container_width=True)

with col2:
    st.subheader("Distribution of Tests per Child")
    fig_dist = px.bar(x=['1 Test', '2 Tests', '3+ Tests'], y=[t1, t2, t3plus],
                      labels={'x': 'Number of Tests', 'y': 'Number of Children'},
                      color=['1 Test', '2 Tests', '3+ Tests'],
                      color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_dist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Final Outcomes at 2 Years")
    fig_out = px.pie(values=[o_neg, o_pos, o_unk], 
                     names=['HIV Negative', 'HIV Positive', 'Unknown/Pending'],
                     hole=0.4,
                     color=['HIV Negative', 'HIV Positive', 'Unknown/Pending'],
                     color_discrete_map={'HIV Positive': '#E41A1C', 'HIV Negative': '#377EB8', 'Unknown/Pending': '#999999'})
    st.plotly_chart(fig_out, use_container_width=True)

with col4:
    st.subheader("Top 10 Facilities (by Unique Children)")
    all_facs = pd.concat([c1_grouped['facility'], c2_grouped['facility']])
    top_facs = all_facs.value_counts().head(10)
    fig_fac = px.bar(top_facs, orientation='h', labels={'value': 'Children', 'index': 'Facility'})
    st.plotly_chart(fig_fac, use_container_width=True)

st.divider()

# --- Summary at the Bottom ---
st.subheader("📋 Summary")
sum_left, sum_right = st.columns(2)

with sum_left:
    st.markdown(f"""
    **Cohort Overview:**
    * **{total_infants}** HIV-exposed infants born between {start_date.strftime('%b %Y')} and {end_date.strftime('%b %Y')}
    * **{count_trace}** infants ({(count_trace/total_infants*100):.1f}%) with mother trace
    * **{count_no_trace}** infants ({(count_no_trace/total_infants*100):.1f}%) without mother trace
    * **{total_tested}** infants ({(total_tested/total_infants*100):.1f}%) were tested for HIV
    * **{total_not_tested}** infants ({(total_not_tested/total_infants*100):.1f}%) were not tested
    """)

with sum_right:
    st.markdown(f"""
    **Testing Coverage:**
    * **{t1}** children had 1 test
    * **{t2}** children had 2 tests
    * **{t3plus}** children had 3 or more tests
    
    **Final Outcomes (Unique Children):**
    * **{o_neg}** children ({(o_neg/total_infants*100):.1f}%) tested HIV NEGATIVE
    * **{o_pos}** children ({(o_pos/total_infants*100):.1f}%) tested HIV POSITIVE
    * **{o_unk}** children ({(o_unk/total_infants*100):.1f}%) have unknown/pending results
    
    **Facilities:**
    * Services provided across **{all_facs.nunique()}** facilities
    """)