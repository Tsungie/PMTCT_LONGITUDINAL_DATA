import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Zimbabwe PMTCT Portal", layout="wide")
st.title("🇿🇼 PMTCT Longitudinal Monitoring Tool")
st.markdown("---")

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
    
    # 1. Clean Dates
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # 2. UNIFORM WORDS: Fix Capitalization & Whitespace for ALL columns
    # This turns "Positive ", "positive", and "POSITIVE" all into "POSITIVE"
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.upper().str.strip()
    
    # Handle the 'nan' strings created by the upper() function on empty cells
    df = df.replace('NAN', pd.NA)
    
    return df

df = load_and_clean_data()

# --- SIDEBAR: FACILITY MANAGEMENT ---
st.sidebar.header("Navigation & Filters")
all_sites = sorted(df['facility'].dropna().unique().tolist())

# Select All / Clear All Logic
if "sites" not in st.session_state:
    st.session_state.sites = all_sites

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Select All"):
    st.session_state.sites = all_sites
if col_b.button("Clear All"):
    st.session_state.sites = []

selected_sites = st.sidebar.multiselect(
    "Search Facilities:", 
    options=all_sites, 
    default=st.session_state.sites
)

filtered_df = df[df['facility'].isin(selected_sites)]

# --- MAIN DASHBOARD ---

if filtered_df.empty:
    st.warning("Please select at least one facility from the sidebar.")
else:
    # 1. KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mothers Enrolled", len(filtered_df))
    c2.metric("On ART", filtered_df['mother_art_number'].notna().sum())
    c3.metric("Infants Tested", filtered_df['infant_hiv_test_result'].notna().sum())
    
    # Calculate Seroconversion count (simplified logic)
    seroconverts = len(filtered_df[
        (filtered_df['mother_hiv_status_at_booking'] == 'NEGATIVE') & 
        (filtered_df['mother_hiv_test_result'] == 'POSITIVE')
    ])
    c4.metric("Seroconversions", seroconverts, delta_color="inverse")

    st.markdown("### Clinical Journey Analysis")
    tab1, tab2, tab3 = st.tabs(["The Cascade Funnel", "Risk Comparison", "Geospatial View"])

    with tab1:
        # Funnel Logic
        steps = ["Total ANC", "Tested", "On ART", "Infant Tested", "Final Negative"]
        counts = [
            filtered_df['anc_id'].nunique(),
            filtered_df[filtered_df['mother_hiv_test_result'].notna()]['anc_id'].nunique(),
            filtered_df[filtered_df['mother_art_id'].notna()]['anc_id'].nunique(),
            filtered_df[filtered_df['infant_hiv_test_result'].notna()]['anc_id'].nunique(),
            filtered_df[filtered_df['child_hiv_status_at_6_months'] == 'NEGATIVE']['anc_id'].nunique()
        ]
        fig_funnel = px.funnel(x=counts, y=steps, color=steps, title="PMTCT Attrition Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)

    with tab2:
        # Risk Story
        fig_risk = px.histogram(
            filtered_df[filtered_df['infant_hiv_test_result'].notna()],
            x="mother_started_art_before_current_pregnancy",
            color="infant_hiv_test_result",
            barmode="group",
            title="Infant Outcome by Maternal ART History"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with tab3:
        st.info("Map view requires GPS coordinates (Lat/Lon) for each facility.")
        # Simplified Map (Placeholder)
        # In a real app, you'd merge with a CSV containing facility coordinates
        st.map(pd.DataFrame({'lat': [-19.01], 'lon': [29.15]})) # Center of Zimbabwe

    # 4. DATA QUALITY ALERT: The "Pending Results"
    st.markdown("---")
    st.subheader("⚠️ Data Entry Alerts: Pending Infant Results")
    pending = filtered_df[filtered_df['infant_hiv_test_date'].notna() & filtered_df['infant_hiv_test_result'].isna()]
    if not pending.empty:
        st.error(f"There are {len(pending)} infants with a test date but NO result recorded. These require urgent follow-up.")
        st.dataframe(pending[['facility', 'anc_id', 'infant_hiv_test_date']].head(10))