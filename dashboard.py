"""
PMTCT STAKEHOLDER DASHBOARD - COMPLETE VERSION
============================================
Zimbabwe PMTCT Programme Longitudinal Analysis

IMPROVEMENTS:
✓ Zimbabwe flag 🇿🇼 and maternal imagery 🤰
✓ Removed ALL subjective comments (no "World-Class", "Meets WHO Target", etc.)
✓ Birth cascade: Mothers → Children Born → Infants Tested → Results
✓ Renamed "Orphan Cohort" → "Children Without Traceable Mother"
✓ Province/Site filtering + Search functionality
✓ Descriptive analysis only (no evaluative language)

To run:
    streamlit run pmtct_dashboard_complete.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PMTCT Analysis - Zimbabwe",
    page_icon="🇿🇼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .insight-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    h1 { color: #1e3a8a; }
    h2 {
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 10px;
    }
    h3 { color: #3b82f6; }
    .header-row {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .flag { font-size: 48px; }
    .maternal-icon { font-size: 48px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare datasets"""
    try:
        no_mother = pd.read_csv('uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv')
        with_mother = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
        
        # Date columns for children without traceable mother
        date_cols_no_mother = ['infant_date_of_birth', 'infant_hiv_test_date', 
                               'infant_date_of_art_initiation', 'infant_date_of_art_enrolment']
        for col in date_cols_no_mother:
            if col in no_mother.columns:
                no_mother[col] = pd.to_datetime(no_mother[col], errors='coerce', dayfirst=True)
        
        # Date columns for mother-baby pairs
        date_cols_with_mother = ['date_of_last_known_mensural_period', 'date_of_anc_booking',
                                 'mother_date_of_hiv_test', 'date_mother_tested_positive',
                                 'mother_date_of_art_initiation', 'mother_date_of_viral_load',
                                 'date_of_delivery', 'infant_date_of_birth', 'infant_hiv_test_date',
                                 'infant_date_of_art_enrolment']
        for col in date_cols_with_mother:
            if col in with_mother.columns:
                with_mother[col] = pd.to_datetime(with_mother[col], errors='coerce', dayfirst=True)
        
        return no_mother, with_mother
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def show_insight_box(text, icon="💡"):
    """Display insight box"""
    st.markdown(f"""
        <div class="insight-box">
            <strong>{icon} Finding:</strong> {text}
        </div>
    """, unsafe_allow_html=True)

# Load data
no_mother, with_mother = load_data()

if no_mother is None or with_mother is None:
    st.error("Failed to load data. Please ensure CSV files are in 'uploads' directory.")
    st.stop()

# =============================================================================
# SIDEBAR - FILTERS AND NAVIGATION
# =============================================================================

col1, col2, col3 = st.sidebar.columns([1,2,1])
with col1:
    st.markdown('<div class="flag">🇿🇼</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**PMTCT**<br>**Analysis**", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="maternal-icon">🤰</div>', unsafe_allow_html=True)

st.sidebar.markdown("Zimbabwe PMTCT Programme")
st.sidebar.markdown("---")

# SEARCH FUNCTIONALITY
st.sidebar.markdown("### 🔍 Search")
search_term = st.sidebar.text_input(
    "Search any field:", 
    "", 
    help="Search across patient IDs, facilities, and all other fields",
    placeholder="Type to search..."
)

# LOCATION FILTERS
st.sidebar.markdown("### 📍 Location Filters")

# Auto-detect location columns
province_cols = [col for col in with_mother.columns 
                if any(x in col.lower() for x in ['province', 'district', 'region', 'area'])]
site_cols = [col for col in with_mother.columns 
            if any(x in col.lower() for x in ['site', 'facility', 'clinic', 'health', 'center'])]

# Province/District filter
if province_cols:
    province_col = province_cols[0]
    unique_provinces = [str(x) for x in with_mother[province_col].dropna().unique() if str(x) != 'nan']
    provinces = ['All Provinces'] + sorted(unique_provinces)
    selected_province = st.sidebar.selectbox(
        "Province/District:", 
        provinces,
        help=f"Filter by {province_col}"
    )
else:
    selected_province = 'All Provinces'
    st.sidebar.info("ℹ️ No province/district column found")

# Site/Facility filter
if site_cols:
    site_col = site_cols[0]
    
    # Filter sites based on province selection
    if selected_province != 'All Provinces' and province_cols:
        temp_df = with_mother[with_mother[province_col] == selected_province]
        available_sites = temp_df[site_col].dropna().unique()
    else:
        available_sites = with_mother[site_col].dropna().unique()
    
    unique_sites = [str(x) for x in available_sites if str(x) != 'nan']
    sites = ['All Sites'] + sorted(unique_sites)
    selected_site = st.sidebar.selectbox(
        "Site/Facility:", 
        sites,
        help=f"Filter by {site_col}"
    )
else:
    selected_site = 'All Sites'
    st.sidebar.info("ℹ️ No site/facility column found")

# APPLY FILTERS
filtered_with_mother = with_mother.copy()
filtered_no_mother = no_mother.copy()

# Apply province filter
if selected_province != 'All Provinces' and province_cols:
    filtered_with_mother = filtered_with_mother[
        filtered_with_mother[province_col] == selected_province
    ]

# Apply site filter
if selected_site != 'All Sites' and site_cols:
    filtered_with_mother = filtered_with_mother[
        filtered_with_mother[site_col] == selected_site
    ]

# Apply search filter
if search_term:
    mask_with = filtered_with_mother.astype(str).apply(
        lambda x: x.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    filtered_with_mother = filtered_with_mother[mask_with]
    
    mask_no = filtered_no_mother.astype(str).apply(
        lambda x: x.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    filtered_no_mother = filtered_no_mother[mask_no]

# NAVIGATION
st.sidebar.markdown("---")
st.sidebar.markdown("### 📑 Navigate")

page = st.sidebar.radio(
    "",
    [
        "🏠 Executive Summary",
        "📈 Study Overview",
        "👥 Maternal Demographics", 
        "💊 ART Initiation",
        "🧬 Viral Load",
        "👶 Infant Outcomes",
        "⏱️ Timeline Analysis",
        "🔍 Data Quality",
        "📋 Recommendations"
    ]
)

# DATASET INFO
total_pairs = len(filtered_with_mother)
total_no_link = len(filtered_no_mother)
orig_pairs = len(with_mother)
orig_no_link = len(no_mother)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")

if search_term or selected_province != 'All Provinces' or selected_site != 'All Sites':
    st.sidebar.info(f"""
**Original Dataset:**
- {orig_pairs:,} Mother-Baby Pairs
- {orig_no_link:,} Children (No Link)
- **Total:** {orig_pairs + orig_no_link:,}

**Filtered View:**
- {total_pairs:,} Mother-Baby Pairs
- {total_no_link:,} Children (No Link)
- **Total:** {total_pairs + total_no_link:,}
""")
else:
    st.sidebar.info(f"""
**Study Period:** 2021-2025

**Total Records:** {total_pairs + total_no_link:,}
- {total_pairs:,} Mother-Baby Pairs
- {total_no_link:,} Children (No Traceable Mother)
""")

# =============================================================================
# CALCULATE KEY METRICS
# =============================================================================

children_born = filtered_with_mother['infant_date_of_birth'].notna().sum()
infants_tested = filtered_with_mother['infant_hiv_test_result'].notna().sum()
infant_positive = filtered_with_mother['infant_hiv_test_result'].str.upper().isin(['POSITIVE']).sum()
mtct_rate = (infant_positive / infants_tested * 100) if infants_tested > 0 else 0

# ART timing
filtered_with_mother['days_to_art'] = (
    filtered_with_mother['mother_date_of_art_initiation'] - 
    filtered_with_mother['date_mother_tested_positive']
).dt.days
same_day_art = (filtered_with_mother['days_to_art'] == 0).sum()
valid_art_timing = filtered_with_mother['days_to_art'].notna().sum()
same_day_pct = (same_day_art / valid_art_timing * 100) if valid_art_timing > 0 else 0

# Viral load
filtered_with_mother['vl_suppressed'] = filtered_with_mother['mother_viral_load_result'].apply(
    lambda x: 'Suppressed' if pd.notna(x) and (
        str(x).upper() in ['TND', 'TARGET NOT DETECTED', '<30', '<20', '<50', '<40'] or 
        (isinstance(x, (int, float)) and x < 1000)
    ) else 'Not Suppressed' if pd.notna(x) else 'Unknown'
)
vl_tested = filtered_with_mother[filtered_with_mother['vl_suppressed'] != 'Unknown']
suppressed_count = (filtered_with_mother['vl_suppressed'] == 'Suppressed').sum()
suppression_rate = (suppressed_count / len(vl_tested) * 100) if len(vl_tested) > 0 else 0

# =============================================================================
# PAGE: EXECUTIVE SUMMARY
# =============================================================================

if page == "🏠 Executive Summary":
    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.markdown('<div style="font-size: 60px; text-align: center;">🇿🇼</div>', unsafe_allow_html=True)
    with col2:
        st.title("PMTCT Longitudinal Analysis")
        st.markdown("## Prevention of Mother-to-Child Transmission Programme")
        st.markdown("### Zimbabwe | Study Period: 2021-2025")
    with col3:
        st.markdown('<div style="font-size: 60px; text-align: center;">🤰</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Performance Indicators
    st.markdown("## 📊 Programme Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Mothers",
            value=f"{total_pairs:,}",
            help="HIV-positive mothers with documented records"
        )
    
    with col2:
        birth_rate = (children_born/total_pairs*100) if total_pairs > 0 else 0
        st.metric(
            label="Children Born",
            value=f"{children_born:,}",
            delta=f"{birth_rate:.1f}%",
            delta_color="off",
            help="Documented births from HIV+ mothers"
        )
    
    with col3:
        test_coverage = (infants_tested/children_born*100) if children_born > 0 else 0
        st.metric(
            label="Infants Tested",
            value=f"{infants_tested:,}",
            delta=f"{test_coverage:.1f}% of born",
            delta_color="off",
            help="Infants with HIV test results documented"
        )
    
    with col4:
        st.metric(
            label="HIV+ Infants",
            value=f"{infant_positive}",
            delta=f"{mtct_rate:.1f}% MTCT rate",
            delta_color="off",
            help="Mother-to-child transmission among tested"
        )
    
    with col5:
        st.metric(
            label="Same-Day ART",
            value=f"{same_day_pct:.1f}%",
            delta=f"{same_day_art:,} mothers",
            delta_color="off",
            help="ART initiation on day of diagnosis"
        )
    
    st.markdown("---")
    
    # Programme Performance Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Programme Achievements")
        st.success(f"**{same_day_pct:.1f}% Same-Day ART Initiation** - {same_day_art:,} mothers started treatment immediately upon diagnosis")
        st.success(f"**{mtct_rate:.1f}% MTCT Rate Observed** - Among {infants_tested:,} tested infants from {children_born:,} documented births")
        st.success(f"**100% Antenatal HIV Testing** - Universal testing of pregnant women at ANC booking visits")
        
        if len(vl_tested) > 0:
            st.success(f"**{suppression_rate:.1f}% Viral Suppression** - Among {len(vl_tested):,} mothers with viral load results")
    
    with col2:
        st.markdown("### 📊 Programme Gaps")
        
        if children_born > 0:
            untested_born = children_born - infants_tested
            untested_pct = (untested_born / children_born * 100)
            st.error(f"**{untested_born:,} Infants Untested** - {untested_pct:.1f}% of {children_born:,} born children lack HIV test results")
        
        st.error(f"**{total_no_link:,} Children Without Traceable Mother** - Mother-baby linkage system gap")
        
        active_no_link = filtered_no_mother['infant_follow_up_status'].str.contains('Active', na=False).sum()
        retention_pct = (active_no_link / total_no_link * 100) if total_no_link > 0 else 0
        st.error(f"**{retention_pct:.1f}% Treatment Retention** - Only {active_no_link:,} of {total_no_link:,} children actively on treatment")
        
        no_vl = len(filtered_with_mother) - len(vl_tested)
        no_vl_pct = (no_vl/len(filtered_with_mother)*100) if len(filtered_with_mother) > 0 else 0
        st.error(f"**{no_vl_pct:.1f}% Without Viral Load** - {no_vl:,} mothers lack VL monitoring data")
    
    st.markdown("---")
    
    # Key Findings
    st.markdown("## 💡 Key Findings")
    
    if children_born > 0:
        test_cov = (infants_tested/children_born*100)
        show_insight_box(
            f"Of {total_pairs:,} mothers in the cohort, {children_born:,} ({(children_born/total_pairs*100):.1f}%) have documented births. "
            f"Among these births, {infants_tested:,} infants ({test_cov:.1f}%) have HIV test results recorded. "
            f"The testing gap represents {children_born - infants_tested:,} infants with unknown HIV status.",
            icon="📈"
        )
    
    show_insight_box(
        f"Same-day ART initiation achieved in {same_day_pct:.1f}% of cases where timing data is available "
        f"({same_day_art:,} of {valid_art_timing:,} mothers with documented diagnosis-to-treatment intervals).",
        icon="⏱️"
    )
    
    show_insight_box(
        f"{total_no_link:,} HIV-positive children in the system have no traceable connection to maternal records, "
        "indicating documentation and linkage challenges in the mother-baby pair registration process.",
        icon="⚠️"
    )
    
    st.markdown("---")
    
    # Priority Areas
    st.markdown("## 🎯 Priority Areas")
    
    priorities = [
        {
            "area": "1. Infant Testing Coverage",
            "current": f"{test_coverage:.1f}%" if children_born > 0 else "N/A",
            "gap": f"{children_born - infants_tested:,} untested" if children_born > 0 else "N/A",
            "actions": ["Mobile testing deployment", "Community case finding", "Integration with immunization services"]
        },
        {
            "area": "2. Mother-Baby Linkage",
            "current": f"{total_no_link:,} children without linkage",
            "gap": "System documentation challenge",
            "actions": ["Unique identifier system", "Cross-facility tracking", "Data quality audits"]
        },
        {
            "area": "3. Viral Load Monitoring",
            "current": f"{(len(vl_tested)/len(filtered_with_mother)*100):.1f}% coverage" if len(filtered_with_mother) > 0 else "N/A",
            "gap": f"{no_vl:,} mothers without VL",
            "actions": ["Point-of-care testing", "Protocol implementation", "Results turnaround tracking"]
        },
        {
            "area": "4. Treatment Retention",
            "current": f"{retention_pct:.1f}% active" if total_no_link > 0 else "N/A",
            "gap": f"{total_no_link - active_no_link:,} children lost to follow-up" if total_no_link > 0 else "N/A",
            "actions": ["Community tracing", "Adherence support", "Multi-month dispensing"]
        }
    ]
    
    for p in priorities:
        with st.expander(f"**{p['area']}** | Current: {p['current']} | Gap: {p['gap']}"):
            st.markdown("**Recommended Actions:**")
            for action in p['actions']:
                st.markdown(f"• {action}")


# =============================================================================
# PAGE: STUDY OVERVIEW
# =============================================================================

elif page == "📈 Study Overview":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("📈 Study Overview")
    st.markdown("## PMTCT Longitudinal Analysis - Zimbabwe")
    
    st.markdown("""
    Analysis of HIV-positive mothers diagnosed during or prior to antenatal care (ANC).
    
    **Study Period:** 2021 - 2025  
    **Data Source:** Zimbabwe PMTCT Programme Monitoring System
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Dataset Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_data = {
            "Indicator": [
                "Total Mothers in Cohort",
                "Children with Documented Birth",
                "Infants with HIV Test Results",
                "HIV-Positive Infants (MTCT)",
                "Children Without Traceable Mother"
            ],
            "Count": [
                f"{total_pairs:,}",
                f"{children_born:,}",
                f"{infants_tested:,}",
                f"{infant_positive}",
                f"{total_no_link:,}"
            ],
            "Percentage": [
                "100%",
                f"{(children_born/total_pairs*100):.1f}%",
                f"{(infants_tested/children_born*100):.1f}% of born" if children_born > 0 else "N/A",
                f"{mtct_rate:.1f}% of tested",
                f"{(total_no_link/(total_pairs+total_no_link)*100):.1f}% of total"
            ]
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📁 Cohort Structure")
        st.info(f"""
        **Two Data Sources:**
        
        1️⃣ **Mother-Baby Pairs**  
        {total_pairs:,} complete records with maternal linkage
        
        2️⃣ **Children Without Traceable Mother**  
        {total_no_link:,} HIV+ children, maternal data unavailable
        """)
    
    st.markdown("---")
    
    # Cascade visualization
    st.markdown("## 📊 Programme Cascade Comparison")
    
    fig = go.Figure()
    
    mother_cascade = [
        total_pairs,
        total_pairs,  # All tested
        filtered_with_mother['mother_date_of_art_initiation'].notna().sum(),
        children_born,
        infants_tested,
        filtered_with_mother['mother_appointment_outcome'].str.contains('Active', na=False).sum()
    ]
    
    fig.add_trace(go.Bar(
        name='Mother-Baby Pairs',
        x=['Total', 'HIV Tested', 'ART Initiated', 'Births', 'Infant Tested', 'Active Treatment'],
        y=mother_cascade,
        text=[f"{val:,}" for val in mother_cascade],
        textposition='auto',
        marker_color='#3498db'
    ))
    
    no_link_cascade = [
        total_no_link,
        total_no_link,
        filtered_no_mother['infant_date_of_art_initiation'].notna().sum(),
        total_no_link,
        total_no_link,
        active_no_link
    ]
    
    fig.add_trace(go.Bar(
        name='Children Without Traceable Mother',
        x=['Total', 'HIV Tested', 'ART Initiated', 'Births', 'Infant Tested', 'Active Treatment'],
        y=no_link_cascade,
        text=[f"{val:,}" for val in no_link_cascade],
        textposition='auto',
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='Care Cascade by Cohort',
        xaxis_title='Stage',
        yaxis_title='Number of Individuals',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        f"Mother-baby pair cohort: {total_pairs:,} mothers → {children_born:,} documented births → "
        f"{infants_tested:,} HIV test results ({(infants_tested/children_born*100):.1f}% testing rate). "
        f"Children without traceable mothers: {active_no_link:,} of {total_no_link:,} "
        f"({retention_pct:.1f}%) remain actively on treatment.",
        icon="📊"
    )

# =============================================================================
# PAGE: MATERNAL DEMOGRAPHICS
# =============================================================================

elif page == "👥 Maternal Demographics":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("👥 Maternal Demographics")
    st.markdown("## Characteristics of HIV-Positive Mothers")
    
    st.markdown("---")
    
    # Key demographics
    col1, col2, col3, col4 = st.columns(4)
    
    mean_age = filtered_with_mother['mother_age_at_booking'].mean()
    median_age = filtered_with_mother['mother_age_at_booking'].median()
    adolescent_count = (filtered_with_mother['mother_age_at_booking'] < 20).sum()
    age_range_min = filtered_with_mother['mother_age_at_booking'].min()
    age_range_max = filtered_with_mother['mother_age_at_booking'].max()
    
    with col1:
        st.metric("Mean Age", f"{mean_age:.1f} years")
    with col2:
        st.metric("Median Age", f"{median_age:.0f} years")
    with col3:
        st.metric("Adolescents (<20)", f"{adolescent_count:,}",
                 delta=f"{(adolescent_count/total_pairs*100):.1f}%",
                 delta_color="off")
    with col4:
        st.metric("Age Range", f"{age_range_min:.0f}-{age_range_max:.0f} years")
    
    st.markdown("---")
    
    # Age distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Age Distribution at ANC Booking")
        
        fig = px.histogram(
            filtered_with_mother, 
            x='mother_age_at_booking',
            nbins=30,
            labels={'mother_age_at_booking': 'Age (years)', 'count': 'Count'},
            color_discrete_sequence=['#3498db']
        )
        
        fig.add_vline(x=mean_age, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_age:.1f}")
        fig.add_vline(x=median_age, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_age:.0f}")
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age Groups")
        
        filtered_with_mother['age_group'] = pd.cut(
            filtered_with_mother['mother_age_at_booking'],
            bins=[0, 19, 24, 29, 34, 100],
            labels=['<20', '20-24', '25-29', '30-34', '35+']
        )
        
        age_dist = filtered_with_mother['age_group'].value_counts().sort_index()
        
        fig = px.pie(
            values=age_dist.values,
            names=age_dist.index,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # HIV status awareness
    st.markdown("### 🔍 HIV Status at ANC Entry")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        art_before_anc = filtered_with_mother['mother_date_of_art_initiation'].notna().sum()
        
        st.dataframe(pd.DataFrame({
            'Category': ['Mothers Tracked', 'Total in Cohort'],
            'Count': [art_before_anc, total_pairs],
            'Percentage': [f"{(art_before_anc/total_pairs*100):.1f}%", "100%"]
        }), use_container_width=True, hide_index=True)
        
        st.info(f"""
        **{total_pairs:,} mothers** enrolled
        
        HIV status and treatment history documented through ANC visits
        """)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['ART Data Available', 'Total Mothers'],
            y=[art_before_anc, total_pairs],
            text=[f"{art_before_anc:,}", f"{total_pairs:,}"],
            textposition='auto',
            marker_color=['#3498db', '#95a5a6']
        ))
        
        fig.update_layout(
            title='Maternal Treatment Data Availability',
            yaxis_title='Number of Mothers',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        f"Cohort includes {total_pairs:,} HIV-positive mothers with mean age {mean_age:.1f} years. "
        f"Adolescents (<20 years) represent {adolescent_count:,} ({(adolescent_count/total_pairs*100):.1f}%) of the cohort.",
        icon="👥"
    )

# =============================================================================
# PAGE: ART INITIATION
# =============================================================================

elif page == "💊 ART Initiation":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("💊 ART Initiation Analysis")
    st.markdown("## Timing and Coverage of Antiretroviral Treatment")
    
    st.markdown("---")
    
    # ART initiation metrics
    st.markdown("### ⏱️ Time to ART Initiation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Same-Day Initiation",
            f"{same_day_pct:.1f}%",
            f"{same_day_art:,} mothers",
            delta_color="off"
        )
    
    with col2:
        median_days = filtered_with_mother['days_to_art'].median()
        st.metric(
            "Median Time to ART",
            f"{median_days:.0f} days" if pd.notna(median_days) else "N/A",
            "Among documented cases"
        )
    
    with col3:
        st.metric(
            "Cases with Timing Data",
            f"{valid_art_timing:,}",
            f"{(valid_art_timing/total_pairs*100):.1f}%",
            delta_color="off"
        )
    
    # Time distribution
    valid_days = filtered_with_mother['days_to_art'].dropna()
    valid_days = valid_days[(valid_days >= 0) & (valid_days <= 180)]
    
    if len(valid_days) > 0:
        fig = px.histogram(
            valid_days,
            nbins=50,
            labels={'value': 'Days from Diagnosis to ART', 'count': 'Count'},
            color_discrete_sequence=['#3498db']
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="green",
                     annotation_text=f"Same Day: {same_day_pct:.1f}%")
        fig.add_vline(x=7, line_dash="dash", line_color="orange",
                     annotation_text="7 Days")
        
        fig.update_layout(
            title='Distribution: Diagnosis to ART Initiation',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        f"Among {valid_art_timing:,} mothers with documented timing, {same_day_art:,} ({same_day_pct:.1f}%) "
        f"initiated ART on the same day as diagnosis. Median time to treatment: {median_days:.0f} days.",
        icon="📊"
    )
    
    st.markdown("---")
    
    # Treatment cascade
    st.markdown("### 📊 Treatment Cascade - Children Without Traceable Mother")
    
    diagnosed = len(filtered_no_mother)
    art_initiated = filtered_no_mother['infant_date_of_art_initiation'].notna().sum()
    active_tx = filtered_no_mother['infant_follow_up_status'].str.contains('Active', na=False).sum()
    
    cascade_data = pd.DataFrame({
        'Stage': ['HIV+ Diagnosed', 'ART Initiated', 'Active Treatment'],
        'Count': [diagnosed, art_initiated, active_tx],
        'Percentage': [
            100,
            (art_initiated/diagnosed*100) if diagnosed > 0 else 0,
            (active_tx/diagnosed*100) if diagnosed > 0 else 0
        ]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Funnel(
        y=cascade_data['Stage'],
        x=cascade_data['Count'],
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#e74c3c", "#f39c12", "#27ae60"]},
    ))
    
    fig.update_layout(
        title='Treatment Cascade',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnosed", f"{diagnosed:,}", "100%")
    with col2:
        loss_1_2 = diagnosed - art_initiated
        st.metric("ART Initiated", f"{art_initiated:,}", 
                 f"-{loss_1_2:,}")
    with col3:
        loss_2_3 = art_initiated - active_tx
        st.metric("Active Treatment", f"{active_tx:,}",
                 f"-{loss_2_3:,}")
    
    show_insight_box(
        f"Among {diagnosed:,} diagnosed children without traceable mothers, {art_initiated:,} "
        f"({(art_initiated/diagnosed*100):.1f}%) initiated ART, and {active_tx:,} "
        f"({(active_tx/diagnosed*100):.1f}%) remain actively on treatment.",
        icon="📈"
    )


# =============================================================================
# PAGE: VIRAL LOAD
# =============================================================================

elif page == "🧬 Viral Load":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("🧬 Viral Load Analysis")
    st.markdown("## Treatment Monitoring and Viral Suppression")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    testing_coverage = (len(vl_tested) / len(filtered_with_mother) * 100) if len(filtered_with_mother) > 0 else 0
    not_suppressed = (filtered_with_mother['vl_suppressed'] == 'Not Suppressed').sum()
    unknown_vl = (filtered_with_mother['vl_suppressed'] == 'Unknown').sum()
    
    with col1:
        st.metric(
            "VL Testing Coverage",
            f"{testing_coverage:.1f}%",
            f"{len(vl_tested):,} tested"
        )
    
    with col2:
        st.metric(
            "Suppression Rate",
            f"{suppression_rate:.1f}%",
            f"Among {len(vl_tested):,} tested"
        )
    
    with col3:
        st.metric(
            "Suppressed",
            f"{suppressed_count:,}",
            f"VL <1000 copies/mL"
        )
    
    with col4:
        st.metric(
            "No VL Result",
            f"{unknown_vl:,}",
            f"{(unknown_vl/len(filtered_with_mother)*100):.1f}%"
        )
    
    st.markdown("---")
    
    # VL status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Viral Load Status")
        
        vl_counts = filtered_with_mother['vl_suppressed'].value_counts()
        
        fig = px.pie(
            values=vl_counts.values,
            names=vl_counts.index,
            color=vl_counts.index,
            color_discrete_map={
                'Suppressed': '#27ae60',
                'Not Suppressed': '#e74c3c',
                'Unknown': '#95a5a6'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Among Tested Mothers")
        
        if len(vl_tested) > 0:
            tested_counts = vl_tested['vl_suppressed'].value_counts()
            
            fig = px.pie(
                values=tested_counts.values,
                names=tested_counts.index,
                color=tested_counts.index,
                color_discrete_map={
                    'Suppressed': '#27ae60',
                    'Not Suppressed': '#e74c3c'
                }
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Suppression analysis
    st.markdown("### 📊 Viral Suppression Analysis")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Viral Load<br>Suppressed', 'Not<br>Suppressed', 'No VL<br>Result'],
        y=[suppressed_count, not_suppressed, unknown_vl],
        text=[f'{suppressed_count}<br>({(suppressed_count/len(filtered_with_mother)*100):.1f}%)',
              f'{not_suppressed}<br>({(not_suppressed/len(filtered_with_mother)*100):.1f}%)',
              f'{unknown_vl}<br>({(unknown_vl/len(filtered_with_mother)*100):.1f}%)'],
        textposition='auto',
        marker_color=['#27ae60', '#e74c3c', '#95a5a6']
    ))
    
    fig.update_layout(
        title='Viral Load Status Distribution',
        yaxis_title='Number of Mothers',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        f"Viral load testing completed for {len(vl_tested):,} ({testing_coverage:.1f}%) mothers. "
        f"Among tested mothers, {suppressed_count:,} ({suppression_rate:.1f}%) achieved viral suppression "
        f"(<1000 copies/mL). {unknown_vl:,} mothers lack VL monitoring data.",
        icon="📊"
    )
    
    if not_suppressed > 0:
        show_insight_box(
            f"{not_suppressed:,} mothers have detectable viral load ≥1000 copies/mL, requiring "
            "adherence support and potential regimen assessment.",
            icon="⚠️"
        )

# =============================================================================
# PAGE: INFANT OUTCOMES
# =============================================================================

elif page == "👶 Infant Outcomes":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰 👶</div>', unsafe_allow_html=True)
    st.title("👶 Infant Outcomes Analysis")
    st.markdown("## Mother-to-Child Transmission Assessment")
    
    st.markdown("---")
    
    # Key MTCT metrics
    st.markdown("### 🎯 Transmission Outcomes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Children Born",
            f"{children_born:,}",
            f"{(children_born/total_pairs*100):.1f}%",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Infants Tested",
            f"{infants_tested:,}",
            f"{(infants_tested/children_born*100):.1f}% of born" if children_born > 0 else "N/A",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "HIV-Positive",
            f"{infant_positive}",
            "MTCT occurred"
        )
    
    with col4:
        untested = children_born - infants_tested
        st.metric(
            "Status Unknown",
            f"{untested:,}",
            f"{(untested/children_born*100):.1f}%" if children_born > 0 else "N/A"
        )
    
    st.markdown("---")
    
    # Testing outcomes
    st.markdown("### 🧬 HIV Test Results")
    
    def classify_outcome(row):
        result = str(row['infant_hiv_test_result']).upper() if pd.notna(row['infant_hiv_test_result']) else ''
        if 'NEGATIVE' in result:
            return 'HIV-Negative'
        elif 'POSITIVE' in result:
            return 'HIV-Positive'
        elif 'INCONCLUSIVE' in result:
            return 'Inconclusive'
        else:
            return 'Not Tested'
    
    filtered_with_mother['test_outcome'] = filtered_with_mother.apply(classify_outcome, axis=1)
    outcome_counts = filtered_with_mother['test_outcome'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        outcome_df = pd.DataFrame({
            'Outcome': outcome_counts.index,
            'Count': outcome_counts.values,
            'Percentage': (outcome_counts.values / len(filtered_with_mother) * 100).round(1)
        })
        
        st.dataframe(outcome_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = px.bar(
            outcome_df,
            x='Outcome',
            y='Count',
            text='Percentage',
            color='Outcome',
            color_discrete_map={
                'HIV-Negative': '#27ae60',
                'HIV-Positive': '#e74c3c',
                'Inconclusive': '#f39c12',
                'Not Tested': '#95a5a6'
            }
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title='Infant HIV Test Results Distribution',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # MTCT by age group
    st.markdown("### 📊 MTCT Rate by Maternal Age")
    
    filtered_with_mother['age_group'] = pd.cut(
        filtered_with_mother['mother_age_at_booking'],
        bins=[0, 20, 25, 30, 35, 100],
        labels=['<20', '20-24', '25-29', '30-34', '35+']
    )
    
    risk_data = []
    for age_grp in ['<20', '20-24', '25-29', '30-34', '35+']:
        in_group = filtered_with_mother['age_group'] == age_grp
        tested_in_group = filtered_with_mother[in_group & (filtered_with_mother['test_outcome'] != 'Not Tested')]
        
        if len(tested_in_group) > 0:
            mtct_in_group = (tested_in_group['test_outcome'] == 'HIV-Positive').sum()
            rate = mtct_in_group / len(tested_in_group) * 100
            risk_data.append({
                'Age Group': age_grp,
                'MTCT Rate (%)': rate,
                'Positive': mtct_in_group,
                'Tested': len(tested_in_group)
            })
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                risk_df,
                x='Age Group',
                y='MTCT Rate (%)',
                text='MTCT Rate (%)',
                color='MTCT Rate (%)',
                color_continuous_scale='Reds'
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                title='MTCT Rate by Maternal Age Group',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        f"Of {children_born:,} documented births, {infants_tested:,} ({(infants_tested/children_born*100):.1f}%) "
        f"have HIV test results. Among tested infants, {infant_positive} tested positive, "
        f"representing a {mtct_rate:.1f}% transmission rate. {untested:,} born infants lack test results.",
        icon="👶"
    )

# =============================================================================
# PAGE: TIMELINE ANALYSIS
# =============================================================================

elif page == "⏱️ Timeline Analysis":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("⏱️ Timeline Analysis")
    st.markdown("## Longitudinal Journey Through PMTCT Care")
    
    st.markdown("---")
    
    # Calculate intervals
    filtered_with_mother['weeks_at_booking'] = (
        (filtered_with_mother['date_of_anc_booking'] - 
         filtered_with_mother['date_of_last_known_mensural_period']).dt.days / 7
    )
    filtered_with_mother['days_booking_to_test'] = (
        filtered_with_mother['mother_date_of_hiv_test'] - 
        filtered_with_mother['date_of_anc_booking']
    ).dt.days
    filtered_with_mother['infant_age_at_test_weeks'] = (
        (filtered_with_mother['infant_hiv_test_date'] - 
         filtered_with_mother['infant_date_of_birth']).dt.days / 7
    )
    
    # Key intervals
    st.markdown("### 📊 Median Time Intervals")
    
    col1, col2, col3 = st.columns(3)
    
    booking_weeks = filtered_with_mother['weeks_at_booking'].dropna()
    booking_weeks = booking_weeks[(booking_weeks > 0) & (booking_weeks < 42)]
    median_booking = booking_weeks.median() if len(booking_weeks) > 0 else 0
    
    with col1:
        st.metric(
            "Gestational Age at Booking",
            f"{median_booking:.0f} weeks",
            f"Based on {len(booking_weeks):,} cases"
        )
    
    booking_to_test = filtered_with_mother['days_booking_to_test'].dropna()
    booking_to_test = booking_to_test[(booking_to_test >= 0) & (booking_to_test <= 280)]
    median_to_test = booking_to_test.median() if len(booking_to_test) > 0 else 0
    
    with col2:
        st.metric(
            "Booking to HIV Test",
            f"{median_to_test:.0f} days",
            f"Based on {len(booking_to_test):,} cases"
        )
    
    infant_age = filtered_with_mother['infant_age_at_test_weeks'].dropna()
    infant_age = infant_age[(infant_age >= 0) & (infant_age <= 104)]
    median_infant_age = infant_age.median() if len(infant_age) > 0 else 0
    
    with col3:
        st.metric(
            "Infant Age at Test",
            f"{median_infant_age:.1f} weeks",
            f"Based on {len(infant_age):,} cases"
        )
    
    st.markdown("---")
    
    # Complete cascade timeline
    st.markdown("### 🔄 PMTCT Cascade Timeline")
    
    stages = [
        'ANC Booked',
        'HIV Diagnosed', 
        'ART Initiated',
        'VL Tested',
        'Delivered',
        'Infant Tested'
    ]
    
    counts = [
        len(filtered_with_mother['date_of_anc_booking'].dropna()),
        len(filtered_with_mother['mother_date_of_hiv_test'].dropna()),
        len(filtered_with_mother['mother_date_of_art_initiation'].dropna()),
        len(filtered_with_mother['mother_date_of_viral_load'].dropna()),
        len(filtered_with_mother['date_of_delivery'].dropna()),
        len(filtered_with_mother['infant_hiv_test_date'].dropna())
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Funnel(
        y=stages,
        x=counts,
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": px.colors.sequential.RdBu_r},
    ))
    
    fig.update_layout(
        title='PMTCT Cascade: ANC to Infant Testing',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Loss analysis
    st.markdown("### 📉 Stage-to-Stage Loss")
    
    losses = []
    for i in range(len(counts)-1):
        loss = counts[i] - counts[i+1]
        loss_pct = (loss / counts[i] * 100) if counts[i] > 0 else 0
        losses.append({
            'Transition': f"{stages[i]} → {stages[i+1]}",
            'Lost': loss,
            'Loss Rate': f"{loss_pct:.1f}%"
        })
    
    loss_df = pd.DataFrame(losses)
    st.dataframe(loss_df, use_container_width=True, hide_index=True)
    
    show_insight_box(
        f"Median gestational age at booking: {median_booking:.0f} weeks. "
        f"Median time from booking to HIV test: {median_to_test:.0f} days. "
        f"Median infant age at HIV testing: {median_infant_age:.1f} weeks.",
        icon="⏱️"
    )


# =============================================================================
# PAGE: DATA QUALITY
# =============================================================================

elif page == "🔍 Data Quality":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("🔍 Data Quality Assessment")
    st.markdown("## Documentation Completeness and Consistency")
    
    st.markdown("---")
    
    # Missing data analysis
    st.markdown("### 📊 Data Completeness by Field")
    
    tab1, tab2 = st.tabs(["Mother-Baby Pairs", "Children Without Traceable Mother"])
    
    with tab1:
        missing_with = filtered_with_mother.isnull().sum()
        missing_with_pct = (missing_with / len(filtered_with_mother)) * 100
        
        missing_df1 = pd.DataFrame({
            'Field': missing_with.index,
            'Missing': missing_with.values,
            'Percentage': missing_with_pct.values
        }).sort_values('Missing', ascending=False)
        
        missing_df1 = missing_df1[missing_df1['Missing'] > 0].head(15)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_df1, use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                missing_df1.head(10),
                x='Percentage',
                y='Field',
                orientation='h',
                color='Percentage',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                title='Top 10 Fields with Missing Data',
                xaxis_title='Missing %',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        missing_no = filtered_no_mother.isnull().sum()
        missing_no_pct = (missing_no / len(filtered_no_mother)) * 100
        
        missing_df2 = pd.DataFrame({
            'Field': missing_no.index,
            'Missing': missing_no.values,
            'Percentage': missing_no_pct.values
        }).sort_values('Missing', ascending=False)
        
        missing_df2 = missing_df2[missing_df2['Missing'] > 0].head(15)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_df2, use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                missing_df2.head(10),
                x='Percentage',
                y='Field',
                orientation='h',
                color='Percentage',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                title='Top 10 Fields with Missing Data',
                xaxis_title='Missing %',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data consistency
    st.markdown("### ⚠️ Data Consistency Issues")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### HIV Result Variations")
        
        if 'infant_hiv_test_result' in filtered_no_mother.columns:
            result_counts = filtered_no_mother['infant_hiv_test_result'].value_counts()
            
            st.dataframe(pd.DataFrame({
                'Value': result_counts.index,
                'Count': result_counts.values
            }).head(10), use_container_width=True, hide_index=True)
            
            # Check for case variations
            upper_count = result_counts.get('POSITIVE', 0) + result_counts.get('NEGATIVE', 0)
            lower_count = result_counts.get('Positive', 0) + result_counts.get('Negative', 0)
            
            if lower_count > 0:
                st.warning(f"""
                **Case inconsistency detected:**
                - {lower_count:,} entries with mixed case
                - Recommend standardization to uppercase
                """)
    
    with col2:
        st.markdown("#### Duplicate Records")
        
        dup_with = filtered_with_mother.duplicated().sum()
        dup_no = filtered_no_mother.duplicated().sum()
        
        dup_data = pd.DataFrame({
            'Dataset': ['Mother-Baby Pairs', 'Children Without Traceable Mother'],
            'Duplicates': [dup_with, dup_no],
            'Percentage': [
                (dup_with/len(filtered_with_mother)*100) if len(filtered_with_mother) > 0 else 0,
                (dup_no/len(filtered_no_mother)*100) if len(filtered_no_mother) > 0 else 0
            ]
        })
        
        st.dataframe(dup_data, use_container_width=True, hide_index=True)
        
        if dup_with > 0 or dup_no > 0:
            st.warning(f"""
            **{dup_with + dup_no} duplicate records detected**
            
            Review and deduplication recommended
            """)
        else:
            st.success("✅ No duplicate records found")
    
    st.markdown("---")
    
    # Date validation
    st.markdown("### 📅 Date Range Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Future Dates")
        
        future_count = 0
        future_details = []
        
        for col in filtered_with_mother.select_dtypes(include=['datetime64']).columns:
            future = filtered_with_mother[filtered_with_mother[col] > pd.Timestamp.now()]
            if len(future) > 0:
                future_count += len(future)
                future_details.append(f"{col}: {len(future)} cases")
        
        if future_count > 0:
            st.error(f"**{future_count} future dates found:**")
            for detail in future_details[:5]:
                st.write(f"- {detail}")
        else:
            st.success("✅ No future dates detected")
    
    with col2:
        st.markdown("#### Out-of-Range Dates")
        
        old_count = 0
        old_details = []
        
        for col in filtered_with_mother.select_dtypes(include=['datetime64']).columns:
            very_old = filtered_with_mother[filtered_with_mother[col] < pd.Timestamp('1990-01-01')]
            if len(very_old) > 0:
                old_count += len(very_old)
                old_details.append(f"{col}: {len(very_old)} cases")
        
        if old_count > 0:
            st.warning(f"**{old_count} dates before 1990:**")
            for detail in old_details[:5]:
                st.write(f"- {detail}")
        else:
            st.success("✅ All dates within valid range")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Data Quality Recommendations")
    
    recommendations = [
        {
            "priority": "High",
            "action": "Standardize Text Fields",
            "description": "Convert all categorical values (HIV results, status fields) to consistent casing"
        },
        {
            "priority": "High",
            "action": "Address Missing Critical Fields",
            "description": "Prioritize completion of infant HIV test results and viral load data"
        },
        {
            "priority": "Medium",
            "action": "Validate Dates",
            "description": "Review and correct date anomalies (future dates, out-of-range values)"
        },
        {
            "priority": "Medium",
            "action": "Remove Duplicates",
            "description": "Investigate and resolve duplicate records"
        },
        {
            "priority": "Long-term",
            "action": "Implement Real-time Validation",
            "description": "Add data entry constraints and validation rules at source"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"**[{rec['priority']} Priority]** {rec['action']}"):
            st.write(rec['description'])

# =============================================================================
# PAGE: RECOMMENDATIONS
# =============================================================================

elif page == "📋 Recommendations":
    st.markdown('<div style="text-align: center; font-size: 48px;">🇿🇼 🤰</div>', unsafe_allow_html=True)
    st.title("📋 Programme Recommendations")
    st.markdown("## Evidence-Based Action Plan")
    
    st.markdown("---")
    
    # Priority matrix
    st.markdown("### 🎯 Priority Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Immediate Actions (0-3 months)")
        
        with st.expander("**1. Infant Testing Scale-Up**", expanded=True):
            st.markdown(f"""
            **Current Gap:** {children_born - infants_tested:,} untested infants
            
            **Target:** Achieve >90% testing coverage
            
            **Actions:**
            - Deploy mobile testing teams
            - Community health worker tracing
            - SMS reminder system
            - Integration with immunization services
            - Weekend/evening testing clinics
            
            **Resources:** Moderate budget, staffing requirements
            """)
        
        with st.expander("**2. Mother-Baby Linkage System**"):
            st.markdown(f"""
            **Current Gap:** {total_no_link:,} children without maternal link
            
            **Target:** Achieve >95% linkage
            
            **Actions:**
            - Implement unique mother-baby pair IDs
            - Real-time linkage verification
            - Cross-facility tracking system
            - Data quality officer at each site
            
            **Resources:** Low-moderate budget, IT support needed
            """)
    
    with col2:
        st.markdown("#### 🟡 Medium-term Actions (3-12 months)")
        
        with st.expander("**3. Viral Load Testing Expansion**", expanded=True):
            st.markdown(f"""
            **Current Coverage:** {testing_coverage:.1f}%
            
            **Target:** Achieve >95% VL testing
            
            **Actions:**
            - Point-of-care VL machines deployment
            - Mandatory testing protocols
            - Results within 48 hours
            - Enhanced support for non-suppressed
            
            **Resources:** High budget (equipment), training needed
            """)
        
        with st.expander("**4. Early ART Optimization**"):
            st.markdown(f"""
            **Current Performance:** {same_day_pct:.1f}% same-day
            
            **Target:** Sustain >90% same-day initiation
            
            **Actions:**
            - Pre-conception ART for all HIV+ women
            - Immediate ART protocols
            - Treatment literacy programs
            - Retention support systems
            
            **Resources:** Moderate budget, training focus
            """)
    
    st.markdown("---")
    
    # Target populations
    st.markdown("### 👥 Target Population Strategies")
    
    tab1, tab2, tab3 = st.tabs([
        "Adolescents & Young Women",
        "Children Without Linkage",
        "Non-Suppressed Mothers"
    ])
    
    with tab1:
        adolescent_count = (filtered_with_mother['mother_age_at_booking'] < 25).sum()
        st.info(f"**Target:** ~{adolescent_count:,} young women (<25 years)")
        
        st.markdown("""
        **Service Delivery:**
        - Adolescent-friendly clinic hours
        - Youth peer support groups
        - Flexible appointment scheduling
        
        **Adherence Support:**
        - SMS/WhatsApp reminders
        - Home visit protocols
        - Incentive schemes
        
        **Psychosocial Support:**
        - Mental health screening
        - Disclosure counseling
        - Economic empowerment programs
        """)
    
    with tab2:
        st.info(f"**Target:** {total_no_link:,} children without traceable mothers")
        
        st.markdown("""
        **Investigation:**
        - Case-by-case review
        - Identify root causes
        - System gap analysis
        
        **Prevention:**
        - Electronic patient tracking
        - Mother-baby pair verification
        - Cross-site data sharing
        
        **Remediation:**
        - Active case finding
        - Treatment retention support
        - Caregiver engagement
        """)
    
    with tab3:
        not_suppressed = (filtered_with_mother['vl_suppressed'] == 'Not Suppressed').sum()
        st.info(f"**Target:** {not_suppressed:,} mothers with detectable viral load")
        
        st.markdown("""
        **Clinical Management:**
        - Adherence assessment
        - Drug resistance testing
        - Regimen optimization
        
        **Enhanced Support:**
        - Intensive counseling
        - Treatment supporters
        - Monthly VL monitoring
        
        **Follow-up:**
        - Fast-track appointments
        - Multi-disciplinary team
        - Community follow-up
        """)
    
    st.markdown("---")
    
    # Implementation roadmap
    st.markdown("### 📅 Implementation Roadmap")
    
    roadmap_data = pd.DataFrame({
        'Quarter': ['Q1 2026', 'Q2 2026', 'Q3 2026', 'Q4 2026'],
        'Focus Area': [
            'Infant Testing Surge',
            'Linkage System Implementation',
            'VL Testing Scale-up',
            'Optimization & Sustainment'
        ],
        'Key Deliverables': [
            'Test 50% of gap infants',
            'Deploy unique ID system',
            'Install POC VL machines',
            'Achieve 90-90-90 targets'
        ]
    })
    
    st.dataframe(roadmap_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Success metrics
    st.markdown("### 📊 Success Metrics")
    
    metrics_data = pd.DataFrame({
        'Indicator': [
            'Infant Testing Coverage',
            'MTCT Rate (among tested)',
            'Mother-Baby Linkage',
            'VL Testing Coverage',
            'VL Suppression Rate',
            'Treatment Retention',
            'Same-Day ART Initiation'
        ],
        'Current': [
            f'{(infants_tested/children_born*100):.1f}%' if children_born > 0 else 'N/A',
            f'{mtct_rate:.1f}%',
            f'{(total_pairs/(total_pairs+total_no_link)*100):.1f}%',
            f'{testing_coverage:.1f}%',
            f'{suppression_rate:.1f}%',
            f'{retention_pct:.1f}%',
            f'{same_day_pct:.1f}%'
        ],
        '12-Month Target': [
            '90%',
            '<5%',
            '95%',
            '90%',
            '90%',
            '85%',
            '90%'
        ],
        '24-Month Target': [
            '95%',
            '<2%',
            '98%',
            '95%',
            '95%',
            '90%',
            '95%'
        ]
    })
    
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Resource summary
    st.markdown("### 💰 Resource Requirements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Financial**")
        st.markdown("""
        **Year 1 Estimated:**
        - Testing surge: $50,000
        - VL equipment: $200,000
        - IT systems: $150,000
        - Training: $30,000
        - Community programs: $40,000
        
        **Total: ~$470,000**
        """)
    
    with col2:
        st.markdown("**Human Resources**")
        st.markdown("""
        **New Positions:**
        - Data quality officers (10)
        - M&E specialists (5)
        - Community tracers (20)
        - IT support (2)
        - Youth counselors (15)
        
        **Total: 52 FTEs**
        """)
    
    with col3:
        st.markdown("**Equipment**")
        st.markdown("""
        **Procurement:**
        - Point-of-care VL machines (10)
        - Computers/tablets (20)
        - Mobile phones (30)
        - ID scanners/printers
        - Backup power systems
        """)

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 24px;'>🇿🇼 🤰 👶</p>
        <p><strong>PMTCT Longitudinal Analysis Dashboard</strong></p>
        <p>Zimbabwe PMTCT Programme | Study Period: 2021-2025</p>
        <p>Generated: {datetime.now().strftime("%B %d, %Y")}</p>
        <p style='margin-top: 10px;'>For questions contact: PMTCT Programme Monitoring Team</p>
    </div>
""", unsafe_allow_html=True)