"""
PMTCT STAKEHOLDER DASHBOARD
============================
Interactive Streamlit dashboard for presenting PMTCT longitudinal analysis
to stakeholders with clear visualizations and insights.

To run:
    streamlit run pmtct_stakeholder_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PMTCT Longitudinal Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-success {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .metric-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .metric-danger {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .insight-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    h1 {
        color: #1e3a8a;
    }
    h2 {
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 10px;
    }
    h3 {
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data():
    """Load and prepare datasets"""
    try:
        no_mother = pd.read_csv('uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv')
        with_mother = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
        
        # Convert dates for no_mother
        date_cols_no_mother = ['infant_date_of_birth', 'infant_hiv_test_date', 
                               'infant_date_of_art_initiation', 'infant_date_of_art_enrolment']
        for col in date_cols_no_mother:
            if col in no_mother.columns:
                no_mother[col] = pd.to_datetime(no_mother[col], errors='coerce', dayfirst=True)
        
        # Convert dates for with_mother
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

def create_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """Create a styled metric card"""
    col = st.columns(1)[0]
    col.metric(label=title, value=value, delta=delta, delta_color=delta_color, help=help_text)

def show_insight_box(text, icon="💡"):
    """Display an insight box"""
    st.markdown(f"""
        <div class="insight-box">
            <strong>{icon} Key Insight:</strong> {text}
        </div>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Section:",
    [
        "🏠 Executive Summary",
        "📈 Study Overview",
        "👥 Maternal Demographics", 
        "💊 ART Initiation & Cascade",
        "🧬 Viral Load & Suppression",
        "👶 Infant Outcomes & MTCT",
        "⏱️ Longitudinal Timeline",
        "🔍 Data Quality Issues",
        "📋 Key Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Study Period:** 2021 - 2025

**Total Cohort:** 3,092 records
- 1,881 Mother-Baby Pairs
- 1,211 Children (No Maternal Link)
""")

# Load data
no_mother, with_mother = load_data()

if no_mother is None or with_mother is None:
    st.error("Failed to load data. Please ensure CSV files are in the correct directory.")
    st.stop()

# Calculate key metrics
total_pairs = len(with_mother)
infants_tested = with_mother['infant_hiv_test_result'].notna().sum()
infant_positive = with_mother['infant_hiv_test_result'].str.upper().isin(['POSITIVE']).sum()
mtct_rate = (infant_positive / infants_tested * 100) if infants_tested > 0 else 0

# PAGE CONTENT
# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================
if page == "🏠 Executive Summary":
    st.title("🏥 PMTCT Longitudinal Analysis")
    st.markdown("## Prevention of Mother-to-Child Transmission Programme")
    st.markdown("### Study Period: 2021 - 2025")
    
    st.markdown("---")
    
    # Key Metrics Dashboard
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Mother-Baby Pairs",
            value=f"{total_pairs:,}",
            help="Complete linkage between mothers and infants"
        )
    
    with col2:
        st.metric(
            label="Infant Testing Coverage",
            value=f"{(infants_tested/total_pairs*100):.1f}%",
            delta=f"{infants_tested:,} tested",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            label="MTCT Rate",
            value=f"{mtct_rate:.1f}%",
            delta="Meets WHO Target" if mtct_rate <= 5 else "Above Target",
            delta_color="normal" if mtct_rate <= 5 else "inverse",
            help="Mother-to-Child Transmission rate among tested infants"
        )
    
    with col4:
        # Calculate same-day ART rate
        with_mother['days_to_art'] = (with_mother['mother_date_of_art_initiation'] - 
                                       with_mother['date_mother_tested_positive']).dt.days
        same_day = (with_mother['days_to_art'] == 0).sum()
        valid_timing = with_mother['days_to_art'].notna().sum()
        same_day_pct = (same_day / valid_timing * 100) if valid_timing > 0 else 0
        
        st.metric(
            label="Same-Day ART Initiation",
            value=f"{same_day_pct:.1f}%",
            delta="World-Class",
            delta_color="normal",
            help="Mothers who started ART on diagnosis day"
        )
    
    st.markdown("---")
    
    # Quick Wins and Critical Gaps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ **Major Successes**")
        st.success(f"**{same_day_pct:.1f}% Same-Day ART Initiation** - World-class performance in Test & Treat")
        st.success(f"**{mtct_rate:.1f}% MTCT Rate** - Meets WHO elimination target (<5%)")
        st.success(f"**100% HIV Testing** - All pregnant women tested at ANC")
        
        # Viral suppression
        with_mother['vl_suppressed'] = with_mother['mother_viral_load_result'].apply(
            lambda x: 'Suppressed' if pd.notna(x) and (str(x).upper() in ['TND', '<30', '<20', '<50'] or 
                     (isinstance(x, (int, float)) and x < 1000)) else 
                     'Not Suppressed' if pd.notna(x) else 'Unknown'
        )
        vl_tested = with_mother[with_mother['vl_suppressed'] != 'Unknown']
        if len(vl_tested) > 0:
            suppressed = (vl_tested['vl_suppressed'] == 'Suppressed').sum()
            suppression_rate = suppressed / len(vl_tested) * 100
            st.success(f"**{suppression_rate:.1f}% Viral Suppression** - Among mothers with VL results")
    
    with col2:
        st.markdown("### 🚨 **Critical Gaps**")
        untested_pct = ((total_pairs - infants_tested) / total_pairs * 100)
        st.error(f"**{untested_pct:.1f}% Infants Untested** - {total_pairs - infants_tested:,} infants with unknown HIV status")
        
        # Children without maternal link
        orphan_count = len(no_mother)
        st.error(f"**{orphan_count:,} Children Without Maternal Link** - System breakdown in documentation")
        
        # Retention
        active_treatment = no_mother['infant_follow_up_status'].str.contains('Active', na=False).sum()
        retention_rate = (active_treatment / orphan_count * 100)
        st.error(f"**{retention_rate:.1f}% Treatment Retention** - Only {active_treatment:,}/{orphan_count:,} children on active treatment")
        
        # VL testing gap
        no_vl = len(with_mother) - len(vl_tested)
        st.error(f"**{(no_vl/len(with_mother)*100):.1f}% No Viral Load** - {no_vl:,} mothers without VL monitoring")
    
    st.markdown("---")
    
    # Executive Insights
    st.markdown("## 💡 Executive Insights")
    
    show_insight_box(
        "The program excels at same-day ART initiation (73.2%) but struggles with the "
        "'last mile' - only 13% of infants have documented HIV test results. The true "
        "population-level MTCT rate remains unknown.",
        icon="🎯"
    )
    
    show_insight_box(
        "ZERO transmissions occurred among mothers who started ART BEFORE pregnancy. "
        "Pre-conception ART is the gold standard for PMTCT.",
        icon="⭐"
    )
    
    show_insight_box(
        "1,211 HIV-positive children have NO traceable maternal data, indicating severe "
        "system breakdown in mother-baby pair registration and follow-up.",
        icon="⚠️"
    )
    
    st.markdown("---")
    
    # Strategic Priorities
    st.markdown("## 🎯 Strategic Priorities")
    
    priorities = [
        {
            "priority": "1. Infant Testing Surge",
            "target": "Test all 1,637 infants with unknown status within 6 months",
            "actions": ["Mobile testing units", "Community tracing", "SMS reminders", "Integration with immunization"],
            "impact": "HIGH"
        },
        {
            "priority": "2. Strengthen Mother-Baby Linkage", 
            "target": "Reduce unlinked children from 1,211 to <100",
            "actions": ["Electronic unique IDs", "Cross-facility tracking", "Real-time dashboards"],
            "impact": "HIGH"
        },
        {
            "priority": "3. Scale Pre-Conception ART",
            "target": "Initiate all HIV+ women of reproductive age on ART",
            "actions": ["Identify all HIV+ women 15-49", "Immediate ART", "Family planning integration"],
            "impact": "CRITICAL"
        },
        {
            "priority": "4. Universal Viral Load Testing",
            "target": "Achieve 95% VL testing coverage",
            "actions": ["Point-of-care VL machines", "Mandatory 28-week VL", "Results-to-action protocols"],
            "impact": "HIGH"
        }
    ]
    
    for p in priorities:
        with st.expander(f"**{p['priority']}** - {p['impact']} IMPACT"):
            st.markdown(f"**Target:** {p['target']}")
            st.markdown("**Key Actions:**")
            for action in p['actions']:
                st.markdown(f"- {action}")

# ============================================================================
# STUDY OVERVIEW
# ============================================================================
elif page == "📈 Study Overview":
    st.title("📈 Study Overview")
    st.markdown("## PMTCT Longitudinal Journey Analysis")
    
    st.markdown("""
    This study focuses on **mothers living with HIV** who were diagnosed either prior to 
    or during their antenatal care (ANC).
    
    **Study Period:** 2021 - 2025  
    **Data Collection:** Zimbabwe PMTCT Programme Monitoring System
    """)
    
    st.markdown("---")
    
    # Dataset Description
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create summary table
        summary_data = {
            "Metric": [
                "Total Mother-Baby Pairs",
                "Infants with HIV Results",
                "Median Age at Testing",
                "Infant Positivity (MTCT Rate)",
                "Children Without Maternal Link"
            ],
            "Value": [
                f"{total_pairs:,}",
                f"{infants_tested:,} (13.0%)",
                "6.9 weeks",
                f"4.9% ({infant_positive} babies)",
                f"{len(no_mother):,}"
            ],
            "Interpretation": [
                "Sample size with complete mother-baby linkage",
                "Only 13% have final HIV test results recorded",
                "Major peak around 6-7 weeks (EID guideline)",
                "Meets WHO target (<5%) among tested infants",
                "Critical gap - no maternal data available"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🔍 Data Sources")
        st.info("""
        **Two Cohorts:**
        
        1️⃣ **Mother-Baby Pairs**  
        1,881 records with complete linkage
        
        2️⃣ **Orphan Cohort**  
        1,211 HIV+ children without maternal data
        """)
    
    st.markdown("---")
    
    # Cohort Comparison
    st.markdown("## 📊 Cohort Comparison")
    
    # Create comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Mother-Baby Pairs',
        x=['Total Records', 'HIV Testing', 'ART Initiated', 'Infant Tested', 'Active Treatment'],
        y=[total_pairs, total_pairs, 611, infants_tested, 
           with_mother['mother_appointment_outcome'].str.contains('Active', na=False).sum()],
        marker_color='#3498db',
        text=[total_pairs, total_pairs, 611, infants_tested, 
              with_mother['mother_appointment_outcome'].str.contains('Active', na=False).sum()],
        textposition='auto',
    ))
    
    orphan_total = len(no_mother)
    orphan_tested = orphan_total  # All tested (inclusion criterion)
    orphan_art = no_mother['infant_date_of_art_initiation'].notna().sum()
    orphan_active = no_mother['infant_follow_up_status'].str.contains('Active', na=False).sum()
    
    fig.add_trace(go.Bar(
        name='Children Without Maternal Link',
        x=['Total Records', 'HIV Testing', 'ART Initiated', 'Infant Tested', 'Active Treatment'],
        y=[orphan_total, orphan_total, orphan_art, orphan_total, orphan_active],
        marker_color='#e74c3c',
        text=[orphan_total, orphan_total, orphan_art, orphan_total, orphan_active],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Cascade Comparison: Both Cohorts',
        xaxis_title='Cascade Stage',
        yaxis_title='Number of Individuals',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        "The mother-baby pair cohort shows excellent early cascade performance but "
        "massive drop-off at infant testing (87% loss). The orphan cohort shows poor "
        "retention with only 48.8% actively on treatment."
    )

# ============================================================================
# MATERNAL DEMOGRAPHICS
# ============================================================================
elif page == "👥 Maternal Demographics":
    st.title("👥 Maternal Demographics")
    st.markdown("## Understanding the Women We Serve")
    
    st.markdown("---")
    
    # Key Stats
    col1, col2, col3, col4 = st.columns(4)
    
    mean_age = with_mother['mother_age_at_booking'].mean()
    adolescent_count = (with_mother['mother_age_at_booking'] < 20).sum()
    first_time = (with_mother['first_time_booking'] == True).sum()
    
    with col1:
        st.metric("Mean Maternal Age", f"{mean_age:.1f} years")
    with col2:
        st.metric("Adolescents (<20)", f"{adolescent_count:,}", 
                 delta=f"{(adolescent_count/total_pairs*100):.1f}%")
    with col3:
        st.metric("First-Time ANC", f"{first_time:,}",
                 delta=f"{(first_time/total_pairs*100):.1f}%")
    with col4:
        st.metric("Age Range", f"{with_mother['mother_age_at_booking'].min():.0f}-{with_mother['mother_age_at_booking'].max():.0f} yrs")
    
    st.markdown("---")
    
    # Age Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Age Distribution at ANC Booking")
        
        # Create age histogram
        fig = px.histogram(
            with_mother, 
            x='mother_age_at_booking',
            nbins=30,
            labels={'mother_age_at_booking': 'Age at Booking (years)', 'count': 'Number of Mothers'},
            color_discrete_sequence=['#3498db']
        )
        
        fig.add_vline(x=mean_age, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_age:.1f} yrs")
        fig.add_vline(x=20, line_dash="dash", line_color="orange",
                     annotation_text="High-Risk: <20 yrs")
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age Categories")
        
        # Create age groups
        with_mother['age_group'] = pd.cut(
            with_mother['mother_age_at_booking'],
            bins=[0, 19, 24, 29, 34, 100],
            labels=['<20', '20-24', '25-29', '30-34', '35+']
        )
        
        age_dist = with_mother['age_group'].value_counts().sort_index()
        
        fig = px.pie(
            values=age_dist.values,
            names=age_dist.index,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Maternal HIV Status at Booking
    st.markdown("### 🔍 HIV Status at ANC Booking")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Calculate categories
        new_diagnosis = 1385  # From PDF
        known_positive = 496  # From PDF
        
        status_data = pd.DataFrame({
            'Category': ['New Diagnosis\n(Found at Clinic)', 'Known Positive\n(Arrived HIV+)'],
            'Count': [new_diagnosis, known_positive],
            'Percentage': [73.6, 26.4]
        })
        
        st.dataframe(status_data, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **{new_diagnosis:,} mothers (73.6%)**  
        Arrived with unknown status,  
        diagnosed by clinic testing
        
        **{known_positive:,} mothers (26.4%)**  
        Already aware of HIV status
        """)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['New Diagnosis', 'Known Positive'],
            y=[new_diagnosis, known_positive],
            text=[f"{new_diagnosis}<br>({73.6}%)", f"{known_positive}<br>({26.4}%)"],
            textposition='auto',
            marker_color=['#e74c3c', '#3498db']
        ))
        
        fig.update_layout(
            title='Maternal HIV Status at First ANC Booking',
            yaxis_title='Number of Mothers',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        "73.6% of mothers were newly diagnosed at ANC - highlighting the critical "
        "importance of universal HIV testing at first antenatal visit. This represents "
        "a major opportunity for early intervention."
    )

# ============================================================================
# ART INITIATION & CASCADE
# ============================================================================
elif page == "💊 ART Initiation & Cascade":
    st.title("💊 ART Initiation & Treatment Cascade")
    st.markdown("## From Diagnosis to Sustained Treatment")
    
    st.markdown("---")
    
    # ART Timing Analysis
    st.markdown("### ⏱️ Timing of ART Initiation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # ART timing data from PDF
        timing_data = pd.DataFrame({
            'Category': [
                'Before Pregnancy',
                'During Pregnancy (>4 wks)',
                'During Pregnancy (<4 wks)',
                'Unknown Timing'
            ],
            'Count': [552, 274, 19, 1036],
            'Percentage': [29.3, 14.6, 1.0, 55.1]
        })
        
        st.dataframe(timing_data, use_container_width=True, hide_index=True)
        
        st.success("""
        **✅ 552 mothers (29.3%)**  
        Started ART BEFORE pregnancy  
        → ZERO transmissions in this group!
        """)
        
        st.warning("""
        **⚠️ 1,036 mothers (55.1%)**  
        Unknown ART timing  
        → Critical data gap
        """)
    
    with col2:
        fig = px.bar(
            timing_data,
            x='Category',
            y='Count',
            text='Percentage',
            color='Category',
            color_discrete_map={
                'Before Pregnancy': '#27ae60',
                'During Pregnancy (>4 wks)': '#3498db',
                'During Pregnancy (<4 wks)': '#f39c12',
                'Unknown Timing': '#95a5a6'
            }
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title='When Did Mothers Start ART?',
            yaxis_title='Number of Mothers',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Same-Day Initiation Success
    st.markdown("### 🎯 Same-Day ART Initiation - A Success Story")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Same-Day Initiation Rate",
            "73.2%",
            delta="World-Class Performance",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Median Delay",
            "0 Days",
            delta="Immediate Initiation",
            delta_color="normal"
        )
    
    with col3:
        # Calculate from data
        with_mother['days_to_art'] = (with_mother['mother_date_of_art_initiation'] - 
                                       with_mother['date_mother_tested_positive']).dt.days
        valid_days = with_mother['days_to_art'].dropna()
        valid_days = valid_days[valid_days >= 0]
        same_day_count = (valid_days == 0).sum()
        
        st.metric(
            "Same-Day Initiations",
            f"{same_day_count:,}",
            delta=f"out of {len(valid_days):,} tracked"
        )
    
    # Time to ART histogram
    if len(valid_days) > 0:
        fig = px.histogram(
            valid_days[valid_days <= 180],
            nbins=50,
            labels={'value': 'Days from Diagnosis to ART', 'count': 'Number of Mothers'},
            color_discrete_sequence=['#3498db']
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="green",
                     annotation_text="Same Day (73.2%)")
        fig.add_vline(x=7, line_dash="dash", line_color="orange",
                     annotation_text="1 Week")
        
        fig.update_layout(
            title='Distribution: Time from Diagnosis to ART Initiation',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        "The 73.2% same-day initiation rate demonstrates EXCELLENT implementation of "
        "'Test & Treat' policies. This is a world-class achievement reflecting strong "
        "healthcare worker training, good ARV availability, and reduced stigma.",
        icon="🏆"
    )
    
    st.markdown("---")
    
    # Treatment Cascade
    st.markdown("### 📊 The Complete Treatment Cascade")
    
    # Calculate cascade stages
    diagnosed = len(no_mother)
    art_initiated = no_mother['infant_date_of_art_initiation'].notna().sum()
    active_treatment = no_mother['infant_follow_up_status'].str.contains('Active', na=False).sum()
    
    cascade_data = {
        'Stage': ['HIV+ Diagnosed', 'ART Initiated', 'Active on Treatment'],
        'Count': [diagnosed, art_initiated, active_treatment],
        'Percentage': [100, (art_initiated/diagnosed*100), (active_treatment/diagnosed*100)]
    }
    
    cascade_df = pd.DataFrame(cascade_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Funnel(
        y=cascade_df['Stage'],
        x=cascade_df['Count'],
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#e74c3c", "#f39c12", "#27ae60"]},
    ))
    
    fig.update_layout(
        title='Treatment Cascade: Children Without Maternal Link',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stage 1: Diagnosed", f"{diagnosed:,}", "100%")
    with col2:
        loss_1_2 = diagnosed - art_initiated
        st.metric("Stage 2: ART Initiated", f"{art_initiated:,}", 
                 f"-{loss_1_2:,} lost", delta_color="inverse")
    with col3:
        loss_2_3 = art_initiated - active_treatment
        st.metric("Stage 3: Active Treatment", f"{active_treatment:,}",
                 f"-{loss_2_3:,} lost", delta_color="inverse")
    
    show_insight_box(
        f"Critical retention gap: Only {(active_treatment/diagnosed*100):.1f}% of diagnosed "
        f"children remain in active treatment. {diagnosed - active_treatment:,} children "
        "have fallen out of care - urgent tracing and re-engagement needed.",
        icon="🚨"
    )

# ============================================================================
# VIRAL LOAD & SUPPRESSION
# ============================================================================
elif page == "🧬 Viral Load & Suppression":
    st.title("🧬 Viral Load & Suppression")
    st.markdown("## Monitoring Treatment Effectiveness")
    
    st.markdown("---")
    
    # Classify viral load
    def classify_vl(vl_value):
        if pd.isna(vl_value):
            return 'Unknown'
        vl_str = str(vl_value).upper().strip()
        if vl_str in ['TND', 'TARGET NOT DETECTED', '<30', '<20', '<50', '<40']:
            return 'Suppressed'
        try:
            vl_num = float(vl_str)
            return 'Suppressed' if vl_num < 1000 else 'Not Suppressed'
        except:
            return 'Unknown'
    
    with_mother['vl_status'] = with_mother['mother_viral_load_result'].apply(classify_vl)
    
    # Calculate metrics
    vl_tested = with_mother[with_mother['vl_status'] != 'Unknown']
    suppressed_count = (with_mother['vl_status'] == 'Suppressed').sum()
    not_suppressed_count = (with_mother['vl_status'] == 'Not Suppressed').sum()
    unknown_count = (with_mother['vl_status'] == 'Unknown').sum()
    
    suppression_rate = (suppressed_count / len(vl_tested) * 100) if len(vl_tested) > 0 else 0
    testing_coverage = (len(vl_tested) / len(with_mother) * 100)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VL Testing Coverage",
            f"{testing_coverage:.1f}%",
            f"{len(vl_tested):,} tested"
        )
    
    with col2:
        st.metric(
            "Viral Suppression Rate",
            f"{suppression_rate:.1f}%",
            "Among tested" if suppression_rate >= 90 else "Below WHO target",
            delta_color="normal" if suppression_rate >= 90 else "inverse"
        )
    
    with col3:
        st.metric(
            "Suppressed Mothers",
            f"{suppressed_count:,}",
            f"U=U Protected"
        )
    
    with col4:
        st.metric(
            "No VL Result",
            f"{unknown_count:,}",
            f"{(unknown_count/len(with_mother)*100):.1f}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Viral Load Status Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Viral Load Status")
        
        vl_counts = with_mother['vl_status'].value_counts()
        
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
        st.markdown("### Among Tested Mothers Only")
        
        if len(vl_tested) > 0:
            tested_counts = vl_tested['vl_status'].value_counts()
            
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
    
    # Comparison to WHO Target
    st.markdown("### 🎯 Comparison to WHO Target")
    
    fig = go.Figure()
    
    categories = ['Current\nSuppression Rate', 'WHO\nTarget (90%)']
    values = [suppression_rate, 90]
    colors = ['#3498db' if suppression_rate >= 90 else '#f39c12', '#27ae60']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        marker_color=colors
    ))
    
    fig.add_hline(y=90, line_dash="dash", line_color="green",
                 annotation_text="WHO 90-90-90 Target")
    
    fig.update_layout(
        title='Viral Suppression vs WHO Target',
        yaxis_title='Percentage (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    if suppression_rate >= 90:
        show_insight_box(
            f"EXCELLENT: {suppression_rate:.1f}% suppression rate EXCEEDS WHO target of 90%. "
            "This demonstrates effective treatment adherence and quality ART services.",
            icon="✅"
        )
    elif suppression_rate >= 80:
        show_insight_box(
            f"GOOD: {suppression_rate:.1f}% suppression rate approaches WHO target. "
            f"Focus on the {not_suppressed_count} non-suppressed mothers for enhanced support.",
            icon="⚠️"
        )
    else:
        show_insight_box(
            f"NEEDS IMPROVEMENT: {suppression_rate:.1f}% suppression rate is BELOW WHO target. "
            f"{not_suppressed_count} mothers urgently need adherence counseling and possible regimen change.",
            icon="🚨"
        )
    
    show_insight_box(
        f"CRITICAL GAP: {unknown_count:,} mothers ({(unknown_count/len(with_mother)*100):.1f}%) "
        "have NO viral load result. Cannot assess treatment effectiveness or MTCT risk. "
        "Urgent scale-up of VL testing needed.",
        icon="🚨"
    )
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Viral Load Monitoring Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Immediate Actions")
        st.markdown("""
        - 🎯 **Scale up VL testing** to >95% coverage
        - 🏥 **Point-of-care VL machines** at high-volume sites
        - 📅 **Mandatory VL at 28 weeks** gestation
        - 🔄 **Fast-track regimen change** for treatment failures
        """)
    
    with col2:
        st.markdown("#### Support for Non-Suppressed")
        st.markdown(f"""
        - 👥 **Enhanced adherence counseling** for {not_suppressed_count} mothers
        - 🏘️ **Community adherence support** groups
        - 📱 **SMS/phone reminders** for medication
        - 🔍 **Investigate barriers** to adherence
        """)

# ============================================================================
# INFANT OUTCOMES & MTCT
# ============================================================================
elif page == "👶 Infant Outcomes & MTCT":
    st.title("👶 Infant Outcomes & MTCT")
    st.markdown("## The Ultimate Measure of PMTCT Success")
    
    st.markdown("---")
    
    # Key MTCT Metrics
    st.markdown("### 🎯 Mother-to-Child Transmission Outcomes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Infants Tested",
            f"{infants_tested:,}",
            f"{(infants_tested/total_pairs*100):.1f}% coverage"
        )
    
    with col2:
        st.metric(
            "HIV-Positive Infants",
            f"{infant_positive}",
            f"MTCT occurred"
        )
    
    with col3:
        st.metric(
            "MTCT Rate",
            f"{mtct_rate:.1f}%",
            "Meets WHO Target" if mtct_rate <= 5 else "Above Target",
            delta_color="normal" if mtct_rate <= 5 else "inverse"
        )
    
    with col4:
        untested = total_pairs - infants_tested
        st.metric(
            "Infants Untested",
            f"{untested:,}",
            f"{(untested/total_pairs*100):.1f}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Seroconversion Outcomes
    st.markdown("### 🧬 Infant Seroconversion Outcomes")
    
    # Classify outcomes
    def classify_outcome(row):
        result = str(row['infant_hiv_test_result']).upper() if pd.notna(row['infant_hiv_test_result']) else ''
        status_6mo = str(row['child_hiv_status_at_6_months']).upper() if pd.notna(row['child_hiv_status_at_6_months']) else ''
        
        if 'NEGATIVE' in result:
            if 'NEGATIVE' in status_6mo:
                return 'HIV-Negative (PMTCT Success)'
            else:
                return 'HIV-Negative (6mo unknown)'
        elif 'POSITIVE' in result or 'POSITIVE' in status_6mo:
            return 'HIV-Positive (MTCT occurred)'
        elif 'INCONCLUSIVE' in result:
            return 'Inconclusive'
        elif result == '':
            return 'Not tested'
        else:
            return 'Unknown'
    
    with_mother['seroconversion_outcome'] = with_mother.apply(classify_outcome, axis=1)
    
    outcome_counts = with_mother['seroconversion_outcome'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        outcome_df = pd.DataFrame({
            'Outcome': outcome_counts.index,
            'Count': outcome_counts.values,
            'Percentage': (outcome_counts.values / len(with_mother) * 100).round(1)
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
                'HIV-Negative (PMTCT Success)': '#27ae60',
                'HIV-Negative (6mo unknown)': '#2ecc71',
                'HIV-Positive (MTCT occurred)': '#e74c3c',
                'Inconclusive': '#f39c12',
                'Not tested': '#95a5a6',
                'Unknown': '#7f8c8d'
            }
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title='Infant HIV Status Distribution',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Timing of Transmission
    st.markdown("### ⏱️ When Did Transmission Occur?")
    
    st.markdown("""
    Based on infant age at first positive test, we can estimate when HIV transmission likely occurred:
    - **<6 weeks**: In utero or during delivery (peripartum)
    - **6 weeks - 6 months**: Early breastfeeding transmission
    - **6-12 months**: Late breastfeeding transmission
    - **>12 months**: Very late breastfeeding or new exposure
    """)
    
    # Calculate timing for positive infants
    positive_infants = with_mother[with_mother['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'].copy()
    
    if len(positive_infants) > 0:
        # Calculate age at test in weeks
        positive_infants['age_at_test_weeks'] = ((positive_infants['infant_hiv_test_date'] - 
                                                   positive_infants['infant_date_of_birth']).dt.days / 7)
        
        positive_infants['transmission_timing'] = pd.cut(
            positive_infants['age_at_test_weeks'],
            bins=[-np.inf, 6, 26, 52, np.inf],
            labels=['In utero/Peripartum\n(<6 weeks)', 
                   'Early Postnatal\n(6w-6mo)',
                   'Late Postnatal\n(6-12mo)',
                   'Very Late\n(>12mo)']
        )
        
        timing_counts = positive_infants['transmission_timing'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            timing_df = pd.DataFrame({
                'Timing': timing_counts.index,
                'Cases': timing_counts.values,
                'Percentage': (timing_counts.values / len(positive_infants) * 100).round(1)
            })
            
            st.dataframe(timing_df, use_container_width=True, hide_index=True)
            
            st.info(f"""
            **{timing_counts.iloc[0] if len(timing_counts) > 0 else 0} cases ({(timing_counts.iloc[0]/len(positive_infants)*100):.0f}%)**  
            occurred in utero or during delivery
            
            This indicates need for:
            - Better viral suppression before delivery
            - Enhanced peripartum interventions
            """)
        
        with col2:
            fig = px.bar(
                timing_df,
                x='Timing',
                y='Cases',
                text='Percentage',
                color='Timing',
                color_discrete_sequence=['#c0392b', '#e74c3c', '#e67e22', '#f39c12']
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                title='When Transmission Occurred',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Factors
    st.markdown("### 📊 MTCT Risk Factor Analysis")
    
    # By maternal age
    with_mother['age_group'] = pd.cut(
        with_mother['mother_age_at_booking'],
        bins=[0, 20, 25, 30, 35, 100],
        labels=['<20', '20-24', '25-29', '30-34', '35+']
    )
    
    risk_data = []
    for age_grp in ['<20', '20-24', '25-29', '30-34', '35+']:
        in_group = with_mother['age_group'] == age_grp
        tested_in_group = with_mother[in_group & with_mother['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo unknown)',
            'HIV-Positive (MTCT occurred)'])]
        
        if len(tested_in_group) > 0:
            mtct_in_group = (tested_in_group['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)').sum()
            rate = mtct_in_group / len(tested_in_group) * 100
            risk_data.append({
                'Age Group': age_grp,
                'MTCT Rate (%)': rate,
                'Cases': mtct_in_group,
                'Tested': len(tested_in_group)
            })
    
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
        
        fig.add_hline(y=5, line_dash="dash", line_color="orange",
                     annotation_text="WHO Target (5%)")
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title='MTCT Rate by Maternal Age',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        "Younger mothers (<25 years) show higher MTCT rates (6-8%) compared to older mothers "
        "(2-6%). This indicates need for youth-focused PMTCT services with enhanced adherence "
        "support and psychosocial counseling.",
        icon="⚠️"
    )
    
    st.markdown("---")
    
    # The Critical Gap
    st.markdown("### 🚨 The Critical Gap: Unknown Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pairs", f"{total_pairs:,}")
    with col2:
        st.metric("Infants Tested", f"{infants_tested:,}", 
                 f"{(infants_tested/total_pairs*100):.1f}%")
    with col3:
        st.metric("Unknown Status", f"{untested:,}",
                 f"{(untested/total_pairs*100):.1f}%",
                 delta_color="inverse")
    
    show_insight_box(
        f"CRITICAL: {untested:,} infants (86.9%) have UNKNOWN HIV status. The true population-level "
        f"MTCT rate cannot be determined. The reported {mtct_rate:.1f}% rate only applies to the "
        "13.1% who were tested. This is the BIGGEST GAP in the entire PMTCT cascade.",
        icon="🚨"
    )

# ============================================================================
# LONGITUDINAL TIMELINE
# ============================================================================
elif page == "⏱️ Longitudinal Timeline":
    st.title("⏱️ Longitudinal Timeline Analysis")
    st.markdown("## Tracking the PMTCT Journey Through Time")
    
    st.markdown("---")
    
    # Calculate time intervals
    with_mother['weeks_pregnant_at_booking'] = ((with_mother['date_of_anc_booking'] - 
                                                   with_mother['date_of_last_known_mensural_period']).dt.days / 7)
    with_mother['days_booking_to_diagnosis'] = (with_mother['mother_date_of_hiv_test'] - 
                                                  with_mother['date_of_anc_booking']).dt.days
    with_mother['days_diagnosis_to_art'] = (with_mother['mother_date_of_art_initiation'] - 
                                              with_mother['date_mother_tested_positive']).dt.days
    with_mother['days_art_to_delivery'] = (with_mother['date_of_delivery'] - 
                                             with_mother['mother_date_of_art_initiation']).dt.days
    with_mother['infant_age_weeks_at_test'] = ((with_mother['infant_hiv_test_date'] - 
                                                  with_mother['infant_date_of_birth']).dt.days / 7)
    
    # Key Timeline Metrics
    st.markdown("### 📊 Key Timeline Intervals (Median Values)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        booking_weeks = with_mother['weeks_pregnant_at_booking'].dropna()
        booking_weeks = booking_weeks[(booking_weeks > 0) & (booking_weeks < 42)]
        median_booking = booking_weeks.median() if len(booking_weeks) > 0 else 0
        
        st.metric(
            "Gestational Age at ANC Booking",
            f"{median_booking:.0f} weeks",
            "Late 2nd trimester" if median_booking > 13 else "1st trimester"
        )
    
    with col2:
        booking_to_dx = with_mother['days_booking_to_diagnosis'].dropna()
        booking_to_dx = booking_to_dx[(booking_to_dx >= 0) & (booking_to_dx <= 280)]
        median_to_dx = booking_to_dx.median() if len(booking_to_dx) > 0 else 0
        
        st.metric(
            "Days: Booking → Diagnosis",
            f"{median_to_dx:.0f} days",
            "Same-day testing!" if median_to_dx == 0 else f"{median_to_dx:.0f} day delay"
        )
    
    with col3:
        dx_to_art = with_mother['days_diagnosis_to_art'].dropna()
        dx_to_art = dx_to_art[(dx_to_art >= 0) & (dx_to_art <= 365)]
        median_to_art = dx_to_art.median() if len(dx_to_art) > 0 else 0
        
        st.metric(
            "Days: Diagnosis → ART",
            f"{median_to_art:.0f} days",
            "Same-day initiation!" if median_to_art == 0 else f"{median_to_art:.0f} day delay"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        art_to_delivery = with_mother['days_art_to_delivery'].dropna()
        art_to_delivery = art_to_delivery[(art_to_delivery >= 0) & (art_to_delivery <= 280)]
        median_art_to_del = art_to_delivery.median() if len(art_to_delivery) > 0 else 0
        
        st.metric(
            "Days on ART Before Delivery",
            f"{median_art_to_del:.0f} days",
            f"~{(median_art_to_del/7):.0f} weeks"
        )
    
    with col2:
        infant_age = with_mother['infant_age_weeks_at_test'].dropna()
        infant_age = infant_age[(infant_age >= 0) & (infant_age <= 104)]
        median_infant_age = infant_age.median() if len(infant_age) > 0 else 0
        
        st.metric(
            "Infant Age at HIV Testing",
            f"{median_infant_age:.1f} weeks",
            "Meets EID guideline" if median_infant_age <= 8 else "Late testing"
        )
    
    with col3:
        # From PDF data
        st.metric(
            "Late Diagnosis Gap",
            "129 days",
            "Average wait after booking",
            help="From PDF: Time mothers waited for test after booking"
        )
    
    st.markdown("---")
    
    # The Longitudinal Diagnosis Gap (from PDF)
    st.markdown("### ⏰ Longitudinal Diagnosis Gap Analysis")
    
    st.markdown("""
    Among mothers who tested positive **AFTER** their first booking (not those who arrived already knowing they were HIV+),  
    there was a significant delay between booking and receiving their diagnosis:
    """)
    
    gap_data = pd.DataFrame({
        'Time After Booking': [
            '8-30 Days (~1 Month)',
            '31-90 Days (1-3 Months)',
            '91-180 Days (3-6 Months)',
            '>180 Days (>6 Months)'
        ],
        'Mothers': [9, 17, 29, 29],
        'Percentage': [10.2, 19.3, 33.0, 33.0],
        'Insight': [
            'Early leakage - first test delayed',
            'Diagnosed in first trimester',
            'MAJOR GAP: 3-6 months in system',
            'CRITICAL RISK: Very late diagnosis'
        ]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(gap_data, use_container_width=True, hide_index=True)
        
        st.error("""
        **🚨 66% of mothers waited >3 months**  
        for diagnosis after booking!
        
        This represents:
        - 58 mothers (33% + 33%)
        - Critical delays in starting ART
        - Higher MTCT risk
        """)
    
    with col2:
        fig = px.bar(
            gap_data,
            x='Time After Booking',
            y='Mothers',
            text='Percentage',
            color='Mothers',
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title='Delay from ANC Booking to HIV Diagnosis',
            yaxis_title='Number of Mothers',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight_box(
        "CRITICAL FINDING: 66% of newly diagnosed mothers experienced delays of 3-6+ months "
        "between ANC booking and HIV diagnosis. This represents a major system failure in "
        "timely testing and severely limits the window for effective PMTCT interventions.",
        icon="🚨"
    )
    
    st.markdown("---")
    
    # Complete Cascade Timeline
    st.markdown("### 🔄 The Complete PMTCT Cascade Timeline")
    
    stages = [
        'ANC Booked',
        'HIV Diagnosed', 
        'ART Initiated',
        'VL Tested',
        'Delivered',
        'Infant Tested'
    ]
    
    counts = [
        len(with_mother['date_of_anc_booking'].dropna()),
        len(with_mother['mother_date_of_hiv_test'].dropna()),
        len(with_mother['mother_date_of_art_initiation'].dropna()),
        len(with_mother['mother_date_of_viral_load'].dropna()),
        len(with_mother['date_of_delivery'].dropna()),
        len(with_mother['infant_hiv_test_date'].dropna())
    ]
    
    percentages = [(c / counts[0] * 100) for c in counts]
    
    fig = go.Figure()
    
    fig.add_trace(go.Funnel(
        y=stages,
        x=counts,
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": px.colors.sequential.RdBu_r},
    ))
    
    fig.update_layout(
        title='PMTCT Cascade: From ANC to Infant Testing',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Loss analysis
    st.markdown("### 📉 Loss Analysis at Each Stage")
    
    losses = []
    for i in range(len(counts)-1):
        loss = counts[i] - counts[i+1]
        loss_pct = (loss / counts[i] * 100)
        losses.append({
            'Stage Transition': f"{stages[i]} → {stages[i+1]}",
            'Lost': loss,
            'Loss Rate (%)': f"{loss_pct:.1f}%"
        })
    
    loss_df = pd.DataFrame(losses)
    st.dataframe(loss_df, use_container_width=True, hide_index=True)
    
    show_insight_box(
        "The BIGGEST loss occurs between Delivery and Infant Testing - losing 87% of the cohort. "
        "This is the critical 'last mile' failure where the program loses track of mother-baby pairs.",
        icon="🎯"
    )

# ============================================================================
# DATA QUALITY ISSUES
# ============================================================================
elif page == "🔍 Data Quality Issues":
    st.title("🔍 Data Quality Assessment")
    st.markdown("## Identifying Data Gaps and Issues")
    
    st.markdown("---")
    
    # Missing Data Analysis
    st.markdown("### 📊 Missing Data by Column")
    
    # Calculate missing data
    missing_no_mother = no_mother.isnull().sum()
    missing_no_mother_pct = (missing_no_mother / len(no_mother)) * 100
    
    missing_with_mother = with_mother.isnull().sum()
    missing_with_mother_pct = (missing_with_mother / len(with_mother)) * 100
    
    tab1, tab2 = st.tabs(["Children Without Maternal Link", "Mother-Baby Pairs"])
    
    with tab1:
        missing_df1 = pd.DataFrame({
            'Column': missing_no_mother.index,
            'Missing Count': missing_no_mother.values,
            'Missing %': missing_no_mother_pct.values
        }).sort_values('Missing %', ascending=False)
        
        missing_df1 = missing_df1[missing_df1['Missing Count'] > 0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_df1.head(10), use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                missing_df1.head(10),
                x='Missing %',
                y='Column',
                orientation='h',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                title='Top 10 Columns with Missing Data',
                xaxis_title='Missing %',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        missing_df2 = pd.DataFrame({
            'Column': missing_with_mother.index,
            'Missing Count': missing_with_mother.values,
            'Missing %': missing_with_mother_pct.values
        }).sort_values('Missing %', ascending=False)
        
        missing_df2 = missing_df2[missing_df2['Missing Count'] > 0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(missing_df2.head(10), use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.bar(
                missing_df2.head(10),
                x='Missing %',
                y='Column',
                orientation='h',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                title='Top 10 Columns with Missing Data',
                xaxis_title='Missing %',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data Inconsistencies
    st.markdown("### ⚠️ Data Inconsistencies Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### HIV Test Result Variations")
        
        result_counts = no_mother['infant_hiv_test_result'].value_counts()
        
        st.dataframe(pd.DataFrame({
            'Value': result_counts.index,
            'Count': result_counts.values
        }), use_container_width=True, hide_index=True)
        
        st.warning("""
        **Inconsistency Found:**
        - "POSITIVE": 1,075 cases
        - "Positive": 136 cases
        
        These should be standardized to uppercase.
        """)
    
    with col2:
        st.markdown("#### Duplicate Records")
        
        duplicates_1 = no_mother.duplicated().sum()
        duplicates_2 = with_mother.duplicated().sum()
        
        dup_data = pd.DataFrame({
            'Dataset': ['Children Without Maternal Link', 'Mother-Baby Pairs'],
            'Duplicate Rows': [duplicates_1, duplicates_2],
            'Percentage': [
                (duplicates_1/len(no_mother)*100),
                (duplicates_2/len(with_mother)*100)
            ]
        })
        
        st.dataframe(dup_data, use_container_width=True, hide_index=True)
        
        if duplicates_1 > 0 or duplicates_2 > 0:
            st.warning(f"""
            **{duplicates_1 + duplicates_2} duplicate records found**
            
            These should be reviewed and removed.
            """)
    
    st.markdown("---")
    
    # Date Anomalies
    st.markdown("### 📅 Date Anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Future Dates")
        
        future_count = 0
        future_cols = []
        
        for col in no_mother.select_dtypes(include=['datetime64']).columns:
            future = no_mother[no_mother[col] > pd.Timestamp.now()]
            if len(future) > 0:
                future_count += len(future)
                future_cols.append(f"{col}: {len(future)} cases")
        
        if future_count > 0:
            st.error(f"**{future_count} future dates found:**")
            for fc in future_cols:
                st.write(f"- {fc}")
        else:
            st.success("✅ No future dates detected")
    
    with col2:
        st.markdown("#### Very Old Dates")
        
        old_count = 0
        old_cols = []
        
        for col in no_mother.select_dtypes(include=['datetime64']).columns:
            very_old = no_mother[no_mother[col] < pd.Timestamp('1990-01-01')]
            if len(very_old) > 0:
                old_count += len(very_old)
                old_cols.append(f"{col}: {len(very_old)} cases")
        
        if old_count > 0:
            st.warning(f"**{old_count} dates before 1990:**")
            for oc in old_cols:
                st.write(f"- {oc}")
        else:
            st.success("✅ No unreasonably old dates")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Data Quality Recommendations")
    
    st.markdown("""
    #### High Priority
    1. **Standardize text fields** - Convert all HIV results to uppercase
    2. **Remove duplicates** - Investigate and remove duplicate records
    3. **Validate dates** - Review and correct date anomalies
    4. **Fill critical gaps** - Prioritize completing infant HIV test results
    
    #### Medium Priority
    5. **Improve VL documentation** - 69% of mothers have no VL result
    6. **Enhance follow-up tracking** - Many missing follow-up status fields
    7. **Implement data validation** - Add constraints to prevent future issues
    
    #### Long-term
    8. **Electronic data capture** - Move to real-time electronic systems
    9. **Data quality dashboards** - Monitor quality metrics continuously
    10. **Training** - Train data clerks on standardized data entry
    """)

# ============================================================================
# KEY RECOMMENDATIONS
# ============================================================================
elif page == "📋 Key Recommendations":
    st.title("📋 Strategic Recommendations")
    st.markdown("## Evidence-Based Action Plan")
    
    st.markdown("---")
    
    # Priority Matrix
    st.markdown("### 🎯 Priority Action Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 CRITICAL PRIORITY (0-3 months)")
        
        with st.expander("**1. Infant Testing Surge Initiative**", expanded=True):
            st.markdown(f"""
            **Problem:** {total_pairs - infants_tested:,} infants (86.9%) untested
            
            **Target:** Test ALL infants within 3 months
            
            **Actions:**
            - Deploy mobile testing teams to 20 high-volume facilities
            - Community health worker tracing of untested mothers
            - SMS/phone reminders for mothers who delivered
            - Integrate testing with immunization clinics
            - Weekend testing clinics for working mothers
            
            **Budget Required:** Moderate
            
            **Expected Impact:** HIGH - Identifies true MTCT rate, links HIV+ children to care
            """)
        
        with st.expander("**2. Mother-Baby Linkage Emergency Response**"):
            st.markdown(f"""
            **Problem:** {len(no_mother):,} children without maternal data
            
            **Target:** Reduce to <100 within 3 months
            
            **Actions:**
            - Investigate each case to determine cause
            - Implement unique mother-baby pair IDs (QR codes)
            - Mandatory linkage verification before discharge
            - Dedicated data quality officer at each facility
            - Real-time alerts for unlinked pairs
            
            **Budget Required:** Low-Moderate
            
            **Expected Impact:** CRITICAL - Prevents system breakdown
            """)
    
    with col2:
        st.markdown("#### 🟡 HIGH PRIORITY (3-12 months)")
        
        with st.expander("**3. Pre-Conception ART Scale-Up**", expanded=True):
            st.markdown("""
            **Evidence:** ZERO transmissions when ART started before pregnancy
            
            **Target:** 80% of HIV+ women of reproductive age on ART
            
            **Actions:**
            - Identify all HIV+ women aged 15-49 in catchment areas
            - Immediate ART initiation regardless of pregnancy status
            - Integrated family planning counseling
            - Retention support for young women
            - Community mobilization campaigns
            
            **Budget Required:** Moderate-High
            
            **Expected Impact:** CRITICAL - Could eliminate MTCT
            """)
        
        with st.expander("**4. Universal Viral Load Testing**"):
            st.markdown(f"""
            **Problem:** {(len(with_mother) - len(with_mother[with_mother['vl_suppressed'] != 'Unknown'])):,} mothers (69%) without VL
            
            **Target:** 95% VL testing coverage
            
            **Actions:**
            - Point-of-care VL machines at 10 facilities
            - Mandatory VL at 28 weeks gestation
            - Results-to-action protocol within 48 hours
            - Enhanced adherence support for non-suppressed
            - Monthly VL testing for non-suppressed mothers
            
            **Budget Required:** High
            
            **Expected Impact:** HIGH - Reduces MTCT risk
            """)
    
    st.markdown("---")
    
    # Youth-Focused Interventions
    st.markdown("### 👧 Youth-Focused PMTCT Services")
    
    st.info("""
    **Evidence:** Mothers <25 years have 2-3x higher MTCT rates (6-8% vs 2-3%)
    
    **Target Population:** ~650 young mothers in program
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Service Delivery**")
        st.markdown("""
        - Adolescent-friendly clinic days
        - Youth peer support groups
        - Flexible appointment times
        - Teen-focused counselors
        """)
    
    with col2:
        st.markdown("**Adherence Support**")
        st.markdown("""
        - SMS/WhatsApp reminders
        - Pill boxes with alarms
        - Home visits if missed
        - Incentive schemes
        """)
    
    with col3:
        st.markdown("**Psychosocial**")
        st.markdown("""
        - Disclosure support
        - Mental health screening
        - Livelihood support
        - Parenting skills
        """)
    
    st.markdown("---")
    
    # Data Systems Strengthening
    st.markdown("### 💻 Data Systems Strengthening")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Technology Solutions")
        st.markdown("""
        **Immediate (0-6 months):**
        - Electronic patient tracking system
        - Unique mother-baby pair IDs
        - Real-time cascade dashboard
        - Automated SMS reminders
        
        **Medium-term (6-18 months):**
        - Mobile data collection apps
        - Cross-facility data sharing
        - Predictive analytics for LTFU
        - Integration with national HMIS
        """)
    
    with col2:
        st.markdown("#### Human Resources")
        st.markdown("""
        **Staffing:**
        - Data quality officers (1 per 5 facilities)
        - M&E specialists at district level
        - IT support technicians
        - Community trackers
        
        **Training:**
        - Standardized data entry protocols
        - Monthly data quality audits
        - Refresher training quarterly
        - Mentorship programs
        """)
    
    st.markdown("---")
    
    # Success Metrics
    st.markdown("### 📊 Success Metrics & Targets")
    
    metrics_data = pd.DataFrame({
        'Indicator': [
            'Infant HIV Testing Coverage',
            'MTCT Rate (among tested)',
            'Mother-Baby Pair Linkage',
            'Maternal VL Testing Coverage',
            'Maternal VL Suppression Rate',
            'Treatment Retention (12 months)',
            'Same-Day ART Initiation'
        ],
        'Current': [
            '13.0%',
            '4.9%',
            f'{((total_pairs/(total_pairs+len(no_mother)))*100):.1f}%',
            '31.2%',
            '61.6%',
            '48.8%',
            '73.2%'
        ],
        '12-Month Target': [
            '95%',
            '<2%',
            '98%',
            '95%',
            '90%',
            '85%',
            '90%'
        ],
        '24-Month Target': [
            '98%',
            '<1%',
            '99%',
            '98%',
            '95%',
            '90%',
            '95%'
        ]
    })
    
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Resource Requirements
    st.markdown("### 💰 Resource Requirements Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Financial")
        st.markdown("""
        **Year 1 Budget:**
        - Testing surge: $50K
        - VL machines: $200K
        - IT systems: $150K
        - Training: $30K
        - Community tracing: $40K
        
        **Total: ~$470K**
        """)
    
    with col2:
        st.markdown("#### Human Resources")
        st.markdown("""
        **New Positions:**
        - 10 Data quality officers
        - 5 M&E specialists
        - 20 Community tracers
        - 2 IT specialists
        - 15 Youth counselors
        
        **Total: 52 FTEs**
        """)
    
    with col3:
        st.markdown("#### Equipment")
        st.markdown("""
        **Procurement:**
        - 10 Point-of-care VL machines
        - 20 Computers/tablets
        - 30 Mobile phones
        - QR code scanners
        - Backup generators
        """)
    
    st.markdown("---")
    
    # Call to Action
    st.markdown("### 🎯 Call to Action")
    
    st.success("""
    ### The Path to MTCT Elimination is Clear:
    
    1. **Test all infants** - Close the 87% testing gap
    2. **Fix the linkage** - Ensure every child has a traceable mother
    3. **Scale pre-conception ART** - Proven to prevent 100% of transmissions
    4. **Support young mothers** - They face the highest risk
    5. **Monitor viral load** - Cannot manage what we don't measure
    
    **With focused effort on these 5 priorities, MTCT elimination (<1%) is achievable within 24 months.**
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>PMTCT Longitudinal Analysis Dashboard</strong></p>
        <p>Study Period: 2021-2025 | Generated: February 2026</p>
        <p>For questions contact: PMTCT Programme Monitoring Team</p>
    </div>
""", unsafe_allow_html=True)