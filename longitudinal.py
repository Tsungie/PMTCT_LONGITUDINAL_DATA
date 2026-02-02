"""
PMTCT LONGITUDINAL ANALYSIS WITH SEROCONVERSION TRACKING
=========================================================
This script analyzes the longitudinal trajectories of mother-baby pairs
and tracks infant seroconversion outcomes over time.

Key Longitudinal Analyses:
1. Time-based progression through PMTCT cascade
2. Infant seroconversion patterns (negative → positive or maintaining negative)
3. Survival/retention analysis over time
4. Time-to-event analysis (ART initiation, viral suppression, etc.)
5. Follow-up milestones achievement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load and prepare datasets"""
    no_mother = pd.read_csv('uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv')
    with_mother = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
    
    # Convert dates
    date_cols_no_mother = ['infant_date_of_birth', 'infant_hiv_test_date', 
                           'infant_date_of_art_initiation', 'infant_date_of_art_enrolment']
    
    for col in date_cols_no_mother:
        if col in no_mother.columns:
            no_mother[col] = pd.to_datetime(no_mother[col], errors='coerce', dayfirst=True)
    
    date_cols_with_mother = ['date_of_last_known_mensural_period', 'date_of_anc_booking',
                             'mother_date_of_hiv_test', 'date_mother_tested_positive',
                             'mother_date_of_art_initiation', 'mother_date_of_viral_load',
                             'date_of_delivery', 'infant_date_of_birth', 'infant_hiv_test_date',
                             'infant_date_of_art_enrolment']
    
    for col in date_cols_with_mother:
        if col in with_mother.columns:
            with_mother[col] = pd.to_datetime(with_mother[col], errors='coerce', dayfirst=True)
    
    return no_mother, with_mother

# ============================================================================
# LONGITUDINAL TRAJECTORY ANALYSIS
# ============================================================================

def analyze_pmtct_timeline(df):
    """
    Analyze the complete PMTCT timeline from pregnancy to infant outcome
    
    Timeline stages:
    T0: Last menstrual period (LMP) - pregnancy begins
    T1: ANC booking - entry into care
    T2: HIV testing/diagnosis
    T3: ART initiation
    T4: Viral load testing
    T5: Delivery
    T6: Infant HIV testing (6 weeks, 6 months, etc.)
    """
    
    print("\n" + "="*70)
    print("LONGITUDINAL PMTCT TIMELINE ANALYSIS")
    print("="*70)
    
    # Calculate time intervals between key events
    
    # 1. Time from LMP to ANC booking (gestational age at first visit)
    df['weeks_pregnant_at_booking'] = ((df['date_of_anc_booking'] - 
                                         df['date_of_last_known_mensural_period']).dt.days / 7)
    
    # 2. Time from ANC booking to HIV diagnosis
    df['days_booking_to_diagnosis'] = (df['mother_date_of_hiv_test'] - 
                                        df['date_of_anc_booking']).dt.days
    
    # 3. Time from diagnosis to ART initiation
    df['days_diagnosis_to_art'] = (df['mother_date_of_art_initiation'] - 
                                    df['date_mother_tested_positive']).dt.days
    
    # 4. Time from ART initiation to viral load testing
    df['days_art_to_vl'] = (df['mother_date_of_viral_load'] - 
                             df['mother_date_of_art_initiation']).dt.days
    
    # 5. Time from ART initiation to delivery
    df['days_art_to_delivery'] = (df['date_of_delivery'] - 
                                   df['mother_date_of_art_initiation']).dt.days
    
    # 6. Time from delivery to infant testing
    df['days_delivery_to_infant_test'] = (df['infant_hiv_test_date'] - 
                                           df['date_of_delivery']).dt.days
    
    # 7. Infant age at testing (in weeks)
    df['infant_age_weeks_at_test'] = ((df['infant_hiv_test_date'] - 
                                       df['infant_date_of_birth']).dt.days / 7)
    
    print("\n📊 LONGITUDINAL TIME INTERVALS")
    print("="*70)
    
    # Analyze each interval
    intervals = {
        'Gestational age at ANC booking': ('weeks_pregnant_at_booking', 'weeks', 0, 42),
        'Days from booking to HIV diagnosis': ('days_booking_to_diagnosis', 'days', 0, 280),
        'Days from diagnosis to ART': ('days_diagnosis_to_art', 'days', 0, 365),
        'Days from ART to viral load test': ('days_art_to_vl', 'days', 0, 365),
        'Days on ART before delivery': ('days_art_to_delivery', 'days', 0, 280),
        'Days from delivery to infant test': ('days_delivery_to_infant_test', 'days', 0, 730),
        'Infant age at testing': ('infant_age_weeks_at_test', 'weeks', 0, 104)
    }
    
    results = {}
    for name, (col, unit, min_val, max_val) in intervals.items():
        valid_data = df[col].dropna()
        valid_data = valid_data[(valid_data >= min_val) & (valid_data <= max_val)]
        
        if len(valid_data) > 0:
            results[name] = {
                'n': len(valid_data),
                'mean': valid_data.mean(),
                'median': valid_data.median(),
                'std': valid_data.std(),
                'min': valid_data.min(),
                'max': valid_data.max()
            }
            
            print(f"\n{name}:")
            print(f"  N: {len(valid_data):,} | Mean: {valid_data.mean():.1f} {unit} | "
                  f"Median: {valid_data.median():.1f} {unit}")
    
    return df, results

def visualize_pmtct_timeline(df):
    """Create comprehensive timeline visualizations"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('PMTCT Longitudinal Timeline Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Gestational age at booking
    ax = axes[0, 0]
    valid = df['weeks_pregnant_at_booking'].dropna()
    valid = valid[(valid > 0) & (valid < 42)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(12, color='green', linestyle='--', linewidth=2, label='1st trimester')
        ax.axvline(28, color='orange', linestyle='--', linewidth=2, label='3rd trimester')
        ax.set_xlabel('Gestational Age (weeks)', fontsize=10)
        ax.set_ylabel('Number of Mothers', fontsize=10)
        ax.set_title('When Mothers Enter ANC', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Time from booking to diagnosis
    ax = axes[0, 1]
    valid = df['days_booking_to_diagnosis'].dropna()
    valid = valid[(valid >= 0) & (valid <= 180)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Same day (ideal)')
        ax.set_xlabel('Days from Booking to Diagnosis', fontsize=10)
        ax.set_ylabel('Number of Mothers', fontsize=10)
        ax.set_title('Testing Delay After ANC Entry', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Time from diagnosis to ART
    ax = axes[0, 2]
    valid = df['days_diagnosis_to_art'].dropna()
    valid = valid[(valid >= 0) & (valid <= 180)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#27ae60', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Same day (ideal)')
        ax.set_xlabel('Days from Diagnosis to ART', fontsize=10)
        ax.set_ylabel('Number of Mothers', fontsize=10)
        ax.set_title('ART Initiation Delay', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Days on ART before delivery
    ax = axes[1, 0]
    valid = df['days_art_to_delivery'].dropna()
    valid = valid[(valid >= 0) & (valid <= 280)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(28, color='red', linestyle='--', linewidth=2, label='4 weeks (minimum)')
        ax.axvline(84, color='orange', linestyle='--', linewidth=2, label='12 weeks (optimal)')
        ax.set_xlabel('Days on ART Before Delivery', fontsize=10)
        ax.set_ylabel('Number of Mothers', fontsize=10)
        ax.set_title('Duration on ART Pre-Delivery', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Time from ART to VL testing
    ax = axes[1, 1]
    valid = df['days_art_to_vl'].dropna()
    valid = valid[(valid >= 0) & (valid <= 365)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
        ax.axvline(168, color='green', linestyle='--', linewidth=2, label='6 months (WHO rec.)')
        ax.set_xlabel('Days from ART to VL Test', fontsize=10)
        ax.set_ylabel('Number of Mothers', fontsize=10)
        ax.set_title('VL Monitoring Timeline', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Infant age at testing
    ax = axes[1, 2]
    valid = df['infant_age_weeks_at_test'].dropna()
    valid = valid[(valid >= 0) & (valid <= 104)]
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#e67e22', alpha=0.7, edgecolor='black')
        ax.axvline(6, color='green', linestyle='--', linewidth=2, label='6 weeks (EID)')
        ax.axvline(26, color='orange', linestyle='--', linewidth=2, label='6 months')
        ax.axvline(52, color='red', linestyle='--', linewidth=2, label='12 months')
        ax.set_xlabel('Infant Age at Testing (weeks)', fontsize=10)
        ax.set_ylabel('Number of Infants', fontsize=10)
        ax.set_title('When Infants Are Tested', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Timeline flow diagram (Sankey-style bar progression)
    ax = axes[2, 0]
    stages = ['ANC\nBooked', 'HIV+\nDiagnosed', 'ART\nInitiated', 'VL\nTested', 
              'Delivered', 'Infant\nTested']
    counts = [
        len(df['date_of_anc_booking'].dropna()),
        len(df['mother_date_of_hiv_test'].dropna()),
        len(df['mother_date_of_art_initiation'].dropna()),
        len(df['mother_date_of_viral_load'].dropna()),
        len(df['date_of_delivery'].dropna()),
        len(df['infant_hiv_test_date'].dropna())
    ]
    
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stages)))
    bars = ax.barh(range(len(stages)), counts, color=colors_gradient, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel('Number of Mother-Baby Pairs', fontsize=10)
    ax.set_title('PMTCT Cascade: Losses at Each Stage', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if i == 0:
            retention = 100.0
        else:
            retention = (count / counts[0]) * 100
        ax.text(count + 20, i, f'{count}\n({retention:.0f}%)', 
                va='center', fontweight='bold', fontsize=9)
    
    # 8. Cumulative time to final outcome
    ax = axes[2, 1]
    # Calculate total time from ANC to infant testing
    df['total_cascade_days'] = (df['infant_hiv_test_date'] - df['date_of_anc_booking']).dt.days
    valid = df['total_cascade_days'].dropna()
    valid = valid[(valid >= 0) & (valid <= 730)]
    
    if len(valid) > 0:
        ax.hist(valid, bins=30, color='#34495e', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Days from ANC Booking to Infant Testing', fontsize=10)
        ax.set_ylabel('Number of Pairs', fontsize=10)
        ax.set_title('Complete Cascade Duration', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 9. Retention over time (by test year)
    ax = axes[2, 2]
    df['test_year'] = df['infant_hiv_test_date'].dt.year
    yearly = df.groupby('test_year').size()
    if len(yearly) > 0:
        ax.plot(yearly.index, yearly.values, marker='o', markersize=8, 
                linewidth=2.5, color='#2c3e50')
        ax.fill_between(yearly.index, yearly.values, alpha=0.3, color='#2c3e50')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Number of Infants Tested', fontsize=10)
        ax.set_title('Testing Volume Over Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# SEROCONVERSION ANALYSIS
# ============================================================================

def analyze_seroconversion(df):
    """
    Analyze infant seroconversion outcomes
    
    Seroconversion scenarios:
    1. Exposed, uninfected: Negative at all time points (PMTCT SUCCESS)
    2. Early infection: Positive at first test (in utero or peripartum transmission)
    3. Late infection: Negative → Positive (postnatal transmission, likely breastfeeding)
    4. Indeterminate: Mixed or inconclusive results
    """
    
    print("\n" + "="*70)
    print("INFANT SEROCONVERSION ANALYSIS")
    print("="*70)
    
    # Standardize test results
    df['infant_result_clean'] = df['infant_hiv_test_result'].str.upper().str.strip()
    df['status_6mo_clean'] = df['child_hiv_status_at_6_months'].str.upper().str.strip()
    df['final_status_clean'] = df['child_hiv_status'].str.upper().str.strip()
    
    # Classify outcomes
    def classify_outcome(row):
        result = row['infant_result_clean']
        status_6mo = row['status_6mo_clean']
        final_status = row['final_status_clean']
        
        # All negative = PMTCT success
        if pd.notna(result) and 'NEGATIVE' in str(result):
            if pd.notna(status_6mo) and 'NEGATIVE' in str(status_6mo):
                return 'HIV-Negative (PMTCT Success)'
            elif pd.isna(status_6mo):
                return 'HIV-Negative (6mo status unknown)'
        
        # Any positive = transmission occurred
        if (pd.notna(result) and 'POSITIVE' in str(result)) or \
           (pd.notna(status_6mo) and 'POSITIVE' in str(status_6mo)) or \
           (pd.notna(final_status) and 'POSITIVE' in str(final_status)):
            return 'HIV-Positive (MTCT occurred)'
        
        # Inconclusive
        if (pd.notna(result) and 'INCONCLUSIVE' in str(result)):
            return 'Inconclusive'
        
        # No test results
        if pd.isna(result) and pd.isna(status_6mo) and pd.isna(final_status):
            return 'Not tested'
        
        return 'Unknown'
    
    df['seroconversion_outcome'] = df.apply(classify_outcome, axis=1)
    
    # Analyze outcomes
    outcomes = df['seroconversion_outcome'].value_counts()
    
    print("\n📊 SEROCONVERSION OUTCOMES:")
    print("="*70)
    for outcome in outcomes.index:
        count = outcomes[outcome]
        pct = (count / len(df)) * 100
        
        if 'Success' in outcome:
            icon = '✅'
        elif 'MTCT occurred' in outcome:
            icon = '🚨'
        elif 'Not tested' in outcome:
            icon = '⚠️'
        else:
            icon = '❓'
        
        print(f"{icon} {outcome}: {count:,} ({pct:.1f}%)")
    
    # Calculate MTCT rate (among tested)
    tested = df[df['seroconversion_outcome'].isin(['HIV-Negative (PMTCT Success)', 
                                                    'HIV-Negative (6mo status unknown)',
                                                    'HIV-Positive (MTCT occurred)'])]
    
    if len(tested) > 0:
        positive = df[df['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)']
        mtct_rate = len(positive) / len(tested) * 100
        
        print(f"\n{'='*70}")
        print(f"📊 MOTHER-TO-CHILD TRANSMISSION RATE: {mtct_rate:.2f}%")
        print(f"   ({len(positive):,} infections among {len(tested):,} tested infants)")
        print(f"{'='*70}")
        
        # Compare to WHO elimination target
        who_target = 5.0
        if mtct_rate <= 2.0:
            status = "✅ EXCELLENT - Elimination achieved!"
            color = 'success'
        elif mtct_rate <= who_target:
            status = "✅ GOOD - Meets WHO target (<5%)"
            color = 'warning'
        else:
            status = "🚨 NEEDS IMPROVEMENT - Above WHO target"
            color = 'danger'
        
        print(f"\n{status}")
        print(f"WHO MTCT Elimination Target: <{who_target}%")
        print(f"Current MTCT Rate: {mtct_rate:.2f}%")
    
    # Timing of transmission analysis
    print(f"\n{'='*70}")
    print("TIMING OF TRANSMISSION (When Was Infant Infected?)")
    print(f"{'='*70}")
    
    positive_infants = df[df['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'].copy()
    
    if len(positive_infants) > 0:
        # Categorize by age at testing
        positive_infants['transmission_timing'] = pd.cut(
            positive_infants['infant_age_weeks_at_test'],
            bins=[-np.inf, 6, 26, 52, np.inf],
            labels=['In utero/Peripartum (<6 weeks)', 
                   'Early postnatal (6w-6mo)',
                   'Late postnatal (6-12mo)',
                   'Very late (>12mo)']
        )
        
        timing = positive_infants['transmission_timing'].value_counts()
        print("\nWhen transmission likely occurred:")
        for category in timing.index:
            count = timing[category]
            pct = (count / len(positive_infants)) * 100
            print(f"  • {category}: {count} ({pct:.1f}%)")
    
    return df, outcomes

def analyze_risk_factors_for_mtct(df):
    """Analyze risk factors associated with MTCT"""
    
    print("\n" + "="*70)
    print("RISK FACTORS FOR MOTHER-TO-CHILD TRANSMISSION")
    print("="*70)
    
    # Classify MTCT outcome
    mtct_occurred = df['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'
    
    # 1. Maternal age
    print("\n1. MATERNAL AGE:")
    print("-"*50)
    df['age_group'] = pd.cut(df['mother_age_at_booking'], 
                              bins=[0, 20, 25, 30, 35, 100],
                              labels=['<20', '20-24', '25-29', '30-34', '35+'])
    
    for age_grp in df['age_group'].dropna().unique():
        in_group = df['age_group'] == age_grp
        tested_in_group = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        
        if len(tested_in_group) > 0:
            mtct_in_group = tested_in_group['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'
            rate = mtct_in_group.sum() / len(tested_in_group) * 100
            print(f"   {age_grp} years: {mtct_in_group.sum()}/{len(tested_in_group)} ({rate:.1f}%)")
    
    # 2. ART timing
    print("\n2. TIMING OF ART INITIATION:")
    print("-"*50)
    
    art_groups = {
        'Before pregnancy': 'mother_started_art_before_current_pregnancy',
        'During pregnancy (>4wks before delivery)': 'mother_started_art_during_pregnancy_gt4weeks_before_delivery',
        'During pregnancy (<4wks before delivery)': 'mother_started_art_during_pregnancy_lss4weeks_before_delivery'
    }
    
    for label, col in art_groups.items():
        in_group = df[col] == 'Yes'
        tested_in_group = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        
        if len(tested_in_group) > 0:
            mtct_in_group = tested_in_group['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'
            rate = mtct_in_group.sum() / len(tested_in_group) * 100
            print(f"   {label}: {mtct_in_group.sum()}/{len(tested_in_group)} ({rate:.1f}%)")
    
    # 3. Viral load suppression
    print("\n3. MATERNAL VIRAL LOAD STATUS:")
    print("-"*50)
    
    # Classify VL
    def classify_vl(vl_value):
        if pd.isna(vl_value):
            return 'Unknown'
        vl_str = str(vl_value).upper().strip()
        if vl_str in ['TND', '<30', '<20', '<50', '<40']:
            return 'Suppressed'
        try:
            vl_num = float(vl_str)
            return 'Suppressed' if vl_num < 1000 else 'Not Suppressed'
        except:
            return 'Unknown'
    
    df['vl_category'] = df['mother_viral_load_result'].apply(classify_vl)
    
    for vl_status in ['Suppressed', 'Not Suppressed', 'Unknown']:
        in_group = df['vl_category'] == vl_status
        tested_in_group = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        
        if len(tested_in_group) > 0:
            mtct_in_group = tested_in_group['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'
            rate = mtct_in_group.sum() / len(tested_in_group) * 100
            print(f"   {vl_status}: {mtct_in_group.sum()}/{len(tested_in_group)} ({rate:.1f}%)")
    
    return df

def visualize_seroconversion(df):
    """Create seroconversion visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Infant Seroconversion Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall outcomes pie chart
    ax = axes[0, 0]
    outcomes = df['seroconversion_outcome'].value_counts()
    colors_outcome = {'HIV-Negative (PMTCT Success)': '#27ae60',
                     'HIV-Negative (6mo status unknown)': '#2ecc71',
                     'HIV-Positive (MTCT occurred)': '#e74c3c',
                     'Inconclusive': '#f39c12',
                     'Not tested': '#95a5a6',
                     'Unknown': '#7f8c8d'}
    outcome_colors = [colors_outcome.get(x, '#95a5a6') for x in outcomes.index]
    
    ax.pie(outcomes.values, labels=outcomes.index, autopct='%1.1f%%',
           colors=outcome_colors, startangle=90)
    ax.set_title('Infant Seroconversion Outcomes', fontsize=12, fontweight='bold')
    
    # 2. MTCT rate by maternal age
    ax = axes[0, 1]
    age_mtct = []
    age_labels = []
    for age_grp in ['<20', '20-24', '25-29', '30-34', '35+']:
        in_group = df['age_group'] == age_grp
        tested = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        if len(tested) > 0:
            mtct_rate = (tested['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)').sum() / len(tested) * 100
            age_mtct.append(mtct_rate)
            age_labels.append(f"{age_grp}\n(n={len(tested)})")
    
    if age_mtct:
        bars = ax.bar(range(len(age_mtct)), age_mtct, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.axhline(5, color='orange', linestyle='--', linewidth=2, label='WHO target (5%)')
        ax.axhline(2, color='green', linestyle='--', linewidth=2, label='Elimination (2%)')
        ax.set_xticks(range(len(age_labels)))
        ax.set_xticklabels(age_labels, fontsize=9)
        ax.set_ylabel('MTCT Rate (%)', fontsize=10)
        ax.set_title('MTCT Rate by Maternal Age', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, val in enumerate(age_mtct):
            ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. MTCT rate by ART timing
    ax = axes[0, 2]
    timing_mtct = []
    timing_labels = []
    
    art_cats = [
        ('Before\nPregnancy', 'mother_started_art_before_current_pregnancy'),
        ('During\n(>4wks)', 'mother_started_art_during_pregnancy_gt4weeks_before_delivery'),
        ('During\n(<4wks)', 'mother_started_art_during_pregnancy_lss4weeks_before_delivery')
    ]
    
    for label, col in art_cats:
        in_group = df[col] == 'Yes'
        tested = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        if len(tested) > 0:
            mtct_rate = (tested['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)').sum() / len(tested) * 100
            timing_mtct.append(mtct_rate)
            timing_labels.append(f"{label}\n(n={len(tested)})")
    
    if timing_mtct:
        colors_timing = ['#27ae60', '#f39c12', '#e74c3c']
        bars = ax.bar(range(len(timing_mtct)), timing_mtct, 
                     color=colors_timing[:len(timing_mtct)], alpha=0.7, edgecolor='black')
        ax.axhline(5, color='orange', linestyle='--', linewidth=2, label='WHO target')
        ax.set_xticks(range(len(timing_labels)))
        ax.set_xticklabels(timing_labels, fontsize=9)
        ax.set_ylabel('MTCT Rate (%)', fontsize=10)
        ax.set_title('MTCT Rate by ART Initiation Timing', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, val in enumerate(timing_mtct):
            ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. MTCT rate by viral load status
    ax = axes[1, 0]
    vl_mtct = []
    vl_labels = []
    
    for vl_status in ['Suppressed', 'Not Suppressed', 'Unknown']:
        in_group = df['vl_category'] == vl_status
        tested = df[in_group & df['seroconversion_outcome'].isin([
            'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
            'HIV-Positive (MTCT occurred)'])]
        if len(tested) > 0:
            mtct_rate = (tested['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)').sum() / len(tested) * 100
            vl_mtct.append(mtct_rate)
            vl_labels.append(f"{vl_status}\n(n={len(tested)})")
    
    if vl_mtct:
        colors_vl = ['#27ae60', '#e74c3c', '#95a5a6']
        bars = ax.bar(range(len(vl_mtct)), vl_mtct, 
                     color=colors_vl[:len(vl_mtct)], alpha=0.7, edgecolor='black')
        ax.axhline(5, color='orange', linestyle='--', linewidth=2, label='WHO target')
        ax.set_xticks(range(len(vl_labels)))
        ax.set_xticklabels(vl_labels, fontsize=9)
        ax.set_ylabel('MTCT Rate (%)', fontsize=10)
        ax.set_title('MTCT Rate by Maternal Viral Load', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, val in enumerate(vl_mtct):
            ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Timing of transmission
    ax = axes[1, 1]
    positive_infants = df[df['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'].copy()
    
    if len(positive_infants) > 0:
        positive_infants['transmission_timing'] = pd.cut(
            positive_infants['infant_age_weeks_at_test'],
            bins=[-np.inf, 6, 26, 52, np.inf],
            labels=['In utero/\nPeripartum\n(<6w)', 
                   'Early\nPostnatal\n(6w-6mo)',
                   'Late\nPostnatal\n(6-12mo)',
                   'Very Late\n(>12mo)']
        )
        
        timing_dist = positive_infants['transmission_timing'].value_counts()
        colors_timing = ['#c0392b', '#e74c3c', '#e67e22', '#f39c12']
        
        ax.bar(range(len(timing_dist)), timing_dist.values, 
               color=colors_timing[:len(timing_dist)], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(timing_dist)))
        ax.set_xticklabels(timing_dist.index, fontsize=8)
        ax.set_ylabel('Number of Infections', fontsize=10)
        ax.set_title('Timing of MTCT', fontsize=12, fontweight='bold')
        
        for i, val in enumerate(timing_dist.values):
            pct = (val / len(positive_infants)) * 100
            ax.text(i, val + 0.2, f'{val}\n({pct:.0f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 6. Testing coverage over time
    ax = axes[1, 2]
    df['test_year'] = df['infant_hiv_test_date'].dt.year
    yearly = df.groupby('test_year')['seroconversion_outcome'].value_counts().unstack(fill_value=0)
    
    if not yearly.empty and 'HIV-Positive (MTCT occurred)' in yearly.columns:
        yearly_mtct_rate = (yearly['HIV-Positive (MTCT occurred)'] / yearly.sum(axis=1) * 100)
        
        ax.plot(yearly_mtct_rate.index, yearly_mtct_rate.values, 
               marker='o', markersize=8, linewidth=2.5, color='#e74c3c')
        ax.axhline(5, color='orange', linestyle='--', linewidth=2, label='WHO target (5%)')
        ax.axhline(2, color='green', linestyle='--', linewidth=2, label='Elimination (2%)')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('MTCT Rate (%)', fontsize=10)
        ax.set_title('MTCT Rate Trend Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print("PMTCT LONGITUDINAL AND SEROCONVERSION ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    no_mother, with_mother = load_data()
    
    # Longitudinal timeline analysis
    print("\n\n" + "="*70)
    print("PART 1: LONGITUDINAL TIMELINE ANALYSIS")
    print("="*70)
    with_mother, timeline_results = analyze_pmtct_timeline(with_mother)
    
    print("\nCreating timeline visualizations...")
    fig1 = visualize_pmtct_timeline(with_mother)
    fig1.savefig('/mnt/user-data/outputs/pmtct_longitudinal_timeline.png', 
                 dpi=300, bbox_inches='tight')
    print("✅ Saved: pmtct_longitudinal_timeline.png")
    
    # Seroconversion analysis
    print("\n\n" + "="*70)
    print("PART 2: SEROCONVERSION ANALYSIS")
    print("="*70)
    with_mother, sero_outcomes = analyze_seroconversion(with_mother)
    
    # Risk factors
    with_mother = analyze_risk_factors_for_mtct(with_mother)
    
    print("\nCreating seroconversion visualizations...")
    fig2 = visualize_seroconversion(with_mother)
    fig2.savefig('/mnt/user-data/outputs/pmtct_seroconversion_analysis.png', 
                 dpi=300, bbox_inches='tight')
    print("✅ Saved: pmtct_seroconversion_analysis.png")
    
    # Summary
    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  • pmtct_longitudinal_timeline.png")
    print("  • pmtct_seroconversion_analysis.png")
    
    print("\n\n" + "="*70)
    print("KEY LONGITUDINAL INSIGHTS")
    print("="*70)
    
    # Print key metrics
    if 'weeks_pregnant_at_booking' in timeline_results:
        weeks_at_booking = timeline_results['Gestational age at ANC booking']
        print(f"\n✅ Mothers book ANC at median {weeks_at_booking['median']:.0f} weeks gestation")
    
    if 'Days from diagnosis to ART' in timeline_results:
        days_to_art = timeline_results['Days from diagnosis to ART']
        print(f"✅ Median {days_to_art['median']:.0f} days from diagnosis to ART (same-day!)")
    
    if 'Infant age at testing' in timeline_results:
        infant_age = timeline_results['Infant age at testing']
        print(f"⚠️  Infants tested at median {infant_age['median']:.0f} weeks of age")
    
    # Seroconversion summary
    tested_total = len(with_mother[with_mother['seroconversion_outcome'].isin([
        'HIV-Negative (PMTCT Success)', 'HIV-Negative (6mo status unknown)',
        'HIV-Positive (MTCT occurred)'])])
    
    mtct_cases = len(with_mother[with_mother['seroconversion_outcome'] == 'HIV-Positive (MTCT occurred)'])
    
    if tested_total > 0:
        mtct_rate = (mtct_cases / tested_total) * 100
        print(f"\n📊 FINAL MTCT RATE: {mtct_rate:.2f}% ({mtct_cases}/{tested_total})")
        
        if mtct_rate <= 2:
            print("   ✅✅ MTCT ELIMINATION ACHIEVED!")
        elif mtct_rate <= 5:
            print("   ✅ Meets WHO PMTCT target (<5%)")
        else:
            print("   🚨 Above WHO target - needs improvement")

if __name__ == "__main__":
    main()