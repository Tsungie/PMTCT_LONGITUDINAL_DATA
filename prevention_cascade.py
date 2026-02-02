import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. SETUP
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Load Data
df = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')

# --- DATA CLEANING ---
date_cols = ['date_of_anc_booking', 'mother_date_of_hiv_test', 'date_mother_tested_positive',
             'mother_date_of_art_initiation', 'infant_hiv_test_date', 'infant_date_of_birth',
             'date_of_delivery', 'date_of_last_known_mensural_period']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Clean Text Columns
text_cols = ['mother_hiv_status_at_booking', 'mother_hiv_test_result', 
             'mother_viral_load_result', 'infant_hiv_test_result',
             'mother_started_art_before_current_pregnancy',
             'mother_started_art_during_pregnancy_gt4weeks_before_delivery',
             'mother_started_art_during_pregnancy_lss4weeks_before_delivery']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# --- 1. ART TIMING (Pie Chart) ---
n_total = len(df)
n_before = df['mother_started_art_before_current_pregnancy'].eq('YES').sum()
n_during_gt4 = df['mother_started_art_during_pregnancy_gt4weeks_before_delivery'].eq('YES').sum()
n_during_lt4 = df['mother_started_art_during_pregnancy_lss4weeks_before_delivery'].eq('YES').sum()
n_unknown = n_total - (n_before + n_during_gt4 + n_during_lt4)

labels = ['Before Pregnancy', 'During (>4wks before del)', 'During (<4wks before del)', 'Unknown Timing']
counts = [n_before, n_during_gt4, n_during_lt4, n_unknown]

# Helper for Pie Chart Labels
def make_autopct(values):
    def my_autopct(pct):
        val = int(round(pct*sum(values)/100.0))
        return '{v:d}\n({p:.1f}%)'.format(p=pct, v=val)
    return my_autopct

plt.figure(figsize=(9, 7))
colors = sns.color_palette('pastel')[0:4]
plt.pie(counts, labels=labels, autopct=make_autopct(counts), colors=colors, startangle=140)
plt.title('ART Initiation Timing', fontsize=14)
plt.savefig('outputs/art_timing_pie.png')
plt.close()

# --- 2. SAME DAY INITIATION ---
df['art_delay'] = (df['mother_date_of_art_initiation'] - df['mother_date_of_hiv_test']).dt.days
valid_dates = df[(df['art_delay'].notnull()) & (df['art_delay'] >= 0)]
same_day_rate = (len(valid_dates[valid_dates['art_delay'] == 0]) / len(valid_dates)) * 100

plt.figure(figsize=(6, 5))
sns.barplot(x=['Same Day Initiation'], y=[same_day_rate], color='teal')
plt.ylim(0, 100)
plt.ylabel('Percentage (%)')
plt.title(f'Same Day Initiation Success: {same_day_rate:.1f}%')
plt.text(0, same_day_rate/2, f'{same_day_rate:.1f}%', ha='center', color='white', weight='bold', fontsize=14)
plt.savefig('outputs/same_day_initiation.png')
plt.close()

# --- 3. VIRAL SUPPRESSION (Among 586 mothers) ---
# Filter non-empty results
vl_df = df[~df['mother_viral_load_result'].isin(['NAN', 'NONE', 'UNKNOWN', 'nan'])].copy()
vl_df = vl_df[vl_df['mother_viral_load_result'].str.len() > 0]

def categorize_vl(x):
    s = str(x).strip().upper()
    if s == 'TND': return 'Suppressed'
    try:
        return 'Suppressed' if float(s) < 1000 else 'Not Suppressed'
    except:
        return 'Invalid/Other'

vl_counts = vl_df['mother_viral_load_result'].apply(categorize_vl).value_counts()

plt.figure(figsize=(7, 7))
plt.pie(vl_counts, labels=vl_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999', 'silver'])
plt.title(f'Viral Suppression (N={vl_counts.sum()})')
plt.savefig('outputs/viral_suppression.png')
plt.close()

# --- 4. INFANT OUTCOMES ---
doc_counts = df[~df['infant_hiv_test_result'].isin(['NAN', 'NONE', 'UNKNOWN'])]['infant_hiv_test_result'].value_counts()
n_undoc = len(df) - doc_counts.sum()

plt.figure(figsize=(8, 8))
# Pie of Documented Results
plt.pie(doc_counts, labels=doc_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
plt.title(f'Infant Outcomes (Documented Results Only)\nNote: {n_undoc} infants have NO documented result')
plt.savefig('outputs/infant_outcomes_detailed.png')
plt.close()

# --- 5. GESTATIONAL AGE ---
df['ga_weeks'] = (df['date_of_delivery'] - df['date_of_last_known_mensural_period']).dt.days / 7
valid_ga = df[(df['ga_weeks'] > 20) & (df['ga_weeks'] < 46)]

plt.figure(figsize=(8, 5))
sns.histplot(valid_ga['ga_weeks'], bins=15, kde=True, color='purple')
plt.axvline(valid_ga['ga_weeks'].mean(), color='red', linestyle='--', label=f"Mean: {valid_ga['ga_weeks'].mean():.1f}")
plt.axvline(valid_ga['ga_weeks'].median(), color='blue', linestyle='-', label=f"Median: {valid_ga['ga_weeks'].median():.1f}")
plt.title('Gestational Age at Delivery (Weeks)')
plt.legend()
plt.savefig('outputs/gestational_age.png')
plt.close()

print("All charts generated and saved to 'outputs/' folder.")