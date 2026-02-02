import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD THE DATA (This defines 'df')
df = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')

# 2. PRE-PROCESS DATES
# We convert strings to 'datetime' objects so we can calculate time gaps
date_cols = ['date_of_anc_booking', 'mother_date_of_hiv_test', 'infant_date_of_birth', 'infant_hiv_test_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# 3. ANALYSIS: ENGAGEMENT AT BOOKING
# This tells us how many mothers knew their status before they even arrived
booking_status = df['mother_hiv_status_at_booking'].fillna('UNKNOWN').value_counts()

# 4. ANALYSIS: SEROCONVERSION / LATE DIAGNOSIS
# We calculate the days between their first booking and their actual HIV test
df['diagnosis_gap_days'] = (df['mother_date_of_hiv_test'] - df['date_of_anc_booking']).dt.days

# Mothers who tested positive more than 7 days after booking are likely seroconverters 
# or were missed during initial screening.
seroconversion_proxy = df[df['diagnosis_gap_days'] > 7]

# 5. PRINT RESULTS FOR YOUR STORY
print("--- 1. Engagement at Booking ---")
print(booking_status)
print(f"\n--- 2. Late Diagnosis (Seroconversion) ---")
print(f"Number of mothers diagnosed late: {len(seroconversion_proxy)}")
print(f"Average days delayed: {seroconversion_proxy['diagnosis_gap_days'].mean():.1f} days")

# 6. VISUALIZATION
plt.figure(figsize=(10,6))
sns.histplot(df[df['diagnosis_gap_days'] > 0]['diagnosis_gap_days'], color='orange', kde=True)
plt.title('Longitudinal Gap: Days from Booking to HIV Diagnosis')
plt.xlabel('Days Elapsed')
plt.ylabel('Number of Mothers')
plt.savefig('my_plot.png')