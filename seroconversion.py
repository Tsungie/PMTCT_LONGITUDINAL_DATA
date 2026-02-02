import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prep
df = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
df['date_of_anc_booking'] = pd.to_datetime(df['date_of_anc_booking'], dayfirst=True, errors='coerce')
df['mother_date_of_hiv_test'] = pd.to_datetime(df['mother_date_of_hiv_test'], dayfirst=True, errors='coerce')

# 2. Calculate Gap and filter the 88 "Late" Positive mothers
df['diag_gap'] = (df['mother_date_of_hiv_test'] - df['date_of_anc_booking']).dt.days
late_df = df[(df['diag_gap'] > 7) & (df['mother_hiv_test_result'] == 'POSITIVE')].copy()

# 3. Categorize for the Table
def get_bucket(days):
    if 8 <= days <= 30: return '8 - 30 Days (~1 Month)'
    if 31 <= days <= 90: return '31 - 90 Days (1-3 Months)'
    if 91 <= days <= 180: return '91 - 180 Days (3-6 Months)'
    return 'Over 180 Days (>6 Months)'

late_df['Gap Range'] = late_df['diag_gap'].apply(get_bucket)
summary = late_df['Gap Range'].value_counts().reset_index()
summary.columns = ['Time Elapsed', 'Mothers']
summary['%'] = ((summary['Mothers'] / 88) * 100).round(1)

# 4. Create Graph with Table below it
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(late_df['diag_gap'], bins=15, color='salmon', kde=True, ax=ax)
ax.axvline(late_df['diag_gap'].mean(), color='red', linestyle='--', label=f'Avg: {late_df["diag_gap"].mean():.1f} days')

# Add Table
plt.table(cellText=summary.values, colLabels=summary.columns, 
          loc='bottom', bbox=[0, -0.45, 1, 0.3])

plt.subplots_adjust(bottom=0.4)
plt.title('Breakdown of the 178-Day Longitudinal Gap')
plt.savefig('outputs/longitudinal_gap_with_table.png')
plt.show()