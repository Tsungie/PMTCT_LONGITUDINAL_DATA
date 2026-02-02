import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')

# 2. Clean data and map labels
df['hiv_status_at_booking'] = df['mother_hiv_status_at_booking'].fillna('UNKNOWN/NEW').str.upper()
status_mapping = {
    'POSITIVE': 'Known Positive',
    'UNKNOWN/NEW': 'Unknown Status / New Diagnosis'
}
df['Status Label'] = df['hiv_status_at_booking'].map(status_mapping).fillna('Unknown Status / New Diagnosis')

# 3. Prepare summary table with percentage
summary_df = df['Status Label'].value_counts().reset_index()
summary_df.columns = ['Maternal Status at Booking', 'Total Mothers']
total_mothers = summary_df['Total Mothers'].sum()
summary_df['Percentage (%)'] = ((summary_df['Total Mothers'] / total_mothers) * 100).round(1)

# 4. Create the combined figure (Graph + Table)
fig, ax = plt.subplots(figsize=(10, 8))

# Add the Bar Chart
sns.barplot(x='Maternal Status at Booking', y='Total Mothers', data=summary_df, palette='viridis', ax=ax)
ax.set_title('Maternal HIV Status at Time of ANC Booking', fontsize=16, pad=20)
ax.set_ylabel('Number of Mothers', fontsize=12)

# Add data labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# Add the Data Table at the bottom
# We convert the dataframe to a list and add the '%' sign for display
table_data = summary_df.values.tolist()
for row in table_data:
    row[2] = f"{row[2]}%"

plt.table(cellText=table_data,
          colLabels=summary_df.columns,
          cellLoc='center',
          loc='bottom',
          bbox=[0, -0.35, 1, 0.25])

# Adjust layout to fit the table
plt.subplots_adjust(bottom=0.35)
plt.savefig('outputs/booking_status_with_percentage.png')
plt.show()