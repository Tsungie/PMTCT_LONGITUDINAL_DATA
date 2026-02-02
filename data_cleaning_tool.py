"""
VISUAL DATA CLEANING UTILITY FOR PMTCT DATASETS
================================================
This script provides visual tools to identify and fix data quality issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_data_quality_report(df, dataset_name):
    """
    Generate comprehensive visual data quality report
    """
    print("="*70)
    print(f"DATA QUALITY REPORT: {dataset_name}")
    print("="*70)
    
    # Basic info
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing data
    print(f"\n📊 MISSING DATA:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    }).sort_values('Missing', ascending=False)
    
    print(missing_df[missing_df['Missing'] > 0].to_string(index=False))
    
    # Data types
    print(f"\n📊 DATA TYPES:")
    print(df.dtypes.value_counts())
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n📊 DUPLICATES: {duplicates:,} rows ({duplicates/len(df)*100:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Data Quality Dashboard: {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Missing data heatmap
    ax = axes[0, 0]
    missing_matrix = df.isnull()
    # Sample if too large
    if len(df) > 1000:
        missing_matrix = missing_matrix.sample(1000)
    sns.heatmap(missing_matrix, cbar=False, yticklabels=False, 
                cmap='RdYlGn_r', ax=ax)
    ax.set_title('Missing Data Pattern (Sample)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Columns')
    
    # 2. Missing data bar chart
    ax = axes[0, 1]
    top_missing = missing_df.head(15)
    ax.barh(range(len(top_missing)), top_missing['Percentage'], 
            color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing['Column'], fontsize=9)
    ax.set_xlabel('Missing %', fontsize=10)
    ax.set_title('Top 15 Columns with Missing Data', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # 3. Data type distribution
    ax = axes[1, 0]
    dtype_counts = df.dtypes.value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
           colors=colors[:len(dtype_counts)], startangle=90)
    ax.set_title('Data Type Distribution', fontsize=12, fontweight='bold')
    
    # 4. Column completeness
    ax = axes[1, 1]
    completeness = ((len(df) - missing) / len(df)) * 100
    completeness_sorted = completeness.sort_values()
    
    # Color code: red if <50%, yellow if <90%, green if >=90%
    colors_comp = ['#e74c3c' if x < 50 else '#f39c12' if x < 90 else '#27ae60' 
                   for x in completeness_sorted.values]
    
    ax.barh(range(len(completeness_sorted)), completeness_sorted.values,
            color=colors_comp, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(completeness_sorted)))
    ax.set_yticklabels(completeness_sorted.index, fontsize=7)
    ax.set_xlabel('Completeness %', fontsize=10)
    ax.set_title('Column Completeness', fontsize=12, fontweight='bold')
    ax.axvline(90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% threshold')
    ax.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    return fig, missing_df

def identify_inconsistent_values(df, column_name):
    """
    Identify inconsistent values in a column (e.g., POSITIVE vs Positive)
    """
    print(f"\n{'='*70}")
    print(f"INCONSISTENCY ANALYSIS: {column_name}")
    print(f"{'='*70}")
    
    if column_name not in df.columns:
        print(f"⚠️  Column '{column_name}' not found!")
        return
    
    # Get value counts
    values = df[column_name].value_counts()
    
    print(f"\nUnique values: {len(values)}")
    print(f"\nValue distribution:")
    print(values.head(20))
    
    # Check for case variations
    if df[column_name].dtype == 'object':
        # Group by uppercase version
        df['_temp_upper'] = df[column_name].str.upper()
        upper_groups = df.groupby('_temp_upper')[column_name].apply(lambda x: x.value_counts())
        
        print(f"\n{'='*70}")
        print("CASE VARIATION DETECTION:")
        print(f"{'='*70}")
        
        found_variations = False
        for upper_val, variations in upper_groups.groupby(level=0):
            if len(variations) > 1:
                found_variations = True
                print(f"\n'{upper_val}' has {len(variations)} variations:")
                for var, count in variations.items():
                    print(f"  - '{var}': {count}")
        
        if not found_variations:
            print("\n✅ No case variations found!")
        
        df.drop('_temp_upper', axis=1, inplace=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Value distribution
    ax = axes[0]
    top_20 = values.head(20)
    ax.barh(range(len(top_20)), top_20.values, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index, fontsize=9)
    ax.set_xlabel('Count', fontsize=10)
    ax.set_title(f'Top 20 Values: {column_name}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Missing vs non-missing
    ax = axes[1]
    missing_count = df[column_name].isnull().sum()
    non_missing = len(df) - missing_count
    ax.pie([non_missing, missing_count], 
           labels=['Non-Missing', 'Missing'],
           autopct='%1.1f%%',
           colors=['#27ae60', '#e74c3c'],
           startangle=90)
    ax.set_title(f'Completeness: {column_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_date_columns(df):
    """
    Analyze date columns for issues
    """
    print(f"\n{'='*70}")
    print("DATE COLUMN ANALYSIS")
    print(f"{'='*70}")
    
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    if len(date_cols) == 0:
        print("\n⚠️  No datetime columns found. Converting date-like columns...")
        
        # Try to identify date columns
        for col in df.columns:
            if 'date' in col.lower():
                print(f"\nTrying to convert: {col}")
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    print(f"  ✅ Converted successfully")
                except:
                    print(f"  ❌ Failed to convert")
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
    
    print(f"\nFound {len(date_cols)} date columns:")
    
    for col in date_cols:
        print(f"\n{col}:")
        print(f"  Valid dates: {df[col].notna().sum():,}")
        print(f"  Missing: {df[col].isna().sum():,}")
        
        valid_dates = df[col].dropna()
        if len(valid_dates) > 0:
            print(f"  Earliest: {valid_dates.min()}")
            print(f"  Latest: {valid_dates.max()}")
            
            # Check for anomalies
            future_dates = valid_dates[valid_dates > pd.Timestamp.now()]
            if len(future_dates) > 0:
                print(f"  ⚠️  Future dates: {len(future_dates)}")
            
            very_old = valid_dates[valid_dates < pd.Timestamp('1900-01-01')]
            if len(very_old) > 0:
                print(f"  ⚠️  Very old dates (<1900): {len(very_old)}")
    
    # Visualize date ranges
    if len(date_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, max(6, len(date_cols) * 0.8)))
        
        for i, col in enumerate(date_cols):
            valid_dates = df[col].dropna()
            if len(valid_dates) > 0:
                years = valid_dates.dt.year.value_counts().sort_index()
                ax.barh([i] * len(years), years.values, 
                       left=years.index, height=0.8,
                       alpha=0.7, label=col)
        
        ax.set_yticks(range(len(date_cols)))
        ax.set_yticklabels(date_cols)
        ax.set_xlabel('Year', fontsize=10)
        ax.set_title('Date Column Ranges', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig

def suggest_cleaning_actions(df):
    """
    Suggest cleaning actions based on data quality issues
    """
    print(f"\n{'='*70}")
    print("🔧 SUGGESTED CLEANING ACTIONS")
    print(f"{'='*70}")
    
    suggestions = []
    
    # Check missing data
    missing = df.isnull().sum()
    high_missing = missing[missing / len(df) > 0.5]
    
    if len(high_missing) > 0:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'{len(high_missing)} columns with >50% missing data',
            'action': f'Consider dropping: {list(high_missing.index)}',
            'code': f"df.drop({list(high_missing.index)}, axis=1, inplace=True)"
        })
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'{duplicates} duplicate rows found',
            'action': 'Remove duplicates',
            'code': "df.drop_duplicates(inplace=True)"
        })
    
    # Check for object columns (potential inconsistencies)
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'{len(object_cols)} text columns may have inconsistencies',
            'action': 'Standardize text (uppercase/strip whitespace)',
            'code': "df['column'] = df['column'].str.upper().str.strip()"
        })
    
    # Check date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    non_datetime_dates = [col for col in date_cols if df[col].dtype != 'datetime64[ns]']
    if len(non_datetime_dates) > 0:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'{len(non_datetime_dates)} date columns not in datetime format',
            'action': 'Convert to datetime',
            'code': "df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce', dayfirst=True)"
        })
    
    # Print suggestions
    for i, sug in enumerate(suggestions, 1):
        priority_icon = '🚨' if sug['priority'] == 'HIGH' else '⚠️'
        print(f"\n{priority_icon} {sug['priority']} PRIORITY #{i}:")
        print(f"   Issue: {sug['issue']}")
        print(f"   Action: {sug['action']}")
        print(f"   Code: {sug['code']}")
    
    if len(suggestions) == 0:
        print("\n✅ No major data quality issues detected!")
    
    return suggestions

def clean_text_column(df, column_name, standardize=True, strip=True, 
                     remove_extra_spaces=True):
    """
    Clean a text column
    """
    print(f"\nCleaning column: {column_name}")
    
    if column_name not in df.columns:
        print(f"⚠️  Column not found!")
        return df
    
    original_unique = df[column_name].nunique()
    
    if standardize:
        df[column_name] = df[column_name].str.upper()
    
    if strip:
        df[column_name] = df[column_name].str.strip()
    
    if remove_extra_spaces:
        df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
    
    new_unique = df[column_name].nunique()
    
    print(f"✅ Cleaned!")
    print(f"   Unique values before: {original_unique}")
    print(f"   Unique values after: {new_unique}")
    print(f"   Reduced by: {original_unique - new_unique}")
    
    return df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def analyze_datasets():
    """Main analysis function"""
    
    # Load datasets
    print("Loading datasets...\n")
    no_mother = pd.read_csv('uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv')
    with_mother = pd.read_csv('uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv')
    
    # Dataset 1 analysis
    print("\n\n" + "="*70)
    print("ANALYZING: DATASET 1 (NO TRACEABLE MOTHER)")
    print("="*70)
    
    fig1, missing1 = generate_data_quality_report(no_mother, "No Traceable Mother")
    fig1.savefig('outputs/cleaning/data_quality_no_mother.png', 
                 dpi=300, bbox_inches='tight')
    print("\n✅ Saved: data_quality_no_mother.png")
    
    # Analyze specific columns
    print("\n" + "="*70)
    print("COLUMN-SPECIFIC ANALYSIS")
    print("="*70)
    
    # Check HIV test result consistency
    fig2 = identify_inconsistent_values(no_mother, 'infant_hiv_test_result')
    if fig2:
        fig2.savefig('outputs/cleaning/hiv_result_consistency.png', 
                     dpi=300, bbox_inches='tight')
        print("\n✅ Saved: hiv_result_consistency.png")
    
    # Date analysis
    fig3 = analyze_date_columns(no_mother)
    if fig3:
        fig3.savefig('outputs/cleaning/date_analysis_no_mother.png', 
                     dpi=300, bbox_inches='tight')
        print("\n✅ Saved: date_analysis_no_mother.png")
    
    # Suggestions
    suggestions1 = suggest_cleaning_actions(no_mother)
    
    # Dataset 2 analysis
    print("\n\n" + "="*70)
    print("ANALYZING: DATASET 2 (WITH MOTHER DATA)")
    print("="*70)
    
    fig4, missing2 = generate_data_quality_report(with_mother, "With Mother Data")
    fig4.savefig('outputs/cleaning/data_quality_with_mother.png', 
                 dpi=300, bbox_inches='tight')
    print("\n✅ Saved: data_quality_with_mother.png")
    
    # Date analysis
    fig5 = analyze_date_columns(with_mother)
    if fig5:
        fig5.savefig('outputs/cleaning/date_analysis_with_mother.png', 
                     dpi=300, bbox_inches='tight')
        print("\n✅ Saved: date_analysis_with_mother.png")
    
    # Suggestions
    suggestions2 = suggest_cleaning_actions(with_mother)
    
    print("\n\n" + "="*70)
    print("DATA QUALITY ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Generated visualizations:")
    print("  • data_quality_no_mother.png")
    print("  • hiv_result_consistency.png")
    print("  • date_analysis_no_mother.png")
    print("  • data_quality_with_mother.png")
    print("  • date_analysis_with_mother.png")

if __name__ == "__main__":
    analyze_datasets()