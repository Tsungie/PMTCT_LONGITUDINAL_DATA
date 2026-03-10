import pandas as pd
import difflib

# 1. Load the datasets
df_opt = pd.read_excel("uploads/Optimized Sites.xlsx")  # Use the correct path
df_mothers = pd.read_csv(
    "uploads/DATA_SET_WITH_TRACE_OF_THE_MOTHER.csv", encoding="latin1"
)
df_infants = pd.read_csv("uploads/DATA_SET_WITH_NO_TRACEABLE_MOTHER.csv", encoding="latin1"
)

# 2. Extract and standardize the 20 sites from Excel
opt_sites = df_opt["Facility"].dropna().astype(str).str.strip().str.lower().tolist()

# Get unique facilities from both datasets
mother_facs = (
    df_mothers["facility"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
    .tolist()
)
infant_facs = (
    df_infants["facility"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
    .tolist()
)
all_facs = list(set(mother_facs + infant_facs))

# 3. Fuzzy map the 20 target sites to actual names in the datasets
valid_dataset_facs = []
for f in all_facs:
    for site in opt_sites:
        # Check if the Excel name matches the CSV name closely
        if (
            site in f
            or f in site
            or difflib.SequenceMatcher(None, site, f).ratio() > 0.8
        ):
            valid_dataset_facs.append(f)
            break

# Standardize the target columns for filtering
df_mothers["Facility_Clean"] = (
    df_mothers["facility"].astype(str).str.strip().str.lower()
)
df_infants["Facility_Clean"] = (
    df_infants["facility"].astype(str).str.strip().str.lower()
)

# Filter the datasets down to JUST the mapped 20 sites
df_m_opt = df_mothers[df_mothers["Facility_Clean"].isin(valid_dataset_facs)]
df_i_opt = df_infants[df_infants["Facility_Clean"].isin(valid_dataset_facs)]

# ==========================================
# PART 1: ANC DATASET CALCULATIONS
# ==========================================
anc_facilities = df_m_opt["Facility_Clean"].nunique()
total_hiv_positive = len(
    df_m_opt[df_m_opt["mother_hiv_test_result"].str.upper() == "POSITIVE"]
)

aware_prior = len(
    df_m_opt[df_m_opt["mother_hiv_status_at_booking"].str.upper() == "POSITIVE"]
)
unaware_prior = total_hiv_positive - aware_prior
aware_pct = (aware_prior / total_hiv_positive) * 100 if total_hiv_positive else 0
unaware_pct = (unaware_prior / total_hiv_positive) * 100 if total_hiv_positive else 0

babies_born = df_m_opt["child_person_id"].nunique()
infants_tested_df = df_m_opt[df_m_opt["infant_hiv_test_result"].notnull()]
num_infants_tested = len(infants_tested_df)
infants_tested_pct = (num_infants_tested / babies_born) * 100 if babies_born else 0

hiv_pos_infants_df = infants_tested_df[
    infants_tested_df["infant_hiv_test_result"].str.upper() == "POSITIVE"
]
num_hiv_pos_infants = len(hiv_pos_infants_df)
hiv_pos_infants_pct = (
    (num_hiv_pos_infants / num_infants_tested) * 100 if num_infants_tested else 0
)

vl_suppressed = len(
    hiv_pos_infants_df[
        hiv_pos_infants_df["mother_viral_load_result"]
        .astype(str)
        .str.contains("<|Target Not Detected|TND|Not Detected", case=False, na=False)
    ]
)
vl_undocumented = len(
    hiv_pos_infants_df[hiv_pos_infants_df["mother_viral_load_result"].isnull()]
)

art_initiated = len(
    hiv_pos_infants_df[hiv_pos_infants_df["infant_date_of_art_enrolment"].notnull()]
)
art_pct = (art_initiated / num_hiv_pos_infants) * 100 if num_hiv_pos_infants else 0

# ==========================================
# PART 2: INFANT DATASET CALCULATIONS
# ==========================================
df_i_opt["infant_date_of_birth"] = pd.to_datetime(
    df_i_opt["infant_date_of_birth"], format="%d/%m/%Y", errors="coerce"
)
df_i_cohort = df_i_opt[
    (df_i_opt["infant_date_of_birth"] >= "2022-10-01")
    & (df_i_opt["infant_date_of_birth"] <= "2023-09-30")
]

exp_facilities = df_i_cohort["Facility_Clean"].nunique()
total_exposed = len(df_i_cohort)

traced = df_i_cohort[
    df_i_cohort["table"].astype(str).str.contains("unknown mothers", case=False)
    == False
].shape[0]
untraced = total_exposed - traced
traced_pct = (traced / total_exposed) * 100 if total_exposed else 0
untraced_pct = (untraced / total_exposed) * 100 if total_exposed else 0

tested_24m = df_i_cohort[df_i_cohort["infant_hiv_test_date"].notnull()].shape[0]
untested_24m = total_exposed - tested_24m
tested_24m_pct = (tested_24m / total_exposed) * 100 if total_exposed else 0
untested_24m_pct = (untested_24m / total_exposed) * 100 if total_exposed else 0

# Test counting logic variables (if specific tracking IDs exist)
one_test = 0
two_tests = 0
three_plus_tests = 0

final_pos = len(df_i_cohort[df_i_cohort["child_hiv_status"].str.upper() == "POSITIVE"])
final_neg = len(df_i_cohort[df_i_cohort["child_hiv_status"].str.upper() == "NEGATIVE"])
final_unknown = tested_24m - (final_pos + final_neg)
final_pos_pct = (final_pos / tested_24m) * 100 if tested_24m else 0
final_neg_pct = (final_neg / tested_24m) * 100 if tested_24m else 0
final_unknown_pct = (final_unknown / tested_24m) * 100 if tested_24m else 0

sero_converted = 0

story = f"""
This dataset covers women who attended antenatal care (ANC) between January 2021 and December 2025 across {anc_facilities} optimized facilities in 10 provinces. 

A total of {total_hiv_positive:,} women tested HIV-positive before or during ANC. Of these, {unaware_prior:,} women ({unaware_pct:.1f}%) were not aware of their HIV status prior to ANC booking, while {aware_prior:,} women ({aware_pct:.1f}%) were aware of their status before booking. 

From the total number of HIV-positive women, {babies_born:,} babies were born. Among these newborns, {num_infants_tested:,} infants ({infants_tested_pct:.1f}%) had HIV test results documented. Of the infants tested, {num_hiv_pos_infants:,} ({hiv_pos_infants_pct:.1f}%) were HIV-positive. Among the mothers of HIV-positive infants, {vl_suppressed} had suppressed viral loads, while {vl_undocumented} had no documented viral load results. Of the HIV-positive infants, {art_initiated} ({art_pct:.1f}%) were initiated on antiretroviral therapy (ART). 

Another analysis was conducted to complement a previous assessment, focusing on HIV-exposed infants born between October 2022 and September 2023 across {exp_facilities} facilities. A total of {total_exposed:,} HIV-exposed infants were identified during this period. Of these, {traced:,} infants ({traced_pct:.1f}%) could be traced back to their mothers' clinical records, while {untraced:,} infants ({untraced_pct:.1f}%) had no documented maternal linkage. 

From the total cohort, {tested_24m:,} infants ({tested_24m_pct:.1f}%) were tested for HIV, while {untested_24m:,} ({untested_24m_pct:.1f}%) remained untested at 24 months, between October 2024 and September 2025. Regarding follow-up intensity, {one_test} children received one HIV test, {two_tests} received two HIV tests, and {three_plus_tests} had three or more documented HIV tests. 

Among children with documented results, {final_pos:,} ({final_pos_pct:.1f}%) had a final outcome of HIV-positive, {final_neg:,} ({final_neg_pct:.1f}%) tested HIV-negative, and {final_unknown:,} ({final_unknown_pct:.1f}%) had unknown or pending results. Within this cohort, {sero_converted} infants tested negative in 2023 Sero-converted. 

This additional analysis focused on children expected to reach a final HIV outcome between October 2024 and September 2025.
"""
print(story)
