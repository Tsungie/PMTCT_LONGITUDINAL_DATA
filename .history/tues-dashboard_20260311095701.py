import streamlit as st
import pandas as pd
import difflib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PMTCT Optimized Sites Dashboard", layout="wide")
st.title("PMTCT Longitudinal Data: Optimized Sites Dashboard")
st.markdown(
    "This interactive dashboard tracks ANC attendance and HIV-exposed infant outcomes across the 20 Optimized Sites, using exclusively the Traceable Mothers dataset."
)


# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_process_data():
    # 1. Load Data
    df_opt = pd.read_csv("Optimized Sites.xlsx - Sheet1.csv")
    df_mothers = pd.read_csv("DATA SET WITH TRACE OF THE MOTHER.csv", encoding="latin1")

    # 2. Clean facility strings
    opt_sites = df_opt["Facility"].dropna().astype(str).str.strip().str.lower().tolist()
    df_mothers["Facility_Clean"] = (
        df_mothers["facility"].astype(str).str.strip().str.lower()
    )

    all_facs = df_mothers["Facility_Clean"].unique().tolist()

    # 3. Map dataset facilities back to the clean Optimized Site names
    fac_to_optsite = {}
    for f in all_facs:
        for site in opt_sites:
            if (
                site in f
                or f in site
                or difflib.SequenceMatcher(None, site, f).ratio() > 0.8
            ):
                fac_to_optsite[f] = site.title()
                break

    # 4. Filter dataset to strictly the 20 matched sites
    df_m_opt = df_mothers[
        df_mothers["Facility_Clean"].isin(fac_to_optsite.keys())
    ].copy()
    df_m_opt["Optimized_Site"] = df_m_opt["Facility_Clean"].map(fac_to_optsite)

    # 5. Format infant_date_of_birth robustly (Force reads mixed Excel formats row-by-row)
    # Convert to string, clean spaces, and use format='mixed' to handle changing formats
    df_m_opt["infant_date_of_birth"] = (
        df_m_opt["infant_date_of_birth"].astype(str).str.strip()
    )

    try:
        df_m_opt["infant_date_of_birth"] = pd.to_datetime(
            df_m_opt["infant_date_of_birth"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except ValueError:
        # Fallback for slightly older versions of Pandas
        df_m_opt["infant_date_of_birth"] = pd.to_datetime(
            df_m_opt["infant_date_of_birth"],
            infer_datetime_format=True,
            dayfirst=True,
            errors="coerce",
        )

    # ==========================================
    # DATASET FILTERING
    # ==========================================
    # Part 1: ANC Dataset (ALL TIME - No date filters)
    df_anc = df_m_opt.copy()

    # Part 2: Infant Cohort strictly born between Oct 2022 and Sep 2023
    df_infant_cohort = df_m_opt[
        (df_m_opt["infant_date_of_birth"] >= "2022-10-01")
        & (df_m_opt["infant_date_of_birth"] <= "2023-09-30")
    ].copy()
    df_infant_cohort = df_infant_cohort[
        df_infant_cohort["mother_hiv_test_result"].str.upper() == "POSITIVE"
    ]

    # ==========================================
    # AGGREGATE DASHBOARD TABLE (PART 1)
    # ==========================================
    df_m_pos = df_anc[df_anc["mother_hiv_test_result"].str.upper() == "POSITIVE"]
    anc_summary = (
        df_m_pos.groupby("Optimized_Site")
        .agg(
            Total_HIV_Positive=("mother_hiv_test_result", "count"),
            Aware_Prior=(
                "mother_hiv_status_at_booking",
                lambda x: (x.str.upper() == "POSITIVE").sum(),
            ),
            Babies_Born=("child_person_id", "nunique"),
            Infants_Tested=("infant_hiv_test_result", lambda x: x.notnull().sum()),
            Infants_HIV_Pos=(
                "infant_hiv_test_result",
                lambda x: (x.str.upper() == "POSITIVE").sum(),
            ),
        )
        .reset_index()
    )
    anc_summary["Unaware_Prior"] = (
        anc_summary["Total_HIV_Positive"] - anc_summary["Aware_Prior"]
    )

    # ==========================================
    # AGGREGATE DASHBOARD TABLE (PART 2)
    # ==========================================
    infant_summary = (
        df_infant_cohort.groupby("Optimized_Site")
        .agg(
            Total_Exposed_Cohort=("child_person_id", "nunique"),
            Untraced=("child_person_id", lambda x: 0),
            Tested_24m=("infant_hiv_test_date", lambda x: x.notnull().sum()),
            Final_Pos=(
                "child_hiv_status",
                lambda x: (x.str.upper() == "POSITIVE").sum(),
            ),
            Final_Neg=(
                "child_hiv_status",
                lambda x: (x.str.upper() == "NEGATIVE").sum(),
            ),
        )
        .reset_index()
    )
    infant_summary["Traced"] = infant_summary["Total_Exposed_Cohort"]

    dashboard_df = pd.merge(
        anc_summary, infant_summary, on="Optimized_Site", how="outer"
    ).fillna(0)

    # Ensure all 20 sites show up
    all_opt_title = [s.title() for s in opt_sites]
    missing_sites = set(all_opt_title) - set(dashboard_df["Optimized_Site"])
    if missing_sites:
        missing_df = pd.DataFrame({"Optimized_Site": list(missing_sites)})
        dashboard_df = pd.concat([dashboard_df, missing_df], ignore_index=True).fillna(
            0
        )

    num_cols = dashboard_df.columns.drop("Optimized_Site")
    dashboard_df[num_cols] = dashboard_df[num_cols].astype(int)
    dashboard_df = dashboard_df.sort_values("Total_HIV_Positive", ascending=True)

    # ==========================================
    # CALCULATE NARRATIVE TEXT METRICS
    # ==========================================
    # Part 1 Metrics (All Time)
    anc_facilities = df_anc["Facility_Clean"].nunique()
    total_hiv_positive = len(df_m_pos)
    aware_prior = len(
        df_m_pos[df_m_pos["mother_hiv_status_at_booking"].str.upper() == "POSITIVE"]
    )
    unaware_prior = total_hiv_positive - aware_prior
    aware_pct = (aware_prior / total_hiv_positive) * 100 if total_hiv_positive else 0
    unaware_pct = (
        (unaware_prior / total_hiv_positive) * 100 if total_hiv_positive else 0
    )

    babies_born_all_time = df_m_pos["child_person_id"].nunique()
    infants_tested_df = df_m_pos[df_m_pos["infant_hiv_test_result"].notnull()]
    num_infants_tested = len(infants_tested_df)
    infants_tested_pct = (
        (num_infants_tested / babies_born_all_time) * 100 if babies_born_all_time else 0
    )

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
            .str.contains(
                "<|Target Not Detected|TND|Not Detected", case=False, na=False
            )
        ]
    )
    vl_undocumented = len(
        hiv_pos_infants_df[hiv_pos_infants_df["mother_viral_load_result"].isnull()]
    )

    art_initiated = len(
        hiv_pos_infants_df[hiv_pos_infants_df["infant_date_of_art_enrolment"].notnull()]
    )
    art_pct = (art_initiated / num_hiv_pos_infants) * 100 if num_hiv_pos_infants else 0

    # Part 2 Metrics (Oct 2022 - Sep 2023 Cohort)
    exp_facilities = df_infant_cohort["Facility_Clean"].nunique()
    total_exposed_cohort = len(df_infant_cohort)
    traced = total_exposed_cohort
    untraced = 0
    traced_pct = 100.0 if total_exposed_cohort else 0
    untraced_pct = 0.0

    tested_24m = df_infant_cohort["infant_hiv_test_date"].notnull().sum()
    untested_24m = total_exposed_cohort - tested_24m
    tested_24m_pct = (
        (tested_24m / total_exposed_cohort) * 100 if total_exposed_cohort else 0
    )
    untested_24m_pct = (
        (untested_24m / total_exposed_cohort) * 100 if total_exposed_cohort else 0
    )

    final_pos = (df_infant_cohort["child_hiv_status"].str.upper() == "POSITIVE").sum()
    final_neg = (df_infant_cohort["child_hiv_status"].str.upper() == "NEGATIVE").sum()
    final_unknown = tested_24m - (final_pos + final_neg)
    final_pos_pct = (final_pos / tested_24m) * 100 if tested_24m else 0
    final_neg_pct = (final_neg / tested_24m) * 100 if tested_24m else 0
    final_unknown_pct = (final_unknown / tested_24m) * 100 if tested_24m else 0

    story_part1 = f"""**ANC Dataset Overview:** This dataset covers women who attended antenatal care (ANC) across {anc_facilities} optimized facilities in 10 provinces. 
    
A total of {total_hiv_positive:,} women tested HIV-positive before or during ANC. Of these, {unaware_prior:,} women ({unaware_pct:.1f}%) were not aware of their HIV status prior to ANC booking, while {aware_prior:,} women ({aware_pct:.1f}%) were aware of their status before booking. 
    
From the total number of HIV-positive women, {babies_born_all_time:,} babies were born. Among these newborns, {num_infants_tested:,} infants ({infants_tested_pct:.1f}%) had HIV test results documented. Of the infants tested, {num_hiv_pos_infants:,} ({hiv_pos_infants_pct:.1f}%) were HIV-positive. Among the mothers of HIV-positive infants, {vl_suppressed} had suppressed viral loads, while {vl_undocumented} had no documented viral load results. Of the HIV-positive infants, {art_initiated} ({art_pct:.1f}%) were initiated on antiretroviral therapy (ART)."""

    story_part2 = f"""**HIV-Exposed Infant Cohort (Oct 2022 - Sep 2023):** Another analysis was conducted to complement a previous assessment, focusing on HIV-exposed infants born between October 2022 and September 2023 across {exp_facilities} facilities. A total of {total_exposed_cohort:,} HIV-exposed infants were identified during this period. Of these, {traced:,} infants ({traced_pct:.1f}%) could be traced back to their mothers' clinical records, while {untraced:,} infants ({untraced_pct:.1f}%) had no documented maternal linkage. 
    
From the total cohort, {tested_24m:,} infants ({tested_24m_pct:.1f}%) were tested for HIV, while {untested_24m:,} ({untested_24m_pct:.1f}%) remained untested at 24 months. Among children with documented results, {final_pos:,} ({final_pos_pct:.1f}%) had a final outcome of HIV-positive, {final_neg:,} ({final_neg_pct:.1f}%) tested HIV-negative, and {final_unknown:,} ({final_unknown_pct:.1f}%) had unknown or pending results."""

    return dashboard_df, story_part1, story_part2, num_hiv_pos_infants


# Fetch the data and stories
with st.spinner("Loading and generating report metrics..."):
    df, story_part1, story_part2, num_hiv_pos_infants = load_and_process_data()

# --- TOP LEVEL METRICS ---
st.header("Overall Summary (20 Optimized Sites)")
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_hiv_pos = df["Total_HIV_Positive"].sum()
total_unaware = df["Unaware_Prior"].sum()
total_aware = df["Aware_Prior"].sum()
total_exposed_all_time = df["Babies_Born"].sum()
total_exposed_cohort = df["Total_Exposed_Cohort"].sum()

col1.metric("Total HIV+ Women (ANC)", f"{total_hiv_pos:,}")
col2.metric(
    "Unaware Prior to Booking",
    f"{total_unaware:,}",
    f"{(total_unaware/total_hiv_pos*100):.1f}%" if total_hiv_pos else "0%",
)
col3.metric(
    "Aware Prior to Booking",
    f"{total_aware:,}",
    f"{(total_aware/total_hiv_pos*100):.1f}%" if total_hiv_pos else "0%",
)
col4.metric("Total Exposed Infants (All Time)", f"{total_exposed_all_time:,}")
col5.metric("Infants Tested Positive (All Time)", f"{num_hiv_pos_infants:,}")
col6.metric("Exposed Infants (Oct '22 - Sep '23)", f"{total_exposed_cohort:,}")

st.divider()

# ==========================================
# PART 1 VISUALIZATIONS
# ==========================================
st.header("Part 1: ANC Dataset Breakdown (All Time)")
st.info(story_part1)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("ANC HIV+ Women: Aware vs Unaware")
    df_melt_anc = df.melt(
        id_vars="Optimized_Site",
        value_vars=["Unaware_Prior", "Aware_Prior"],
        var_name="Status",
        value_name="Count",
    )

    fig_anc = px.bar(
        df_melt_anc,
        x="Count",
        y="Optimized_Site",
        color="Status",
        orientation="h",
        color_discrete_map={"Unaware_Prior": "#ff9999", "Aware_Prior": "#66b3ff"},
        labels={"Count": "Number of Women", "Optimized_Site": "Facility"},
    )
    st.plotly_chart(fig_anc, use_container_width=True, key="chart_anc_aware")

with col_chart2:
    st.subheader("Exposed Infants: Total Born vs Tested (ANC)")
    df_melt_infant = df.melt(
        id_vars="Optimized_Site",
        value_vars=["Babies_Born", "Infants_Tested"],
        var_name="Metric",
        value_name="Count",
    )

    fig_inf = px.bar(
        df_melt_infant,
        x="Count",
        y="Optimized_Site",
        color="Metric",
        orientation="h",
        barmode="group",
        color_discrete_map={"Babies_Born": "#99ff99", "Infants_Tested": "#ffcc99"},
        labels={"Count": "Number of Infants", "Optimized_Site": "Facility"},
    )
    st.plotly_chart(fig_inf, use_container_width=True, key="chart_anc_tested")


# ==========================================
# PART 2 VISUALIZATIONS
# ==========================================
st.divider()
st.header("Part 2: HIV-Exposed Infant Linkage & Outcomes (Oct 2022 - Sep 2023)")
st.info(story_part2)

col_part2_1, col_part2_2 = st.columns(2)

with col_part2_1:
    st.subheader("Maternal Linkage: Traced vs Untraced")
    df["Untraced_Display"] = 0
    df["Traced_Display"] = df["Total_Exposed_Cohort"]

    df_melt_linkage = df[df["Total_Exposed_Cohort"] > 0].melt(
        id_vars="Optimized_Site",
        value_vars=["Untraced_Display", "Traced_Display"],
        var_name="Linkage Status",
        value_name="Count",
    )

    fig_linkage = px.bar(
        df_melt_linkage,
        x="Count",
        y="Optimized_Site",
        color="Linkage Status",
        orientation="h",
        color_discrete_map={"Untraced_Display": "#ff9999", "Traced_Display": "#66b3ff"},
        labels={"Count": "Number of Infants", "Optimized_Site": "Facility"},
    )
    st.plotly_chart(fig_linkage, use_container_width=True, key="chart_part2_linkage")

with col_part2_2:
    st.subheader("Final HIV Outcomes at 24 Months")
    df["Final_Negative"] = df["Final_Neg"]
    df["Final_Unknown"] = df["Tested_24m"] - (df["Final_Pos"] + df["Final_Negative"])

    df_melt_outcomes = df[df["Tested_24m"] > 0].melt(
        id_vars="Optimized_Site",
        value_vars=["Final_Pos", "Final_Negative", "Final_Unknown"],
        var_name="Outcome",
        value_name="Count",
    )

    fig_outcomes = px.bar(
        df_melt_outcomes,
        x="Count",
        y="Optimized_Site",
        color="Outcome",
        orientation="h",
        color_discrete_map={
            "Final_Pos": "#ff6666",
            "Final_Negative": "#99cc99",
            "Final_Unknown": "#cccccc",
        },
        labels={"Count": "Number of Infants", "Optimized_Site": "Facility"},
    )
    st.plotly_chart(fig_outcomes, use_container_width=True, key="chart_part2_outcomes")

st.divider()

# ==========================================
# TABLE 1: ANC & PMTCT OVERVIEW
# ==========================================
st.header("Table 1: ANC Attendance & PMTCT Testing Overview (All Time)")
st.markdown(
    "This table provides a facility-level breakdown of HIV-positive women attending Antenatal Care (ANC) and the subsequent HIV testing and treatment initiation for their newborns."
)

table1_cols = [
    "Optimized_Site",
    "Total_HIV_Positive",
    "Aware_Prior",
    "Unaware_Prior",
    "Babies_Born",
    "Infants_Tested",
    "Infants_HIV_Pos",
]
st.dataframe(df[table1_cols], use_container_width=True, hide_index=True)

st.write("\n")

# ==========================================
# TABLE 2: INFANT LINKAGE & OUTCOMES
# ==========================================
st.header("Table 2: HIV-Exposed Infant Linkage & Outcomes (Oct 2022 - Sep 2023 Cohort)")
st.markdown(
    "This table isolates the specific cohort of HIV-exposed infants born between Oct 1, 2022 and Sep 30, 2023. **Note: Because this analysis is drawn exclusively from the Traceable Mothers dataset, 100% of infants are traced.**"
)

part2_table = df[df["Total_Exposed_Cohort"] > 0][
    [
        "Optimized_Site",
        "Total_Exposed_Cohort",
        "Traced",
        "Untraced",
        "Tested_24m",
        "Final_Pos",
        "Final_Negative",
        "Final_Unknown",
    ]
]
st.dataframe(part2_table, use_container_width=True, hide_index=True)
