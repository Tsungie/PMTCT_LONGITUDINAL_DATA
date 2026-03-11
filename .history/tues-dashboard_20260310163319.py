import streamlit as st
import pandas as pd
import difflib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PMTCT Optimized Sites Dashboard", layout="wide")
st.title("PMTCT Longitudinal Data: Optimized Sites Dashboard")
st.markdown(
    "This interactive dashboard tracks ANC attendance and HIV-exposed infant outcomes across the 20 Optimized Sites."
)


# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_process_data():
    # 1. Load Data
    # Update these filenames if they differ on your local machine
    df_opt = pd.read_excel("uploads/Optimized Sites.xlsx")
    df_mothers = pd.read_csv("DATA SET WITH TRACE OF THE MOTHER.csv", encoding="latin1")
    df_infants = pd.read_csv(
        "DATA SET WITH NO TRACEABLE MOTHERS.csv", encoding="latin1"
    )

    # 2. Clean facility strings
    opt_sites = df_opt["Facility"].dropna().astype(str).str.strip().str.lower().tolist()
    df_mothers["Facility_Clean"] = (
        df_mothers["facility"].astype(str).str.strip().str.lower()
    )
    df_infants["Facility_Clean"] = (
        df_infants["facility"].astype(str).str.strip().str.lower()
    )

    all_facs = list(
        set(
            df_mothers["Facility_Clean"].unique().tolist()
            + df_infants["Facility_Clean"].unique().tolist()
        )
    )

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

    # 4. Filter datasets to matched sites
    df_m_opt = df_mothers[
        df_mothers["Facility_Clean"].isin(fac_to_optsite.keys())
    ].copy()
    df_i_opt = df_infants[
        df_infants["Facility_Clean"].isin(fac_to_optsite.keys())
    ].copy()

    df_m_opt["Optimized_Site"] = df_m_opt["Facility_Clean"].map(fac_to_optsite)
    df_i_opt["Optimized_Site"] = df_i_opt["Facility_Clean"].map(fac_to_optsite)

    # 5. Aggregate ANC Data
    df_m_pos = df_m_opt[df_m_opt["mother_hiv_test_result"].str.upper() == "POSITIVE"]
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

    # 6. Aggregate Infant Cohort Data
    df_i_opt["infant_date_of_birth"] = pd.to_datetime(
        df_i_opt["infant_date_of_birth"], format="%d/%m/%Y", errors="coerce"
    )
    df_i_cohort = df_i_opt[
        (df_i_opt["infant_date_of_birth"] >= "2022-10-01")
        & (df_i_opt["infant_date_of_birth"] <= "2023-09-30")
    ]

    infant_summary = (
        df_i_cohort.groupby("Optimized_Site")
        .agg(
            Total_Exposed=("infant_sex", "count"),
            Tested_24m=("infant_hiv_test_date", lambda x: x.notnull().sum()),
            Final_Pos=(
                "child_hiv_status",
                lambda x: (x.str.upper() == "POSITIVE").sum(),
            ),
        )
        .reset_index()
    )

    # 7. Merge into final dashboard dataframe
    dashboard_df = pd.merge(
        anc_summary, infant_summary, on="Optimized_Site", how="outer"
    ).fillna(0)

    # Ensure all sites show up even if 0
    all_opt_title = [s.title() for s in opt_sites]
    missing_sites = set(all_opt_title) - set(dashboard_df["Optimized_Site"])
    if missing_sites:
        missing_df = pd.DataFrame({"Optimized_Site": list(missing_sites)})
        dashboard_df = pd.concat([dashboard_df, missing_df], ignore_index=True).fillna(
            0
        )

    # Convert numeric columns to int
    num_cols = dashboard_df.columns.drop("Optimized_Site")
    dashboard_df[num_cols] = dashboard_df[num_cols].astype(int)

    return dashboard_df.sort_values("Total_HIV_Positive", ascending=True)


# Fetch the data
with st.spinner("Loading and mapping dataset..."):
    df = load_and_process_data()

# --- TOP LEVEL METRICS ---
st.header("Overall Summary (20 Optimized Sites)")
col1, col2, col3, col4 = st.columns(4)

total_hiv_pos = df["Total_HIV_Positive"].sum()
total_aware = df["Aware_Prior"].sum()
total_unaware = df["Unaware_Prior"].sum()
total_exposed = df["Total_Exposed"].sum()

col1.metric("Total HIV+ Women (ANC)", f"{total_hiv_pos:,}")
col2.metric(
    "Unaware Prior to Booking",
    f"{total_unaware:,}",
    f"{(total_unaware/total_hiv_pos*100):.1f}% of total" if total_hiv_pos else "0%",
)
col3.metric(
    "Aware Prior to Booking",
    f"{total_aware:,}",
    f"{(total_aware/total_hiv_pos*100):.1f}% of total" if total_hiv_pos else "0%",
)
col4.metric("Total Exposed Infants (22/23 Cohort)", f"{total_exposed:,}")

st.divider()

# --- VISUALIZATIONS ---
st.header("Site-by-Site Breakdown")
col_chart1, col_chart2 = st.columns(2)

# Chart 1: ANC Women (Stacked Bar)
with col_chart1:
    st.subheader("ANC HIV+ Women: Aware vs Unaware")
    # Melt the dataframe for Plotly stacked bar chart
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
    st.plotly_chart(fig_anc, use_container_width=True)

# Chart 2: Infant Cohort (Grouped Bar)
with col_chart2:
    st.subheader("Exposed Infants: Total vs Tested")
    df_melt_infant = df.melt(
        id_vars="Optimized_Site",
        value_vars=["Total_Exposed", "Tested_24m"],
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
        color_discrete_map={"Total_Exposed": "#99ff99", "Tested_24m": "#ffcc99"},
        labels={"Count": "Number of Infants", "Optimized_Site": "Facility"},
    )
    st.plotly_chart(fig_inf, use_container_width=True)

st.divider()

# --- DATA TABLE ---
st.header("Raw Data Extract")
st.markdown(
    "Use this table to view the exact metrics per facility. You can click the column headers to sort, or download the data as a CSV."
)
st.dataframe(df, use_container_width=True, hide_index=True)
