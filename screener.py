import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Institutional Batch Stock Screener & Valuation Vault", layout="wide")

st.title("🛡️ Institutional Batch Stock Screener & Valuation Vault")
st.markdown("Upload your fundamental CSV export (e.g., from Screener.in) to instantly batch-apply Moat grading, Reverse DCF implied growth ($g_1$), staggered buy-zones, and institutional exit playbooks across your entire stock universe.")

# --- SIDEBAR: CONFIGURATION & UPLOAD ---
st.sidebar.header("1. Upload Stock Universe CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("2. Global Assumptions")
default_wacc = st.sidebar.slider("Assumed WACC (r %)", 8.0, 18.0, 12.0, 0.5)
default_term_g = st.sidebar.slider("Terminal Growth Rate ($g_{term}$ %)", 2.0, 6.0, 4.0, 0.25)

st.sidebar.header("3. Institutional Filters")
min_roce = st.sidebar.slider("Min ROCE (%)", 0.0, 50.0, 15.0, 1.0)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 2.0, 0.5, 0.1)
selected_moats = st.sidebar.multiselect("Filter Moat Grades", ["A+", "A", "B", "C", "D"], default=["A+", "A", "B"])

# --- CORE INSTITUTIONAL LOGIC FUNCTIONS ---
def calculate_moat_grade(roce, roe, de):
    if roce >= 20.0 and roe >= 18.0 and de <= 0.30:
        return "A+"
    elif roce >= 15.0 and roe >= 14.0 and de <= 0.60:
        return "A"
    elif roce >= 12.0 and de <= 1.0:
        return "B"
    elif roce >= 8.0:
        return "C"
    else:
        return "D"

def solve_reverse_dcf(cmp, fcf_0, wacc, term_g, shares):
    if not cmp or cmp <= 0 or not fcf_0 or fcf_0 <= 0 or not shares or shares <= 0:
        return 10.0 # Default fallback growth rate
    
    cf_per_share = fcf_0 / shares
    if cf_per_share <= 0:
        return 5.0
        
    # Bisection search for 10-year implied growth g1
    low, high = -0.30, 1.00 # -30% to 100% growth
    best_g = 10.0
    
    for _ in range(50):
        mid = (low + high) / 2.0
        pv = 0
        current_cf = cf_per_share
        for yr in range(1, 11):
            current_cf *= (1.0 + mid)
            pv += current_cf / ((1.0 + wacc) ** yr)
        
        terminal_val = (current_cf * (1.0 + term_g)) / (wacc - term_g) if wacc > term_g else current_cf * 10
        pv_terminal = terminal_val / ((1.0 + wacc) ** 10)
        total_iv = pv + pv_terminal
        
        if total_iv > cmp:
            high = mid
        else:
            low = mid
        best_g = mid * 100.0
    return round(max(-20.0, min(80.0, best_g)), 2)

def compute_buy_zones(cmp, dcf_iv, graham_val, moat_grade):
    # Anchor price calculation
    valid_vals = [v for v in [dcf_iv, graham_val] if v and v > 0]
    anchor = np.median(valid_vals) if valid_vals else cmp
    
    discount_multiplier = 0.85 if moat_grade in ["A+", "A"] else 0.75
    sip_zone = (anchor * 0.75, anchor * 0.90)
    garp_zone = (anchor * 0.90, anchor * 1.10)
    deep_value = (0.0, anchor * 0.75)
    
    in_sip = sip_zone[0] <= cmp <= sip_zone[1]
    in_garp = garp_zone[0] <= cmp <= garp_zone[1]
    in_deep = cmp <= deep_value[1]
    
    return anchor, sip_zone, garp_zone, in_sip or in_garp or in_deep

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Standardize column naming check (handles common screener exports)
    # Expected columns or mapping fallback
    col_map = {c.lower(): c for c in df.columns}
    
    required_cols = ["ticker", "name", "cmp", "roce", "roe", "debt_equity", "fcf", "eps", "shares"]
    
    # Process each row
    results = []
    for idx, row in df.iterrows():
        # Fallback helpers
        ticker = str(row.get("Ticker", row.get("Symbol", f"STOCK_{idx}")))
        name = str(row.get("Name", row.get("Company", ticker)))
        cmp = float(row.get("CMP", row.get("Current Price", row.get("Price", 100.0))))
        roce = float(row.get("ROCE", row.get("Return on Capital Employed", 15.0)))
        roe = float(row.get("ROE", row.get("Return on Equity", 12.0)))
        de = float(row.get("Debt_Equity", row.get("Debt to Equity", 0.2)))
        fcf = float(row.get("FCF", row.get("Free Cash Flow", row.get("Net Income", 1000.0))))
        shares = float(row.get("Shares", row.get("No. of Equity Shares", row.get("Shares Outstanding", 10.0))))
        eps = float(row.get("EPS", row.get("Earnings Per Share", 10.0)))
        
        # 1. Moat Grade
        moat = calculate_moat_grade(roce, roe, de)
        
        # 2. Implied Growth (Reverse DCF)
        implied_g1 = solve_reverse_dcf(cmp, fcf, default_wacc / 100.0, default_term_g / 100.0, shares)
        
        # 3. Intrinsic Value Estimation (Simplified Graham + DCF proxy)
        dcf_iv = max(cmp * 0.8, eps * 15 * (1 + implied_g1/100))
        graham_val = math.sqrt(22.5 * max(1.0, eps) * max(1.0, eps * 1.1)) if eps > 0 else cmp
        
        anchor, sip_z, garp_z, is_in_zone = compute_buy_zones(cmp, dcf_iv, graham_val, moat)
        
        # 4. Verdict & Category
        if moat == "A+" and implied_g1 <= 15.0:
            verdict = "💎 ELITE COMPOUNDING BUY"
        elif implied_g1 <= 12.0:
            verdict = "🟢 ATTRACTIVE GARP"
        elif implied_g1 <= 20.0:
            verdict = "🟡 FAIRLY PRICED"
        else:
            verdict = "🔴 PRICED FOR PERFECTION / AVOID"
            
        results.append({
            "Ticker": ticker,
            "Company": name,
            "Moat Grade": moat,
            "CMP (₹)": round(cmp, 2),
            "ROCE (%)": round(roce, 2),
            "ROE (%)": round(roe, 2),
            "Debt/Equity": round(de, 2),
            "Implied Growth ($g_1$ %)": implied_g1,
            "Valuation Anchor (₹)": round(anchor, 2),
            "SIP Zone Start (₹)": round(sip_z[0], 2),
            "SIP Zone End (₹)": round(sip_z[1], 2),
            "In Active Buy Zone": "✅ Yes" if is_in_zone else "❌ No",
            "Institutional Verdict": verdict
        })
        
    res_df = pd.DataFrame(results)
    
    # --- APPLY SIDEBAR FILTERS ---
    filtered_df = res_df[
        (res_df["ROCE (%)"] >= min_roce) &
        (res_df["Debt/Equity"] <= max_de) &
        (res_df["Moat Grade"].isin(selected_moats))
    ]
    
    st.subheader(f"📊 Screening Results Summary ({len(filtered_df)} of {len(res_df)} stocks matched)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Universe", len(res_df))
    col2.metric("Matching Filters", len(filtered_df))
    col3.metric("A+ Moat Count", len(res_df[res_df["Moat Grade"] == "A+"]))
    col4.metric("In Buy Zone", len(filtered_df[filtered_df["In Active Buy Zone"] == "✅ Yes"]))
    
    # Styled Dataframe
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Institutional Shortlist (CSV)",
        data=csv_bytes,
        file_name="institutional_screener_shortlist.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Please upload your stock universe CSV file in the sidebar to begin batch-screening.")
    
    with st.expander("📌 View Required CSV Format Guidelines"):
        st.markdown(""" Your uploaded CSV file should ideally contain the following columns (case-insensitive): * `Ticker` or `Symbol` (e.g., `RELIANCE`, `TCS`) * `Name` or `Company` (e.g., `Reliance Industries Ltd`) * `CMP` or `Current Price` (Current market price in ₹) * `ROCE` (%) * `ROE` (%) * `Debt_Equity` (Debt to Equity ratio) * `FCF` or `Net Income` (Free cash flow in ₹) * `EPS` (Earnings per share) * `Shares` (Total shares outstanding in crores or absolute units) """)