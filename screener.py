import re
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Multi-Horizon Stock Screener", layout="wide")
st.title("🛡️ Multi-Horizon Stock Screener: Short, Medium & Long Term")
st.caption("Business quality + valuation + growth + price trend + ownership, scored 0-100 and ranked per holding period. "
           "Educational tool, not investment advice.")

# ───────────────────────── SIDEBAR ─────────────────────────
st.sidebar.header("1. Upload tickers")
ticker_file = st.sidebar.file_uploader("NSE symbols (.txt or .csv)", type=["txt", "csv"])

st.sidebar.header("2. Valuation assumptions")
wacc = st.sidebar.slider("Discount rate / WACC (%)", 8.0, 18.0, 12.0, 0.5) / 100
tg = st.sidebar.slider("Terminal growth (%)", 2.0, 6.0, 4.0, 0.25) / 100
tg = min(tg, wacc - 0.02)  # guard: WACC must exceed terminal growth

st.sidebar.header("3. Display filters")
min_score = st.sidebar.slider("Min horizon score", 40, 90, 60, 1)
max_de = st.sidebar.slider("Max Debt/Equity (non-financials)", 0.0, 3.0, 1.0, 0.1)
max_pe = st.sidebar.slider("Max trailing P/E", 5.0, 150.0, 60.0, 1.0)
skip_fin = st.sidebar.checkbox("Hide banks / NBFCs / insurers", False)
top_n = st.sidebar.slider("Show top N per list", 5, 100, 25, 5)
workers = st.sidebar.slider("Parallel downloads", 1, 8, 4)

# ───────────────────────── HELPERS ─────────────────────────
def lin(x, lo, hi, default=40.0):
    """Map x linearly to 0-100 between lo and hi. Missing data gets a slightly below-neutral default."""
    if x is None or not np.isfinite(x):
        return default
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)

def stmt(df, keys, col=0, default=np.nan):
    if df is None or df.empty or col >= df.shape[1]:
        return default
    for k in keys:
        if k in df.index:
            v = df.loc[k].iloc[col]
            if pd.notna(v):
                return float(v)
    return default

def cagr(df, keys):
    """CAGR across all available annual columns (yfinance: newest first, ~4 years)."""
    if df is None or df.empty:
        return np.nan
    for k in keys:
        if k in df.index:
            s = df.loc[k].dropna()
            if len(s) >= 3:
                new, old, n = float(s.iloc[0]), float(s.iloc[-1]), len(s) - 1
                if new > 0 and old > 0:
                    return (new / old) ** (1 / n) - 1
                return -0.05 if new < old else np.nan
    return np.nan

def dcf_ps(cf, g, w, t, n=10):
    """10-yr DCF per share; growth g for yrs 1-5, then fades linearly to terminal growth."""
    pv, c = 0.0, cf
    for y in range(1, n + 1):
        gy = g if y <= 5 else g + (t - g) * (y - 5) / 5
        c *= 1 + gy
        pv += c / (1 + w) ** y
    return pv + c * (1 + t) / (w - t) / (1 + w) ** n

def reverse_dcf(price, cf, w, t):
    """Growth (yrs 1-5) the market is already pricing in. NaN if cash flow is not positive."""
    if not cf or cf <= 0 or price <= 0:
        return np.nan
    lo, hi = -0.30, 0.80
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if dcf_ps(cf, mid, w, t) > price else (mid, hi)
    return round(mid * 100, 2)

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return float((100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1])

@st.cache_data(ttl=3600, show_spinner=False)
def nifty_weekly():
    h = yf.Ticker("^NSEI").history(period="3y")["Close"]
    h.index = h.index.tz_localize(None)
    return h.resample("W-FRI").last().pct_change().dropna()

def real_beta(close, fallback):
    try:
        c = close.copy()
        c.index = c.index.tz_localize(None)
        w = c.resample("W-FRI").last().pct_change().dropna()
        j = pd.concat([w, nifty_weekly()], axis=1, join="inner").dropna()
        if len(j) >= 52:
            return round(float(j.cov().iloc[0, 1] / j.iloc[:, 1].var()), 2)
    except Exception:
        pass
    return round(float(fallback), 2) if fallback else np.nan

def trend(vals, thr=0.25):
    if not vals or len(vals) < 2:
        return "N/A", 0.0
    w = vals[-5:]
    d = round(w[-1] - w[0], 2)
    icon = "📈" if d > thr else ("📉" if d < -thr else "➖")
    return f"{icon} {w[-1]:.1f}% ({d:+.2f})", d

def holdings(sym):
    """Scrape 5-quarter shareholding from Screener.in. Returns text + numeric deltas."""
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"}
    for mode in ("consolidated/", ""):
        try:
            r = requests.get(f"https://www.screener.in/company/{sym}/{mode}", headers=hdr, timeout=8)
            if r.status_code != 200:
                continue
            sec = BeautifulSoup(r.text, "lxml").find("section", id="shareholding")
            tbl = sec.find("table", class_="data-table") if sec else None
            if not tbl:
                continue
            rows = {}
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) < 3:
                    continue
                name = re.sub(r"[\+\-\s]+$", "", cells[0]).lower()
                nums = []
                for c in cells[1:]:
                    try:
                        nums.append(float(c.replace("%", "").replace(",", "")))
                    except ValueError:
                        pass
                if nums:
                    rows[name] = nums
            pick = lambda *ks: next((v for n, v in rows.items() if any(k in n for k in ks)), [])
            prom, fii, dii, pub = pick("promoter"), pick("fii", "foreign"), pick("dii", "domestic"), pick("public")
            inst = [round(a + b, 2) for a, b in zip(fii, dii)] if fii and dii and len(fii) == len(dii) else (fii or dii)
            if prom or inst:
                (pt, pd_), (it, id_), (rt, rd_) = trend(prom), trend(inst), trend(pub)
                return dict(prom=pt, inst=it, retail=rt, prom_d=pd_, inst_d=id_, retail_d=rd_,
                            prom_now=prom[-1] if prom else np.nan)
        except Exception:
            continue
    return dict(prom="N/A", inst="N/A", retail="N/A", prom_d=0.0, inst_d=0.0, retail_d=0.0, prom_now=np.nan)

# ───────────────────────── CORE ANALYSIS ─────────────────────────
def analyze(sym):
    try:
        t = yf.Ticker(sym + ".NS")
        info = t.info or {}
        hist = t.history(period="3y", auto_adjust=True)
        if hist.empty or len(hist) < 220:
            return {"_err": "Insufficient price history (<220 sessions)"}
        close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
        cmp = float(close.iloc[-1])

        inc, bal, cfs = t.financials, t.balance_sheet, t.cashflow
        sector = info.get("sector", "N/A")
        is_fin = sector == "Financial Services"
        shares = info.get("sharesOutstanding") or np.nan
        eps = info.get("trailingEps") or np.nan
        pe = info.get("trailingPE") or (cmp / eps if eps and eps > 0 else np.nan)

        rev = stmt(inc, ["Total Revenue", "Operating Revenue"])
        ebit = stmt(inc, ["EBIT", "Operating Income"])
        ni = stmt(inc, ["Net Income", "Net Income Common Stockholders"])
        if not np.isfinite(ni):
            ni = info.get("netIncomeToCommon", np.nan)
        interest = abs(stmt(inc, ["Interest Expense", "Interest Expense Non Operating"], default=0.0))
        equity = stmt(bal, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        debt = stmt(bal, ["Total Debt"], default=info.get("totalDebt", np.nan))
        assets = stmt(bal, ["Total Assets"])
        cur_liab = stmt(bal, ["Current Liabilities", "Total Current Liabilities"])
        cap_emp = assets - cur_liab if np.isfinite(assets) and np.isfinite(cur_liab) and assets > cur_liab else equity + (debt if np.isfinite(debt) else 0)

        ocf = stmt(cfs, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        fcf = stmt(cfs, ["Free Cash Flow"])
        if not np.isfinite(fcf):
            capex = stmt(cfs, ["Capital Expenditure", "Purchase Of PPE"], default=0.0)
            fcf = ocf - abs(capex) if np.isfinite(ocf) else info.get("freeCashflow", np.nan)

        roe = ni / equity * 100 if np.isfinite(ni) and equity and equity > 0 else np.nan
        roce = (roe if is_fin else (ebit / cap_emp * 100 if np.isfinite(ebit) and cap_emp > 0 else np.nan))
        de = np.nan if is_fin else (debt / equity if np.isfinite(debt) and equity and equity > 0 else np.nan)
        opm = ebit / rev * 100 if np.isfinite(ebit) and np.isfinite(rev) and rev > 0 else np.nan
        icr = np.nan if is_fin else (ebit / interest if interest > 0 and np.isfinite(ebit) else (99.0 if np.isfinite(ebit) else np.nan))
        fcf_conv = fcf / ni if np.isfinite(fcf) and np.isfinite(ni) and ni > 0 else np.nan
        rev_g, ni_g = cagr(inc, ["Total Revenue"]), cagr(inc, ["Net Income", "Net Income Common Stockholders"])
        bvps = equity / shares if np.isfinite(equity) and np.isfinite(shares) and shares > 0 else np.nan

        # ── Profit-making trend (annual consistency + latest quarter momentum) ──
        q_inc = t.quarterly_financials
        qni = []
        if q_inc is not None and not q_inc.empty:
            for k in ("Net Income", "Net Income Common Stockholders"):
                if k in q_inc.index:
                    qni = [float(x) for x in q_inc.loc[k].dropna().tolist()]
                    break
        q_yoy = (qni[0] / qni[4] - 1) * 100 if len(qni) >= 5 and qni[4] > 0 else np.nan
        q_up = float(sum(1 for a, b in zip(qni[:3], qni[1:4]) if a > b)) if len(qni) >= 4 else np.nan
        ni_s = inc.loc["Net Income"].dropna().tolist() if inc is not None and not inc.empty and "Net Income" in inc.index else []
        yrs_n = len(ni_s) - 1
        yrs_up = sum(1 for a, b in zip(ni_s[:-1], ni_s[1:]) if a > b) if yrs_n > 0 else 0
        ptr = (0.5 * lin(yrs_up / yrs_n * 100, 0, 100) if yrs_n > 0 else 0.5 * 40) + 0.3 * lin(q_yoy, -10, 30) + 0.2 * lin(q_up, 0, 3)
        ptr_txt = f"Q YoY {q_yoy:+.0f}% | {yrs_up}/{yrs_n} yrs up" if np.isfinite(q_yoy) else f"{yrs_up}/{max(yrs_n,0)} yrs up"

        # ── Technicals ──
        dma50, dma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        rsi14 = rsi(close)
        ret3m = (cmp / close.iloc[-63] - 1) * 100
        ret6m = (cmp / close.iloc[-126] - 1) * 100
        off_high = (1 - cmp / close.iloc[-252:].max()) * 100
        volx = float(vol.rolling(20).mean().iloc[-1] / max(vol.rolling(100).mean().iloc[-1], 1))
        turnover_cr = float((close * vol).rolling(20).mean().iloc[-1] / 1e7)
        beta = real_beta(close, info.get("beta"))

        # ── Ownership ──
        h = holdings(sym)

        # ── QUALITY (business) score ──
        q = (0.20 * lin(roce, 8, 25) + 0.15 * lin(roe, 8, 22) + 0.10 * lin(opm, 5, 25)
             + 0.15 * lin(rev_g * 100 if np.isfinite(rev_g) else np.nan, 0, 18)
             + 0.15 * lin(ni_g * 100 if np.isfinite(ni_g) else np.nan, 0, 20)
             + 0.10 * (60.0 if is_fin else 100 - lin(de, 0, 1.5, 50))
             + 0.05 * (60.0 if is_fin else lin(icr, 1.5, 8))
             + 0.10 * (60.0 if is_fin else lin(fcf_conv, 0.3, 1.0)))
        moat = "A+" if q >= 80 else "A" if q >= 68 else "B" if q >= 55 else "C" if q >= 40 else "D"

        # ── VALUATION: Graham, DCF, fair-P/E (median = anchor) ──
        g_base = np.nanmean([x for x in [ni_g, rev_g] if np.isfinite(x)] or [0.08])
        g_use = float(np.clip(g_base * 0.8, 0.05, 0.18))  # 20% haircut, capped
        graham = math.sqrt(22.5 * eps * bvps) if np.isfinite(eps) and np.isfinite(bvps) and eps > 0 and bvps > 0 else np.nan
        dcf_iv = (dcf_ps(fcf / shares, g_use, wacc, tg)
                  if not is_fin and np.isfinite(fcf) and fcf > 0 and np.isfinite(shares) else np.nan)
        fair_pe = 12 + q * 0.12  # 12x (poor) to 24x (excellent business)
        pe_iv = eps * fair_pe if np.isfinite(eps) and eps > 0 else np.nan
        ivs = [v for v in (graham, dcf_iv, pe_iv) if np.isfinite(v) and v > 0]
        anchor = float(np.median(ivs)) if ivs else np.nan
        mos = (anchor - cmp) / anchor if np.isfinite(anchor) else np.nan  # margin of safety
        peg = pe / (g_use * 100) if np.isfinite(pe) and pe > 0 else np.nan
        implied_g = reverse_dcf(cmp, fcf / shares, wacc, tg) if np.isfinite(fcf) and np.isfinite(shares) else np.nan
        v = 0.7 * lin(mos, -0.30, 0.40) + 0.3 * (100 - lin(peg, 0.5, 3.0, 50))

        # ── GROWTH / TECHNICAL / OWNERSHIP scores ──
        g = (0.30 * lin(ni_g * 100 if np.isfinite(ni_g) else np.nan, 0, 25) + 0.20 * lin(rev_g * 100 if np.isfinite(rev_g) else np.nan, 0, 20) + 0.50 * ptr)
        tech = ((25 if cmp > dma50 else 0) + (25 if dma50 > dma200 else 0)
                + (20 if 45 <= rsi14 <= 68 else (8 if 68 < rsi14 <= 75 else 0))
                + 0.15 * lin(ret3m, -5, 20) + 0.15 * (100 - lin(off_high, 0, 25))
                ) + (5 if volx > 1.15 else 0)
        tech = min(100.0, tech)
        # promoters/institutions accumulating = good; retail rising while smart money is not = late-stage crowding
        own = 0.35 * lin(h["prom_d"], -2, 2, 50) + 0.45 * lin(h["inst_d"], -2, 3, 50) + 0.20 * (100 - lin(h["retail_d"], -3, 3, 50))

        # ── HORIZON SCORES ──
        long_s = 0.50 * q + 0.30 * v + 0.20 * g
        med_s = 0.30 * q + 0.25 * g + 0.20 * v + 0.15 * tech + 0.10 * own
        short_s = 0.55 * tech + 0.25 * q + 0.10 * v + 0.10 * own

        # ── ELIGIBILITY GATES (hard rules, a high score alone is not enough) ──
        fin_ok = lambda c: c if not is_fin else True
        long_ok = (q >= 60 and (is_fin or (np.isfinite(de) and de <= 1.0)) and (is_fin or (np.isfinite(fcf) and fcf > 0))
                   and (not np.isfinite(mos) or mos >= -0.10) and (not np.isfinite(ni_g) or ni_g > 0))
        med_ok = q >= 50 and ptr >= 40 and cmp > dma200 and g >= 40 and (not np.isfinite(mos) or mos >= -0.25)
        short_ok = (cmp > dma50 and 40 <= rsi14 <= 72 and off_high <= 20 and q >= 40 and turnover_cr >= 2)

        # ── Risk flags ──
        flags = []
        if np.isfinite(fcf) and fcf <= 0: flags.append("Negative FCF")
        if np.isfinite(de) and de > 1: flags.append("High debt")
        if np.isfinite(icr) and icr < 2: flags.append("Weak interest cover")
        if np.isfinite(ni_g) and ni_g < 0: flags.append("Profits shrinking")
        if h["prom_d"] < -1: flags.append("Promoter selling")
        if np.isfinite(roce) and roce < 10: flags.append("Low ROCE")
        if cmp < dma200: flags.append("Below 200DMA")
        if np.isfinite(q_yoy) and q_yoy < 0: flags.append("Latest-quarter profit falling")
        if h["retail_d"] > 1 and h["inst_d"] < -0.5: flags.append("Retail buying / institutions exiting")
        if np.isfinite(implied_g) and implied_g > 20: flags.append("Priced for perfection")

        if q >= 55 and np.isfinite(mos) and mos >= -0.15:
            style = "🚀 Aggressive Compounder" if beta >= 1.05 else "🛡️ Defensive Anchor"
        elif q >= 55:
            style = "🔴 Quality but Expensive"
        elif np.isfinite(beta) and beta > 1.1:
            style = "⚠️ High-Beta Speculative"
        else:
            style = "🛑 Weak Business / Value Trap"

        # ── TARGETS & STOP-LOSSES (volatility + structure based) ──
        lo20, lo60, lo252 = float(low.iloc[-20:].min()), float(low.iloc[-60:].min()), float(low.iloc[-252:].min())
        hi252 = float(high.iloc[-252:].max())
        # Short: tighter of (2xATR) and (below 20-day swing low), risk kept within 3-10%; target = 2R, capped at 52w-high resistance
        s_stop = min(max(cmp - 2 * atr, lo20 - 0.25 * atr), cmp * 0.97)
        s_stop = max(s_stop, cmp * 0.90)
        s_risk = cmp - s_stop
        s_tgt = cmp + 2 * s_risk
        if cmp * 1.02 < hi252 < s_tgt:
            s_tgt = max(hi252, cmp + 1.5 * s_risk)
        # Medium: below 60-day swing low / 200-DMA / 4xATR, risk 5-20%; target = fair value or 2R, max +40%
        m_stop = min(max(lo60 - 0.5 * atr, cmp - 4 * atr, dma200 * 0.97), cmp * 0.95)
        m_stop = max(m_stop, cmp * 0.80)
        m_risk = cmp - m_stop
        m_tgt = min(max(anchor if np.isfinite(anchor) else 0.0, cmp + 2 * m_risk), cmp * 1.40)
        # Long: buy below 10% discount to fair value; 3-yr target = EPS grown at haircut growth x fair P/E; thesis-review stop
        l_buy = anchor * 0.90 if np.isfinite(anchor) else np.nan
        l_tgt = eps * (1 + g_use) ** 3 * fair_pe if np.isfinite(eps) and eps > 0 else anchor
        l_tgt = min(l_tgt, cmp * 2.2) if np.isfinite(l_tgt) else np.nan
        l_stop = max(cmp * 0.75, lo252 * 0.95)
        l_cagr = ((l_tgt / cmp) ** (1 / 3) - 1) * 100 if np.isfinite(l_tgt) and l_tgt > 0 else np.nan
        analyst = info.get("targetMeanPrice") or np.nan
        r = lambda x, n=2: round(float(x), n) if np.isfinite(x) else np.nan
        return {
            "Ticker": sym, "Price Date": str(close.index[-1].date()), "Company": info.get("longName", sym), "Sector": sector, "Style": style, "Beta": beta,
            "Moat": moat, "Quality": r(q, 1), "Valuation": r(v, 1), "Growth": r(g, 1), "Trend": r(tech, 1),
            "Long Score": r(long_s, 1), "Medium Score": r(med_s, 1), "Short Score": r(short_s, 1),
            "Long OK": bool(long_ok), "Medium OK": bool(med_ok), "Short OK": bool(short_ok),
            "CMP": r(cmp), "P/E": r(pe), "PEG": r(peg), "ROCE %": r(roce), "ROE %": r(roe), "D/E": r(de),
            "Op Margin %": r(opm), "Rev CAGR %": r(rev_g * 100 if np.isfinite(rev_g) else np.nan),
            "Profit CAGR %": r(ni_g * 100 if np.isfinite(ni_g) else np.nan), "FCF/PAT": r(fcf_conv),
            "Graham IV": r(graham), "DCF IV": r(dcf_iv), "P/E IV": r(pe_iv), "Anchor IV": r(anchor),
            "Margin of Safety %": r(mos * 100 if np.isfinite(mos) else np.nan, 1), "Implied g %": implied_g,
            "RSI": r(rsi14, 1), "3M Ret %": r(ret3m, 1), "6M Ret %": r(ret6m, 1), "Off 52W High %": r(off_high, 1),
            "Vol Surge x": r(volx), "Turnover ₹Cr": r(turnover_cr, 1),
            "Short Stop": r(s_stop), "Short Target": r(s_tgt), "Short Risk %": r(s_risk / cmp * 100, 1), "Short Upside %": r((s_tgt / cmp - 1) * 100, 1),
            "Medium Stop": r(m_stop), "Medium Target": r(m_tgt), "Medium Risk %": r(m_risk / cmp * 100, 1), "Medium Upside %": r((m_tgt / cmp - 1) * 100, 1),
            "Long Buy Below": r(l_buy), "Long Target (3y)": r(l_tgt), "Long 3y CAGR %": r(l_cagr, 1), "Long Review Stop": r(l_stop),
            "Analyst Target": r(analyst), "Profit Trend": ptr_txt, "Q Profit YoY %": r(q_yoy, 1),
            "Promoter": h["prom"], "Institutions": h["inst"], "Retail": h["retail"], "Promoter %": r(h["prom_now"], 1),
            "Red Flags": ", ".join(flags) if flags else "None",
        }
    except Exception as e:
        return {"_err": f"{type(e).__name__}: {e}"}

# ───────────────────────── LOAD TICKERS ─────────────────────────
def load_tickers(f):
    if f.name.endswith(".csv"):
        df = pd.read_csv(f)
        cand = [c for c in df.columns if any(k in c.lower() for k in ("ticker", "symbol", "name"))]
        vals = df[cand[0]] if cand else df.iloc[:, 0]
    else:
        vals = pd.Series(f.getvalue().decode("utf-8").splitlines())
    out = [re.sub(r"\.(NS|BO)$", "", str(x).strip().upper()) for x in vals.dropna() if str(x).strip()]
    return list(dict.fromkeys(out))  # de-duplicate, keep order

tab_long, tab_med, tab_short, tab_all, tab_track, tab_guide, tab_logic = st.tabs(
    ["🌳 Long Term (3-5y+)", "📈 Medium Term (6-18m)", "⚡ Short Term (1-3m)", "🗂️ All Data",
     "🎯 Track Record", "📖 Guide & Glossary", "🧠 Methodology"])

if ticker_file is None:
    st.info("👈 Upload your ticker file in the sidebar to begin.")
else:
    tickers = load_tickers(ticker_file)
    st.write(f"📥 Loaded **{len(tickers)}** unique symbols.")
    if st.button("🚀 Analyze & Rank", type="primary"):
        res, errs, bar, txt = [], {}, st.progress(0), st.empty()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(analyze, s): s for s in tickers}
            for i, fu in enumerate(as_completed(futs), 1):
                s, out = futs[fu], fu.result()
                (errs.__setitem__(s, out["_err"]) if "_err" in out else res.append(out))
                bar.progress(i / len(tickers))
                txt.text(f"Processed {i}/{len(tickers)}: {s}")
        txt.text("✅ Done")
        st.session_state["df"], st.session_state["errs"] = pd.DataFrame(res), errs

    df = st.session_state.get("df")
    if df is not None and not df.empty:
        base = df[(df["P/E"].isna() | ((df["P/E"] > 0) & (df["P/E"] <= max_pe)))
                  & (df["D/E"].isna() | (df["D/E"] <= max_de))]
        if skip_fin:
            base = base[base["Sector"] != "Financial Services"]

        snap = []
        SNAP = {"long": ("Long Review Stop", "Long Target (3y)", 1095),
                "medium": ("Medium Stop", "Medium Target", 540),
                "short": ("Short Stop", "Short Target", 90)}

        def show(tab, name, score, ok, cols, blurb):
            with tab:
                st.markdown(blurb)
                d = base[base[ok] & (base[score] >= min_score)].sort_values(score, ascending=False).head(top_n)
                if not d.empty:
                    sc, tc, days = SNAP[name]
                    x = d[["Ticker", "Company", score, "Style", "Moat", "Quality", "Price Date", "CMP", sc, tc]].copy()
                    x.columns = ["Ticker", "Company", "Score", "Style", "Moat", "Quality", "Analysis Date", "Entry", "Stop", "Target"]
                    x.insert(2, "Horizon", name.capitalize())
                    x["Horizon Days"] = days
                    snap.append(x)
                st.subheader(f"{len(d)} stocks qualify")
                if d.empty:
                    st.warning("None pass the gates. Lower 'Min horizon score' or relax filters.")
                else:
                    st.dataframe(d[["Ticker", "Company", score, "Style", "Moat"] + cols + ["Red Flags"]],
                                 hide_index=True, use_container_width=True,
                                 column_config={"Ticker": st.column_config.TextColumn(pinned=True),
                                                "Company": st.column_config.TextColumn(pinned=True)})
                    st.download_button("📥 CSV", d.to_csv(index=False).encode(), f"{name}_list.csv", "text/csv", key=name)

        show(tab_long, "long", "Long Score", "Long OK",
             ["Quality", "Valuation", "ROCE %", "ROE %", "D/E", "Profit CAGR %", "FCF/PAT", "P/E", "PEG",
              "Profit Trend", "CMP", "Anchor IV", "Margin of Safety %", "Implied g %", "Long Buy Below", "Long Target (3y)", "Long 3y CAGR %", "Long Review Stop", "Analyst Target", "Promoter", "Institutions", "Retail"],
             "**Wide-moat compounders at sensible prices.** Business quality 50%, valuation 30%, growth 20%. "
             "Gates: Quality ≥ 60, D/E ≤ 1, positive FCF, profits growing, not >10% above fair value. Accumulate below *Buy Below* (10% discount to fair value).")
        show(tab_med, "medium", "Medium Score", "Medium OK",
             ["Quality", "Growth", "Valuation", "Trend", "Profit CAGR %", "Rev CAGR %", "6M Ret %", "CMP",
              "Profit Trend", "Medium Stop", "Medium Target", "Medium Risk %", "Medium Upside %", "Analyst Target", "Margin of Safety %", "Promoter", "Institutions", "Retail"],
             "**Good businesses in an uptrend with earnings momentum.** Quality 30%, growth 25%, valuation 20%, trend 15%, ownership 10%. "
             "Gates: Quality ≥ 50, price above 200-DMA, growth score ≥ 40, not >25% above fair value.")
        show(tab_short, "short", "Short Score", "Short OK",
             ["Trend", "Quality", "RSI", "3M Ret %", "Off 52W High %", "Vol Surge x", "Turnover ₹Cr", "CMP",
              "Profit Trend", "Short Stop", "Short Target", "Short Risk %", "Short Upside %", "Promoter", "Institutions", "Retail"],
             "**Momentum setups with a quality floor.** Trend 55%, quality 25%, valuation 10%, ownership 10%. "
             "Gates: price > 50-DMA, RSI 40-72 (not overheated), within 20% of 52-week high, turnover ≥ ₹2 Cr/day, Quality ≥ 40. "
             "Stop = tighter of CMP − 2×ATR or below 20-day swing low; Target = 2× the risk, capped at 52-week-high resistance.")
        with tab_track:
            if snap:
                sdf = pd.concat(snap, ignore_index=True)
                st.success(f"Snapshot ready: {len(sdf)} picks across all horizons. Download it NOW and keep it safe.")
                st.download_button("📥 Download Snapshot (needed for verification later)", sdf.to_csv(index=False).encode(),
                                   f"snapshot_{sdf['Analysis Date'].max()}.csv", "text/csv", key="snap_dl")
            else:
                st.info("No picks qualify with the current filters, so there is nothing to snapshot.")
        with tab_all:
            st.dataframe(base, hide_index=True, use_container_width=True)
            st.download_button("📥 All data CSV", base.to_csv(index=False).encode(), "all_data.csv", "text/csv")
    elif df is not None:
        st.error("No stock could be analyzed. See failures below.")
    if st.session_state.get("errs"):
        with st.expander(f"⚠️ {len(st.session_state['errs'])} tickers failed"):
            st.json(st.session_state["errs"])

with tab_logic:
    st.markdown("""
### How stocks are chosen
Every stock gets four 0-100 sub-scores, which are combined differently for each holding period. A stock must also pass **hard gates**; a high score cannot override a bad balance sheet or a broken trend.

**1. Quality (is the business good?)** ROCE (20%), ROE (15%), operating margin (10%), revenue CAGR (15%), profit CAGR (15%), low debt (10%), interest cover (5%), cash conversion FCF/PAT (10%). Moat grade: A+ ≥ 80, A ≥ 68, B ≥ 55, C ≥ 40, else D. Banks/NBFCs are scored on ROE, with debt-based metrics set neutral.

**2. Valuation (is the price fair?)** Three intrinsic values (Graham number, 10-year DCF with growth fading to terminal growth, quality-adjusted fair P/E of 12x-24x). The median is the *anchor*. Margin of safety = (anchor − price) / anchor. PEG is blended in. A reverse DCF shows the growth the market already expects.

**3. Growth** Profit and revenue CAGR over available annual statements.

**4. Trend / Timing** 50/200-DMA structure, RSI(14), 3-month return, distance from the 52-week high, volume surge.

**5. Ownership** 5-quarter promoter and institutional (FII+DII) holding change from Screener.in.

**Profit trend** (inside Growth): share of years with rising net profit (50%), latest-quarter YoY profit growth (30%), and how many of the last 3 quarters rose sequentially (20%).

**Institution / retail logic:** promoters and institutions (FII+DII) accumulating is positive; *retail share rising while institutions fall* is treated as late-stage crowding and penalised.

**Target & stop-loss (derived from volatility and price structure, not guesses)**
- *Short:* stop = tighter of 2×ATR(14) or below the 20-day swing low (risk kept 3-10%); target = 2× risk, capped at 52-week-high resistance.
- *Medium:* stop = highest of 60-day swing low − 0.5 ATR, 4×ATR, or 3% under the 200-DMA (risk 5-20%); target = fair value or 2× risk, max +40%.
- *Long:* buy below = 90% of fair value; 3-year target = EPS grown at a haircut growth rate × quality-based fair P/E (shown with implied CAGR); review stop = 25% below price or just under the 52-week low. Long stops are thesis-review levels, not tight exits.
- *Analyst Target* (Yahoo consensus) is shown only as a cross-check.

**Beta** is computed from 3 years of weekly returns against the Nifty 50 (not Yahoo's raw field). Beta ≥ 1.05 with quality and fair price → *Aggressive Compounder*; below → *Defensive Anchor*.

| Horizon | Weights | Key gates |
|---|---|---|
| Long | Q 50 · V 30 · G 20 | Q ≥ 60, D/E ≤ 1, FCF > 0, profits growing |
| Medium | Q 30 · G 25 · V 20 · T 15 · O 10 | Q ≥ 50, price > 200-DMA |
| Short | T 55 · Q 25 · V 10 · O 10 | price > 50-DMA, RSI 40-72, liquid |

**Limits:** Yahoo gives only ~4 years of statements, data for Indian small caps can be incomplete, and the model cannot see management integrity, regulation, or news. Always read the annual report before buying.
""")


# ───────────────────────── TRACK RECORD (outcome verification) ─────────────────────────
OUT_COLS = ["Status", "Exit/Last Price", "Return %", "Nifty %", "Alpha %", "Max Drawdown %", "Max Run-up %", "Days"]

def check_outcome(r, nifty):
    """Replay daily candles after the analysis date. First of target/stop touched wins.
    Same-day touch of both = stop assumed first (conservative). Gaps exit at the open price."""
    try:
        start = pd.to_datetime(r["Analysis Date"])
        h = yf.Ticker(f"{r['Ticker']}.NS").history(start=(start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True)
        h.index = h.index.tz_localize(None)
        h = h[h.index > start]
        if h.empty:
            return {"Status": "⏳ Too early"}
        entry, stop, tgt = float(r["Entry"]), float(r["Stop"]), float(r["Target"])
        ht, hs = h.index[h["High"] >= tgt], h.index[h["Low"] <= stop]
        t0, s0 = (ht[0] if len(ht) else None), (hs[0] if len(hs) else None)
        if s0 is not None and (t0 is None or s0 <= t0):
            status, d, ex = "🛑 Stop hit", s0, min(stop, float(h.loc[s0, "Open"]))
        elif t0 is not None:
            status, d, ex = "🎯 Target hit", t0, max(tgt, float(h.loc[t0, "Open"]))
        else:
            d, ex = h.index[-1], float(h["Close"].iloc[-1])
            status = "⌛ Expired (no hit)" if (d - start).days >= int(r["Horizon Days"]) else "⏳ Open"
        seg = h.loc[:d]
        n0, n1 = nifty.asof(start), nifty.asof(d)
        ret = (ex / entry - 1) * 100
        return {"Status": status, "Exit/Last Price": round(ex, 2), "Return %": round(ret, 2),
                "Nifty %": round((n1 / n0 - 1) * 100, 2), "Alpha %": round(ret - (n1 / n0 - 1) * 100, 2),
                "Max Drawdown %": round((seg["Low"].min() / entry - 1) * 100, 2),
                "Max Run-up %": round((seg["High"].max() / entry - 1) * 100, 2), "Days": (d - start).days}
    except Exception as e:
        return {"Status": f"❌ Error: {type(e).__name__}: {str(e)[:80]}"}

with tab_track:
    st.divider()
    st.subheader("🔍 Verify a past snapshot")
    st.caption("Upload the snapshot CSV you downloaded earlier (e.g., after 3-6 months). The app replays every daily candle since "
               "the analysis date, checks whether Target or Stop was touched first, and compares with the Nifty 50.")
    up = st.file_uploader("Snapshot CSV", type=["csv"], key="snap_up")
    if up is not None:
        old = pd.read_csv(up)
        need = {"Ticker", "Horizon", "Analysis Date", "Entry", "Stop", "Target", "Horizon Days"}
        if not need.issubset(old.columns):
            st.error(f"Wrong file. Required columns: {sorted(need)}. Use the 'Download Snapshot' button, not the per-tab CSVs.")
        elif st.button("✅ Check outcomes", type="primary"):
            nf = yf.Ticker("^NSEI").history(start=(pd.to_datetime(old["Analysis Date"]).min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))["Close"]
            nf.index = nf.index.tz_localize(None)
            rows, bar = [], st.progress(0)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(check_outcome, r, nf): i for i, r in old.iterrows()}
                for k, fu in enumerate(as_completed(futs), 1):
                    rows.append((futs[fu], fu.result()))
                    bar.progress(k / len(futs))
            out = pd.DataFrame([r for _, r in sorted(rows, key=lambda z: z[0])], index=old.index).reindex(columns=OUT_COLS)
            st.session_state["track"] = pd.concat([old[["Ticker", "Horizon", "Analysis Date", "Entry", "Stop", "Target"]], out], axis=1)

    tr = st.session_state.get("track")
    if tr is not None:
        st.subheader("Status overview")
        st.write(tr["Status"].value_counts().to_dict())
        done = tr.dropna(subset=["Return %"])
        if done.empty:
            st.info("No outcome could be computed yet. Usually this means there is no completed trading session after the "
                    "'Analysis Date' (weekend, holiday, or the market has not opened/closed yet). Come back after at least a "
                    "few sessions; a meaningful check needs weeks to months. If you see '❌ Error' rows, Yahoo may have "
                    "rate-limited you; lower 'Parallel downloads' to 2 and retry.")
        summ = []
        for hz, g in done.groupby("Horizon"):
            tgt_n, stp_n = g["Status"].str.contains("Target").sum(), g["Status"].str.contains("Stop").sum()
            summ.append({"Horizon": hz, "Picks": len(g), "🎯 Target %": round(tgt_n / len(g) * 100, 1),
                         "🛑 Stop %": round(stp_n / len(g) * 100, 1),
                         "Still open/expired %": round((len(g) - tgt_n - stp_n) / len(g) * 100, 1),
                         "Target ÷ (Target+Stop) %": round(tgt_n / (tgt_n + stp_n) * 100, 1) if tgt_n + stp_n else np.nan,
                         "Win rate % (ret>0)": round((g["Return %"] > 0).mean() * 100, 1),
                         "Avg return %": round(g["Return %"].mean(), 2), "Median return %": round(g["Return %"].median(), 2),
                         "Avg alpha vs Nifty %": round(g["Alpha %"].mean(), 2), "Avg max drawdown %": round(g["Max Drawdown %"].mean(), 2)})
        st.subheader("Scorecard by horizon")
        st.dataframe(pd.DataFrame(summ), hide_index=True, use_container_width=True)
        if len(done) < 30:
            st.warning("Fewer than 30 picks: treat these numbers as anecdotal, not proof. Repeat the process over several snapshots.")
        st.caption("Open positions are marked-to-market at the latest close. A medium/long pick checked after only 6 months is expected to be mostly 'Open'.")
        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.download_button("📥 Download verification report", tr.to_csv(index=False).encode(), "verification_report.csv", "text/csv")

# ───────────────────────── GUIDE ─────────────────────────
with tab_guide:
    st.markdown("""
### Glossary
| Metric | What it is | Why it matters |
|---|---|---|
| **3M Ret %** | Price change over the last ~63 trading days | Shows whether the market is currently rewarding the stock. Positive and moderate (0-25%) is healthy; very high means chasing |
| **Off 52W High %** | How far price is below its 52-week high | Near the high = strength and little overhead supply. Over 20% below = broken trend, so short-term picks are excluded |
| **Vol Surge x** | 20-day average volume ÷ 100-day average volume | Above ~1.15 means rising participation (often institutions) behind the move. Below 0.7 means fading interest |
| **Turnover ₹Cr** | 20-day average daily traded value (price × volume) | Liquidity check. Under ₹2 Cr/day means entry/exit slippage and stop-loss gaps, so short-term picks are excluded |
| **Rev CAGR %** | Compound annual revenue growth over the available years (~3) | Shows whether the business is actually expanding. Revenue growth with profit growth is a good sign. Profit growth without revenue growth is often cost-cutting or one-offs |

### How to use it
- Run after market close (after 3:30 pm IST) so the price is a clean closing price.
- Re-run: short list weekly, medium list monthly, long list quarterly (after results season).
- Download the **Snapshot** every time. It is your audit trail.
- Verify short picks after 3 months, medium after 6-18 months, long after 1-3 years.
- Never buy on a score alone: read the Red Flags column, the latest results and the annual report.
- Risk at most 1-2% of your capital per trade; size positions from (entry - stop).
- Don't move a stop lower, don't average down a short-term pick, don't ignore a stop because "fundamentals are good".
""")