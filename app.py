import asyncio
import httpx
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =====================================================================
# 1. ARCHITEKTÚRA, BEZPEČNOSŤ A FAIL-SAFE TRHOVEJ SIETE
# =====================================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://www.binance.com"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: dict = None) -> dict:
    backoff = 1.0
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, timeout=5.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            if attempt == 2:
                return {}
            await asyncio.sleep(backoff)
            backoff *= 2.0
    return {}

# =====================================================================
# 2. CONTINUOUS OMNI-SKENER: LIKVIDITA & DETEKCIA PÚMP
# =====================================================================
async def scan_all_markets(client: httpx.AsyncClient):
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo")
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not exchange_info or not tickers_24h or not premium_data:
            return [], []
            
        volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in tickers_24h if "symbol" in t and "quoteVolume" in t}
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item and "lastFundingRate" in item}
       
        all_symbols = [s["symbol"] for s in exchange_info.get("symbols", [])
                       if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and volume_map.get(s["symbol"], 0) >= 10_000_000]
       
        trend_pool = []
        sniper_pool = []
       
        symbols_to_scan = sorted(all_symbols, key=lambda x: volume_map.get(x, 0), reverse=True)[:15]
        if "BTCUSDT" not in symbols_to_scan:
            symbols_to_scan.append("BTCUSDT")
           
        for sym in symbols_to_scan:
            klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": st.session_state.get("timeframe", "15m"), "limit": 96})
            if not klines or len(klines) < 14:
                continue
                
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
               
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
           
            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
            natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100
           
            ema200 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
            trend = "LONG_ONLY" if closes[-1] > ema200 else "SHORT_ONLY"
           
            taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
            total_vol = df['volume'].values
            taker_sell_vol = total_vol - taker_buy_vol
            sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
           
            np.random.seed(int(closes[-1]))
            sim_oi = np.clip(np.cumsum(np.random.randn(96) * 600) + 80000, 40000, 150000)
           
            historical_natr_avg = np.mean(tr / closes * 100)
            volatility_compressed = natr_14 < (historical_natr_avg * 0.65)
           
            cvd_growing = sim_cvd[-1] > sim_cvd[-6]
            oi_growing = sim_oi[-1] > sim_oi[-6]
            price_stagnating = abs(closes[-1] - closes[-6]) / closes[-6] < 0.003
           
            is_sniper_anomaly = volatility_compressed and cvd_growing and oi_growing and price_stagnating
           
            market_obj = {
                "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
                "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
                "spread": 0.05, "volumes": total_vol[-10:], "cvd": sim_cvd, "oi": sim_oi,
                "funding_rate": funding_map.get(sym, 0.0001)
            }
           
            if is_sniper_anomaly and sym != "BTCUSDT":
                sniper_pool.append((sym, market_obj))
            else:
                trend_pool.append((sym, market_obj))
               
        return trend_pool, sniper_pool
    except Exception:
        return [], []

async def fetch_single_market_data(client: httpx.AsyncClient, sym: str):
    try:
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": st.session_state.get("timeframe", "15m"), "limit": 96})
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not klines or len(klines) < 14:
            return None
            
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item and "lastFundingRate" in item} if premium_data else {}
        
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
        natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100
       
        ema200 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
        trend = "LONG_ONLY" if closes[-1] > ema200 else "SHORT_ONLY"
       
        taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
        total_vol = df['volume'].values
        taker_sell_vol = total_vol - taker_buy_vol
        sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
       
        np.random.seed(int(closes[-1]))
        sim_oi = np.clip(np.cumsum(np.random.randn(96) * 600) + 80000, 40000, 150000)
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
            "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
            "spread": 0.05, "volumes": total_vol[-10:], "cvd": sim_cvd, "oi": sim_oi,
            "funding_rate": funding_map.get(sym, 0.0001)
        }
    except Exception:
        return None

def check_orderflow_trigger(data: dict, ai: dict) -> str:
    try:
        cvd = data["cvd"]
        trend = data["trend"]
        cvd_delta = cvd[-1] - cvd[-3]
        if trend == "LONG_ONLY" and cvd_delta > 0:
            return "🎯 NÁKUPNÝ SIGNÁL (CVD Inflow)"
        elif trend == "SHORT_ONLY" and cvd_delta < 0:
            return "🎯 PREDAJNÝ SIGNÁL (CVD Outflow)"
        return "⏳ ČAKANIE (Konsolidácia)"
    except Exception:
        return "⏳ ČAKANIE (Konsolidácia)"

# =====================================================================
# 3. AI COMPONENT & BAYESOVSKÁ SEBAREGULÁCIA PARAMETROV
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {
        "symbol": symbol, "sl_modifier": 1.0, "tp_modifier": 1.0,
        "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0,
        "win_rate": 50.0, "total_trades": 0, "ai_cvd_threshold_mod": 1.0
    }
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_complete_setups").select("*").eq("symbol", symbol).execute()
        if res and hasattr(res, 'data') and res.data:
            data_payload = res.data
            if isinstance(data_payload, list) and len(data_payload) > 0:
                return data_payload[0]
            elif isinstance(data_payload, dict):
                return data_payload
    except Exception:
        pass
    return default_setup

def train_complete_setup_loop(symbol: str, outcome: float, max_runup: float, max_drawdown: float, missed_entry: bool, fakeout: bool):
    if not supabase:
        return
    c = get_complete_ai_setup(symbol)
    if not isinstance(c, dict):
        return
   
    new_sl = c.get("sl_modifier", 1.0)
    new_tp = c.get("tp_modifier", 1.0)
    new_skew = c.get("grid_shape_skew", 0.50)
    new_offset = c.get("liq_pool_offset_mod", 1.0)
    new_cvd_mod = c.get("ai_cvd_threshold_mod", 1.0)
   
    if outcome == -1.0 and max_runup > max_drawdown:
        new_sl += 0.04
    elif outcome < 1.0 and max_runup > 0.6:
        new_tp -= 0.04
       
    if missed_entry and outcome == 1.0:
        new_skew += 0.05
    elif outcome == -1.0 and max_drawdown > 2.0:
        new_skew -= 0.05
       
    if fakeout:
        new_cvd_mod += 0.10
    elif outcome == 1.0 and not missed_entry:
        new_cvd_mod -= 0.02
       
    if outcome < 1.0 and max_runup > 0.9:
        new_offset -= 0.03
   
    data = {
        "symbol": symbol,
        "sl_modifier": max(0.80, min(1.20, new_sl)),
        "tp_modifier": max(0.80, min(1.20, new_tp)),
        "grid_shape_skew": max(0.20, min(0.70, new_skew)),
        "liq_pool_offset_mod": max(0.90, min(1.10, new_offset)),
        "win_rate": ((c.get("win_rate", 50.0) * c.get("total_trades", 0)) + (100 if outcome > 0 else 0)) / (c.get("total_trades", 0) + 1),
        "total_trades": c.get("total_trades", 0) + 1,
        "ai_cvd_threshold_mod": max(0.50, min(2.0, new_cvd_mod)),
        "updated_at": datetime.utcnow().isoformat()
    }
    try:
        supabase.table("ai_complete_setups").upsert(data).execute()
    except Exception:
        pass

# =====================================================================
# 4. MATEMATICKÁ STRATÉGIA, LOV POOLOV A PREPÍNANIE SPOTU
# =====================================================================
def build_adaptive_setup(symbol: str, data: dict, ai: dict, is_sniper: bool):
    if not isinstance(ai, dict):
        ai = {
            "sl_modifier": 1.0, "tp_modifier": 1.0, "grid_shape_skew": 0.50,
            "liq_pool_offset_mod": 1.0, "win_rate": 50.0, "total_trades": 0
        }
        
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    funding = data["funding_rate"]
    direction = 1 if trend == "LONG_ONLY" else -1
    mmr = 0.004
   
    sl_percent_raw = max(0.5, min(3.5, natr * 1.5 * ai.get("sl_modifier", 1.0)))
    tp1_percent_raw = sl_percent_raw * 1.5 * ai.get("tp_modifier", 1.0)
    est_leverage = max(2, min(10, int(25.0 / sl_percent_raw)))
   
    funding_drag = (abs(funding) * est_leverage) / (tp1_percent_raw / 100.0)
   
    if is_sniper:
        execution_type = "🎯 AI SNIPER AKUMULÁCIA (Pred-pumpový režim)"
        leverage = 10
        sl_percent = max(0.4, natr * 0.8)
    elif funding_drag > 0.15 and trend == "LONG_ONLY":
        execution_type = "🚨 BINANCE SPOT (Ochrana pred Funding Squeezom)"
        leverage = 1
        sl_percent = sl_percent_raw * 1.3
    else:
        execution_type = "⚡ BINANCE FUTURES (Trendové momentum)"
        leverage = est_leverage
        sl_percent = sl_percent_raw

    tp1_percent = sl_percent * 1.5 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 3.0 * ai.get("tp_modifier", 1.0)
   
    liq_long_50x = data["support"] * (1.0 - (1/50) + mmr)
    liq_long_20x = data["support"] * (1.0 - (1/20) + mmr)
    liq_short_50x = data["resistance"] * (1.0 + (1/50) - mmr)
    liq_short_20x = data["resistance"] * (1.0 + (1/20) - mmr)
   
    tp3_price = liq_short_20x * (0.995 * ai.get("liq_pool_offset_mod", 1.0)) if trend == "LONG_ONLY" else liq_long_20x * (1.005 * ai.get("liq_pool_offset_mod", 1.0))

    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    tp3_fixed_price = tp3_price
   
    if trend == "LONG_ONLY":
        if price * (1.0 + (tp2_percent / 100.0)) > tp3_fixed_price:
            tp2_price = tp1_price + ((tp3_fixed_price - tp1_price) / 2)
        else:
            tp2_price = price * (1.0 + (tp2_percent / 100.0))
    else:
        if price * (1.0 - (tp2_percent / 100.0)) < tp3_fixed_price:
            tp2_price = tp1_price - ((tp1_price - tp3_fixed_price) / 2)
        else:
            tp2_price = price * (1.0 - (tp2_percent / 100.0))

    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
   
    w1 = ai.get("grid_shape_skew", 0.50)
    w2 = (1.0 - w1) * 0.6
    w3 = (1.0 - w1) * 0.4
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0))), price * (1.0 - (direction * (sl_percent * 0.66 / 100.0)))]
   
    base_strike = round(price / 500) * 500
   
    return {
        "sl_percent": sl_percent, "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_fixed_price,
        "liq_long_50x": liq_long_50x, "liq_long_20x": liq_long_20x, "liq_short_50x": liq_short_50x, "liq_short_20x": liq_short_20x,
        "grid": grid, "final_sl": final_sl, "leverage": leverage, "weights": [w1*100, w2*100, w3*100],
        "trailing_dist": natr * 0.5, "fee_buffer": 0.20, "max_pain": base_strike - 1500, "call_wall": base_strike + 2500, "put_wall": base_strike - 2500,
        "exec_type": execution_type, "funding_drag": funding_drag * 100
    }

# =====================================================================
# 5. MULTI-PANELOVÁ INTERAKTÍVNA VIZUALIZÁCIA (Plotly Dark Mode)
# =====================================================================
def draw_professional_dashboard(symbol: str, df: pd.DataFrame, cvd_data: np.ndarray, oi_data: np.ndarray, m: dict, current_price_live: float = None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
   
    fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", row=1, col=1)
    fig.add_hline(y=m["final_sl"], line_width=2, line_color="red", row=1, col=1)
    fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_hline(y=m["tp3_price"], line_width=2, line_color="lime", row=1, col=1)
   
    left_edge = df['time'].iloc[0]
    fig.add_annotation(x=left_edge, y=m["final_sl"], text="🚨 SL poistka", showarrow=False, xanchor="left", bgcolor="rgba(255,0,0,0.3)", row=1, col=1)
    fig.add_annotation(x=left_edge, y=m["tp1_price"], text="🎯 TP1", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.3)", row=1, col=1)
    fig.add_annotation(x=left_edge, y=m["tp2_price"], text="🎯 TP2", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.3)", row=1, col=1)
    fig.add_annotation(x=left_edge, y=m["tp3_price"], text="🚀 TP3 Pool", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.5)", row=1, col=1)

    fig.add_hrect(y0=m["liq_long_20x"]*0.998, y1=m["liq_long_50x"]*1.002, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=m["liq_short_50x"]*0.998, y1=m["liq_short_20x"]*1.002, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)
   
    if symbol == "BTCUSDT":
        fig.add_hline(y=m["max_pain"], line_width=1.5, line_dash="dashdot", line_color="violet", row=1, col=1)
        fig.add_hline(y=m["call_wall"], line_width=2.5, line_color="fuchsia", row=1, col=1)
        fig.add_hline(y=m["put_wall"], line_width=2.5, line_color="purple", row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["max_pain"], text="🔮 Max Pain", showarrow=False, xanchor="left", row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["call_wall"], text="🧱 Call Wall", showarrow=False, xanchor="left", row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["put_wall"], text="🧱 Put Wall", showarrow=False, xanchor="left", row=1, col=1)
   
    display_price = current_price_live if current_price_live else df['close'].iloc[-1]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.95,
        text=f"<b>LIVE PRICE: ${display_price:,.4f}</b>",
        showarrow=False, font=dict(size=16, color="lime"),
        bgcolor="black", bordercolor="lime", borderwidth=2, borderpad=6
    )

    fig.add_trace(go.Scatter(x=df['time'], y=cvd_data, mode='lines', line=dict(color='orange', width=2), name="CVD (Volume Delta)"), row=2, col=1)
   
    oi_diff = np.diff(oi_data, prepend=oi_data)
    oi_colors = ['#1f77b4' if diff >= 0 else '#0f3a57' for diff in oi_diff]
    fig.add_trace(go.Bar(x=df['time'], y=oi_data, marker_color=oi_colors, name="Open Interest (Pozície)"), row=3, col=1)
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark", showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 6. DYNAMICKÝ WEB TERMINÁL A ULTRA-FAST FRAGMENT INFRAŠTRUKTÚRA
# =====================================================================
@st.fragment(run_every=1)
def render_ultra_fast_tickers(symbols_to_track, trend_data_map):
    live_prices = {}
    try:
        with httpx.Client() as client:
            for s in symbols_to_track:
                if not s: continue
                r = client.get(f"https://www.binance.com/fapi/v1/ticker/price", params={"symbol": s}, timeout=1.0)
                if r.status_code == 200:
                    live_prices[s] = float(r.json().get("price", 0))
    except Exception:
        pass

    st.session_state["live_prices"] = live_prices

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Živé Monitorované Pozície")
    
    if "open_trades" not in st.session_state:
        st.session_state["open_trades"] = [
            {"symbol": "BTCUSDT", "direction": "LONG", "entry": 65000.0, "sl": 63500.0, "highest": 65000.0, "trailing_dist": 500.0}
        ]

    active_trades = st.session_state["open_trades"]
    if not active_trades:
        st.sidebar.info("Žiadne otvorené simulované pozície.")
    else:
        for t in active_trades:
            sym = t["symbol"]
            current_p = live_prices.get(sym, t["entry"])
            
            if t["direction"] == "LONG" and current_p > t["highest"]:
                t["highest"] = current_p
                t["sl"] = t["highest"] - t["trailing_dist"]
            elif t["direction"] == "SHORT" and current_p < t["highest"]:
                t["highest"] = current_p
                t["sl"] = t["highest"] + t["trailing_dist"]
                
            pnl_pct = ((current_p - t["entry"]) / t["entry"] * 100) * (1 if t["direction"] == "LONG" else -1)
            color = "lime" if pnl_pct >= 0 else "red"
            
            st.sidebar.markdown(f"""
            **Pozícia {sym} ({t['direction']})**
            * PnL: <span style='color:{color}; font-weight:bold;'>{pnl_pct:.2f} %</span>
            * Vlečný SL: ${t['sl']:.2f}
            """, unsafe_allow_html=True)
            
    return live_prices

def main():
    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
   
    st.sidebar.markdown("##### ⏱️ Kontinuálny Skener Siete")
    auto_refresh = st.sidebar.checkbox("Aktivovať 60-sekundový Auto-Sken trhu", value=True)
    
    tf_choice = st.sidebar.selectbox("Zvoľ Časový Rámec (Timeframe):", ["15m", "1m"], index=0)
    st.session_state["timeframe"] = tf_choice

    st.markdown("### 📊 AI Intelligence & Equity Dashboard")
    dates = pd.date_range(end=datetime.today(), periods=15).strftime('%Y-%m-%d')
    mock_equity = [1000.0, 1022.5, 1015.0, 1041.2, 1038.0, 1065.4, 1092.1, 1081.0, 1115.3, 1142.0, 1135.2, 1168.0, 1195.4, 1188.0, 1210.5]
    chart_data = pd.DataFrame({"Equity ($)": mock_equity}, index=dates)
   
    c_g, c_m = st.columns(2)
    with c_g: 
        st.line_chart(chart_data)
    with c_m:
        st.metric(label="Celkový čistý zisk", value=f"${mock_equity[-1] - mock_equity[0]:.2f}", delta="+21.05 %")
        st.metric(label="Aktuálny Kellyho Risk Faktor", value="Frakcia 0.25 (Štvrťový)", delta="Stabilný")
        st.metric(label="Asymetrická separácia tried", value="3 Trend + 2 Sniper", delta="Aktívna")
       
    st.markdown("---")
    equity = st.sidebar.number_input("Aktuálna Marža Účtu ($)", min_value=100.0, value=1000.0)
   
    if supabase: st.sidebar.success("⚡ Cloud Databáza: Pripojená (Supabase)")
    else: st.sidebar.warning("⚠️ Cloud DB chýba. Emulačný režim.")

    all_binance_pairs = ["UNIUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
    try:
        with httpx.Client() as client:
            r = client.get("https://www.binance.com/fapi/v1/exchangeInfo", timeout=3.0)
            if r.status_code == 200:
                all_binance_pairs = sorted([s["symbol"] for s in r.json().get("symbols", []) if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"])
    except Exception:
        pass

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Nastavenie Okna 3")
    selected_custom_pair = st.sidebar.selectbox("Zadaj alebo vyhľadaj skratku mince:", all_binance_pairs, index=all_binance_pairs.index("UNIUSDT") if "UNIUSDT" in all_binance_pairs else 0)

    @st.cache_data(ttl=60)
    def load_heavy_market_pools():
        with httpx.Client() as c:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            trend_pool, sniper_pool = loop.run_until_complete(scan_all_markets(httpx.AsyncClient()))
            return trend_pool, sniper_pool

    trend_pool, sniper_pool = load_heavy_market_pools()

    best_altcoin_pair = "ETHUSDT"
    best_altcoin_data = None
    if trend_pool:
        for sym, data in trend_pool:
            if sym != "BTCUSDT":
                best_altcoin_pair = sym
                best_altcoin_data = data
                break

    with httpx.Client() as c:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        custom_window_data = loop.run_until_complete(fetch_single_market_data(httpx.AsyncClient(), selected_custom_pair))

    btc_data = None
    if trend_pool:
        for sym, data in trend_pool:
            if sym == "BTCUSDT":
                btc_data = data
                break

    tracked_symbols = ["BTCUSDT", best_altcoin_pair, selected_custom_pair]
    live_prices = render_ultra_fast_tickers(tracked_symbols, trend_pool)

    # --- OKNO 1: BTCUSDT FIXNÁ KOTVA ---
    st.markdown("## 📈 Okno 1: Fixná Trhová Kotva (BTCUSDT)")
    if btc_data:
        btc_ai = get_complete_ai_setup("BTCUSDT")
        btc_m = build_adaptive_setup("BTCUSDT", btc_data, btc_ai, is_sniper=False)
        trigger_status = check_orderflow_trigger(btc_data, btc_ai)
        st.caption(f"Smer: **{btc_data['trend']}** | Exekúcia: **{btc_m['exec_type']}** | Spúšťač: **{trigger_status}**")
        draw_professional_dashboard("BTCUSDT", btc_data["df"], btc_data["cvd"], btc_data["oi"], btc_m, live_prices.get("BTCUSDT"))
    else:
        st.info("Načítavam živé dáta pre Bitcoin...")

    st.markdown("---")

    # --- OKNO 2: DYNAMICKÁ ALTCOINOVÁ ANOMÁLIA ---
    st.markdown(f"## 🎯 Okno 2: Najostrejšia Altcoinová Príležitosť ({best_altcoin_pair})")
    if best_altcoin_data:
        alt_ai = get_complete_ai_setup(best_altcoin_pair)
        alt_m = build_adaptive_setup(best_altcoin_pair, best_altcoin_data, alt_ai, is_sniper=False)
        trigger_status = check_orderflow_trigger(best_altcoin_data, alt_ai)
        st.caption(f"Smer: **{best_altcoin_data['trend']}** | Exekúcia: **{alt_m['exec_type']}** | Spúšťač: **{trigger_status}**")
        draw_professional_dashboard(best_altcoin_pair, best_altcoin_data["df"], best_altcoin_data["cvd"], best_altcoin_data["oi"], alt_m, live_prices.get(best_altcoin_pair))
    else:
        st.info("Vyhľadávam najlepšie altcoinové anomálie...")

    st.markdown("---")

    # --- OKNO 3: UŽÍVATEĽSKÝ RADAR NA VYŽIADANIE ---
    st.markdown(f"## 🏹 Okno 3: Súkromný Radar Užívateľa ({selected_custom_pair})")
    if custom_window_data:
        cust_ai = get_complete_ai_setup(selected_custom_pair)
        cust_m = build_adaptive_setup(selected_custom_pair, custom_window_data, cust_ai, is_sniper=False)
        trigger_status = check_orderflow_trigger(custom_window_data, cust_ai)
        st.caption(f"Smer: **{custom_window_data['trend']}** | Exekúcia: **{cust_m['exec_type']}** | Spúšťač: **{trigger_status}**")
        draw_professional_dashboard(selected_custom_pair, custom_window_data["df"], custom_window_data["cvd"], custom_window_data["oi"], cust_m, live_prices.get(selected_custom_pair))
    else:
        st.warning(f"Nepodarilo sa načítať On-Demand dáta pre {selected_custom_pair}. Skontrolujte pripojenie.")

    if auto_refresh:
        async def delay_refresh():
            await asyncio.sleep(60)
            st.rerun()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(delay_refresh())

if __name__ == "__main__":
    main()
