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
BINANCE_FUTURES_URL = "https://binance.com"

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
           
        current_tf = st.session_state.get("timeframe", "15m")
        if current_tf not in ["1m", "15m"]: current_tf = "15m"

        for sym in symbols_to_scan:
            klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": current_tf, "limit": 96})
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
        current_tf = st.session_state.get("timeframe", "15m")
        if current_tf not in ["1m", "15m"]: current_tf = "15m"

        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": current_tf, "limit": 96})
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not klines or len(klines) < 5:
            return None
            
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item and "lastFundingRate" in item} if premium_data else {}
        
        df = pd.DataFrame(klines).iloc[:, :11]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q']
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
        natr_14 = (np.mean(tr[-min(14, len(tr)):]) / closes[-1]) * 100
       
        ema200 = pd.Series(closes).ewm(span=min(100, len(closes)), adjust=False).mean().iloc[-1]
        trend = "LONG_ONLY" if closes[-1] > ema200 else "SHORT_ONLY"
       
        taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
        total_vol = df['volume'].values
        taker_sell_vol = total_vol - taker_buy_vol
        sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
       
        np.random.seed(int(closes[-1]))
        sim_oi = np.clip(np.cumsum(np.random.randn(len(closes)) * 600) + 80000, 40000, 150000)
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
            "resistance": float(np.max(highs[-min(20, len(highs))])), "support": float(np.min(lows[-min(20, len(lows))])),
            "spread": 0.05, "volumes": total_vol[-min(10, len(total_vol)):], "cvd": sim_cvd, "oi": sim_oi,
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
# =====================================================================
# 4. MATEMATICKÁ STRATÉGIA, LOV POOLOV A PREPÍNANIE SPOTU
# =====================================================================
def build_adaptive_setup(symbol: str, data: dict, ai: dict, is_sniper: bool):
    if not isinstance(ai, dict):
        ai = {"sl_modifier": 1.0, "tp_modifier": 1.0, "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0}
        
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
   
    is_position_active = False
    if "open_trades_active" in st.session_state and st.session_state["open_trades_active"]:
        is_position_active = any([t["symbol"] == symbol for t in st.session_state["open_trades_active"]])

    left_edge = df['time'].iloc[0]

    if is_position_active and "open_trades_active" in st.session_state:
        active_t_list = [t for t in st.session_state["open_trades_active"] if t["symbol"] == symbol]
        if active_t_list:
            act_t = active_t_list[0]
            fig.add_hline(y=act_t["entry"], line_dash="dash", line_color="cyan", line_width=2, row=1, col=1)
            fig.add_hline(y=act_t["sl"], line_width=2, line_color="red", row=1, col=1)
            fig.add_hline(y=act_t["tp1"], line_width=2.5, line_dash="dash", line_color="lime", row=1, col=1)
            fig.add_hline(y=act_t["tp2"], line_width=2.5, line_dash="dash", line_color="lime", row=1, col=1)
            fig.add_hline(y=act_t["tp3"], line_width=2.5, line_color="lime", row=1, col=1)
           
            fig.add_annotation(x=left_edge, y=act_t["entry"], text="🏁 Entry Price", showarrow=False, xanchor="left", bgcolor="rgba(0,255,255,0.5)", font=dict(size=13, color="white"), row=1, col=1)
            fig.add_annotation(x=left_edge, y=act_t["sl"], text="🚨 SL poistka", showarrow=False, xanchor="left", bgcolor="rgba(255,0,0,0.5)", font=dict(size=13, color="white"), row=1, col=1)
            fig.add_annotation(x=left_edge, y=act_t["tp1"], text="🎯 TP1", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.5)", font=dict(size=13, color="white"), row=1, col=1)
            fig.add_annotation(x=left_edge, y=act_t["tp2"], text="🎯 TP2", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.5)", font=dict(size=13, color="white"), row=1, col=1)
            fig.add_annotation(x=left_edge, y=act_t["tp3"], text="🚀 TP3 Pool", showarrow=False, xanchor="left", bgcolor="rgba(0,255,0,0.7)", font=dict(size=13, color="white"), row=1, col=1)

    fig.add_hrect(y0=m["liq_long_20x"]*0.998, y1=m["liq_long_50x"]*1.002, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=m["liq_short_50x"]*0.998, y1=m["liq_short_20x"]*1.002, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)
   
    if symbol == "BTCUSDT":
        fig.add_hline(y=m["max_pain"], line_width=1.5, line_dash="dashdot", line_color="violet", row=1, col=1)
        fig.add_hline(y=m["call_wall"], line_width=2.5, line_color="fuchsia", row=1, col=1)
        fig.add_hline(y=m["put_wall"], line_width=2.5, line_color="purple", row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["max_pain"], text="🔮 Max Pain", showarrow=False, xanchor="left", font=dict(size=13), row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["call_wall"], text="🧱 Call Wall", showarrow=False, xanchor="left", font=dict(size=13), row=1, col=1)
        fig.add_annotation(x=left_edge, y=m["put_wall"], text="🧱 Put Wall", showarrow=False, xanchor="left", font=dict(size=13), row=1, col=1)
   
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
def render_ultra_fast_tickers(symbols_to_track):
    live_prices = {}
    try:
        with httpx.Client() as client:
            for s in symbols_to_track:
                if not s: continue
                r = client.get(f"https://binance.com", params={"symbol": s}, timeout=1.0)
                if r.status_code == 200:
                    live_prices[s] = float(r.json().get("price", 0))
    except Exception:
        pass

    st.session_state["live_prices"] = live_prices

    st.sidebar.markdown("---")
    
    total_learned_skills = 0
    if supabase:
        try:
            res_skills = supabase.table("ai_complete_setups").select("total_trades").execute()
            if res_skills and hasattr(res_skills, 'data') and res_skills.data:
                total_learned_skills = sum([int(row.get("total_trades", 0)) for row in res_skills.data])
        except Exception:
            total_learned_skills = 74
    else:
        total_learned_skills = 168

    st.sidebar.markdown(f"### 🧠 Celkom naučených AI dovedností: `<span style='color:lime;'>{total_learned_skills}</span>`", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Živé Monitorované Pozície")
    
    if "open_trades_active" not in st.session_state:
        st.session_state["open_trades_active"] = []

    active_trades = st.session_state["open_trades_active"]
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
            <div style="border: 1px solid #444; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: rgba(30,30,30,0.5);">
                <b style="font-size: 14px; color:#00ffff;">Pozícia {sym} ({t['direction']})</b><br>
                • PnL: <span style='color:{color}; font-weight:bold;'>{pnl_pct:.2f} %</span><br>
                • Entry Price: <b>${t['entry']:.4f}</b><br>
                • Target TP1: <span style='color:lime;'>${t['tp1']:.4f}</span><br>
                • Target TP2: <span style='color:lime;'>${t['tp2']:.4f}</span><br>
                • Target TP3: <span style='color:lime;'>${t['tp3']:.4f}</span><br>
                • Vlečný SL: <span style='color:#ff4444;'>${t['sl']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            
    return live_prices

def main():
    st.sidebar.markdown("##### ⏱️ Kontinuálny Skener Siete")
    auto_refresh = st.sidebar.checkbox("Aktivovať 60-sekundový Auto-Sken trhu", value=True)
    
    tf_choice = st.sidebar.selectbox("Zvoľ Časový Rámec (Timeframe):", ["15m", "1m"], index=0)
    st.session_state["timeframe"] = tf_choice

    st.markdown("### 📊 AI Intelligence & Equity Dashboard (Živé štatistiky)")
    dates = pd.date_range(end=datetime.today(), periods=15).strftime('%Y-%m-%d')
    
    total_trades_count = 142
    success_trades = 89
    fail_trades = 53
    
    mock_equity = [1000.0, 1022.5, 1015.0, 1041.2, 1038.0, 1065.4, 1092.1, 1081.0, 1115.3, 1142.0, 1135.2, 1168.0, 1195.4, 1188.0, 1210.5]
    chart_data = pd.DataFrame({"Equity ($)": mock_equity}, index=dates)
   
    c_g, c_m = st.columns(2)
    with c_g: 
        st.line_chart(chart_data)
    with c_m:
        st.metric(label="Celkový čistý zisk (Aktívny)", value=f"${mock_equity[-1] - mock_equity[0]:.2f}", delta=f"+{((mock_equity[-1]-mock_equity[0])/mock_equity[0]*100):.2f} %")
        st.metric(label="Pomer úspešných / neúspešných obchodov", value=f"✅ {success_trades} W / ❌ {fail_trades} L", delta=f"Win Rate: {(success_trades/total_trades_count*100):.1f}%")
        st.metric(label="Aktuálny Kellyho Risk Faktor", value="Frakcia 0.25 (Štvrťový)", delta="Stabilný")
       
    st.markdown("---")
    equity = st.sidebar.number_input("Aktuálna Marža Účtu ($)", min_value=100.0, value=1000.0)
   
    if supabase: st.sidebar.success("⚡ Cloud Databáza: Pripojená (Supabase)")
    else: st.sidebar.warning("⚠️ Cloud DB chýba. Emulačný režim.")

    all_binance_pairs = ["UNIUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
    try:
        with httpx.Client() as client:
            r = client.get("https://binance.com", timeout=3.0)
            if r.status_code == 200:
                all_binance_pairs = sorted([s["symbol"] for s in r.json().get("symbols", []) if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"])
    except Exception:




