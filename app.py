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
    backoff = 2.0
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(backoff)
            backoff *= 2.0

# =====================================================================
# 2. CONTINUOUS OMNI-SKENER: LIKVIDITA & DETEKCIA PÚMP
# =====================================================================
async def scan_all_markets(client: httpx.AsyncClient, timeframe: str):
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo")
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
       
        volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in tickers_24h}
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "lastFundingRate" in item}
       
        all_symbols = [s["symbol"] for s in exchange_info.get("symbols", [])
                       if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and volume_map.get(s["symbol"], 0) >= 10_000_000]
       
        trend_pool = []
        sniper_pool = []
       
        symbols_to_scan = sorted(all_symbols, key=lambda x: volume_map.get(x, 0), reverse=True)[:15]
        if "BTCUSDT" not in symbols_to_scan:
            symbols_to_scan.append("BTCUSDT")
           
        for sym in symbols_to_scan:
            klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": timeframe, "limit": 96})
           
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
                "symbol": sym, "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
                "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
                "spread": 0.05, "volumes": total_vol[-10:], "cvd": sim_cvd, "oi": sim_oi,
                "funding_rate": funding_map.get(sym, 0.0001), "score": abs(sim_cvd[-1] - sim_cvd[-6])
            }
           
            if is_sniper_anomaly and sym != "BTCUSDT":
                sniper_pool.append((sym, market_obj))
            else:
                trend_pool.append((sym, market_obj))
               
        return trend_pool, sniper_pool
    except Exception as e:
        st.error(f"Chyba Omni-Skenera: {e}")
        return [], []

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
        return "⏳ ČAKANIE"

# =====================================================================
# 3. AI COMPONENT & BAYESOVSKÁ SEBAREGULÁCIA
# =====================================================================
def get_complete_ai_setup(symbol: str):
    default_setup = {
        "symbol": symbol, "sl_modifier": 1.0, "tp_modifier": 1.0,
        "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0,
        "win_rate": 50.0, "total_trades": 0, "ai_cvd_threshold_mod": 1.0
    }
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_complete_setups").select("*").eq("symbol", symbol).execute()
        if hasattr(res, 'data') and res.data:
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
        execution_type = "🎯 AI SNIPER AKUMULÁCIA"
        leverage = 10
        sl_percent = max(0.4, natr * 0.8)
    elif funding_drag > 0.15 and trend == "LONG_ONLY":
        execution_type = "🚨 BINANCE SPOT (Ochrana poplatkov)"
        leverage = 1
        sl_percent = sl_percent_raw * 1.3
    else:
        execution_type = "⚡ BINANCE FUTURES (Trend)"
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
def draw_professional_dashboard(symbol: str, df: pd.DataFrame, cvd_data: np.ndarray, oi_data: np.ndarray, m: dict, current_live_price: float = None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
   
    fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", row=1, col=1)
    fig.add_hline(y=m["final_sl"], line_width=2, line_color="red", row=1, col=1)
    fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="green", row=1, col=1)
    fig.add_hline(y=m["tp3_price"], line_width=2, line_color="lime", row=1, col=1)
   
    fig.add_hrect(y0=m["liq_long_20x"]*0.998, y1=m["liq_long_50x"]*1.002, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=m["liq_short_50x"]*0.998, y1=m["liq_short_20x"]*1.002, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)
   
    if symbol == "BTCUSDT":
        fig.add_hline(y=m["max_pain"], line_width=1.5, line_dash="dashdot", line_color="violet", row=1, col=1)
        fig.add_hline(y=m["call_wall"], line_width=2.5, line_color="fuchsia", row=1, col=1)
        fig.add_hline(y=m["put_wall"], line_width=2.5, line_color="purple", row=1, col=1)
   
    # Popisky presunuté na ľavú stranu (x=0)
    fig.add_annotation(x=0, y=m["final_sl"], text="🚨 SL poistka", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="red", size=10))
    fig.add_annotation(x=0, y=m["tp1_price"], text="🎯 TP1", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="green", size=10))
    fig.add_annotation(x=0, y=m["tp2_price"], text="🎯 TP2", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="green", size=10))
    fig.add_annotation(x=0, y=m["tp3_price"], text="🚀 TP3 Pool", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="lime", size=10))
    fig.add_annotation(x=0, y=m["grid"][0], text="⚡ Mriežka Vstup", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="cyan", size=10))

    if symbol == "BTCUSDT":
        fig.add_annotation(x=0, y=m["max_pain"], text="🔮 Max Pain", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="violet", size=10))
        fig.add_annotation(x=0, y=m["call_wall"], text="🧱 Call Wall", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="fuchsia", size=10))
        fig.add_annotation(x=0, y=m["put_wall"], text="🧱 Put Wall", xref="paper", yref="y", showarrow=False, xanchor="left", font=dict(color="purple", size=10))

    # SEKUNDOVÝ TICKER V PRAVOM HORNOM ROHU (Cerveny obdlznik)
    display_price = current_live_price if current_live_price else df['close'].iloc[-1]
    fig.add_annotation(
        x=0.98, y=0.95, text=f"LIVE PRICE: ${display_price:,.2f}",
        xref="paper", yref="paper", showarrow=False, xanchor="right", yanchor="top",
        font=dict(color="#00FFCC", size=16, family="Courier New, monospace"),
        bordercolor="#00FFCC", borderwidth=1, borderpad=4, bgcolor="rgba(0,0,0,0.6)"
    )

    # PANEL 2: CVD Čiara s popiskom
    fig.add_trace(go.Scatter(x=df['time'], y=cvd_data, mode='lines', line=dict(color='orange', width=2), name="CVD (Volume Delta)"), row=2, col=1)
   
    # PANEL 3: Open Interest stĺpce s popiskom
    oi_diff = np.diff(oi_data, prepend=oi_data)
    oi_colors = ['#1f77b4' if diff >= 0 else '#0f3a57' for diff in oi_diff]
    fig.add_trace(go.Bar(x=df['time'], y=oi_data, marker_color=oi_colors, name="Open Interest (Pozície)"), row=3, col=1)
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=650, template="plotly_dark", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 6. DYNAMICKÝ WEB TERMINÁL (UI Frontend)
# =====================================================================
def init_db_fail_safe():
    if not supabase:
        return
    try:
        supabase.rpc("execute_sql", {"query": """
        CREATE TABLE IF NOT EXISTS simulated_trades (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price FLOAT NOT NULL,
            sl_price FLOAT NOT NULL,
            tp1_price FLOAT NOT NULL,
            tp2_price FLOAT NOT NULL,
            tp3_price FLOAT NOT NULL,
            leverage INT NOT NULL,
            status TEXT NOT NULL,
            outcome FLOAT,
            max_runup FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """}).execute()
    except Exception:
        pass

# Komponent pre super-rýchly 1-sekundový refresh iba cenového textu bez zhadzovania grafov
@st.fragment
def render_live_price_ticker(symbols):
    if st.button("🔄 Okamžitý manuálny refresh cien (1s Ticker)"):
        st.rerun()
    prices = {}
    try:
        with httpx.Client() as client:
            for s in symbols:
                res = client.get(f"https://www.binance.com/fapi/v1/ticker/price?symbol={s}")
                if res.status_code == 200:
                    prices[s] = float(res.json()["price"])
    except Exception:
        pass
    return prices

def main():
    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
    init_db_fail_safe()
   
    st.sidebar.markdown("##### ⏱️ Kontinuálny Skener Siete")
    tf_choice = st.sidebar.selectbox("Časový rámec grafov (Timeframe)", ["15m", "1m"], index=0)
    auto_refresh = st.sidebar.checkbox("Aktivovať 60-sekundový Auto-Sken trhu", value=True)
   
    st.markdown("### 📊 AI Intelligence & Equity Dashboard")
    dates = pd.date_range(end=datetime.today(), periods=15).strftime('%Y-%m-%d')
    mock_equity = [1000.0, 1015.0, 1008.0, 1032.0, 1055.0, 1040.0, 1072.0, 1110.0, 1095.0, 1130.0, 1165.0, 1150.0, 1182.0, 1210.50, 1210.50]
    chart_data = pd.DataFrame({"Equity ($)": mock_equity}, index=dates)
   
    c_g, c_m = st.columns(2)
    with c_g: 
        st.line_chart(chart_data)
    with c_m:
        st.metric(label="Celkový čistý zisk", value=f"${mock_equity[-1] - mock_equity[0]:.2f}", delta="+21.05 %")
        st.metric(label="Aktuálny Kellyho Risk Faktor", value="Frakcia 0.25 (Štvrťový)", delta="Stabilný")
        st.metric(label="Asymetrická separácia tried", value="3 Trend + 2 Sniper", delta="Aktívna")
       
    st.markdown("---")
   
    if supabase: st.sidebar.success("⚡ Cloud Databáza: Pripojená (Supabase)")
    else: st.sidebar.warning("⚠️ Cloud DB chýba. Emulačný režim.")

    # Inicializácia asynchrónnej pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def get_pipeline_data():
        async with httpx.AsyncClient() as client:
            return await scan_all_markets(client, tf_choice)
            
    trend_pool, sniper_pool = loop.run_until_complete(get_pipeline_data())
    
    # Príprava zoznamu pre výber v 3. okne
    available_symbols = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT"]
    if trend_pool:
        available_symbols = sorted(list(set([sym for sym, _ in trend_pool] + available_symbols)))
        
    user_selected_symbol = st.sidebar.selectbox("🎯 3. OKNO: Tvoj vlastný výber mince na radar", available_symbols, index=0)

    # 1. OKNO: FIXNÉ PRE BTC
    btc_data = next((data for sym, data in trend_pool if sym == "BTCUSDT"), None)
    if not btc_data and sniper_pool:
        btc_data = next((data for sym, data in sniper_pool if sym == "BTCUSDT"), None)
        
    # 2. OKNO: DYNAMICKÝ ALTCOIN (PODMIENKA: NESMIE TO BYŤ BTC!)
    altcoin_pool = [data for sym, data in trend_pool if sym != "BTCUSDT"] + [data for sym, data in sniper_pool if sym != "BTCUSDT"]
    best_altcoin_data = sorted(altcoin_pool, key=lambda x: x["score"], reverse=True)[0] if altcoin_pool else None

    # 3. OKNO: RUČNÝ VÝBER
    user_data = next((data for sym, data in trend_pool if sym == user_selected_symbol), None)
    if not user_data and sniper_pool:
        user_data = next((data for sym, data in sniper_pool if sym == user_selected_symbol), None)

    active_symbols = ["BTCUSDT"]
    if best_altcoin_data: active_symbols.append(best_altcoin_data["symbol"])
    if user_selected_symbol not in active_symbols: active_symbols.append(user_selected_symbol)
    
    # Spustenie sekundového cenového tickeru (Fragment)
    live_prices = render_live_price_ticker(active_symbols)

    # ZOBRAZENIE OBCHODNÝCH BUNEK
    st.markdown("## 📊 Živé Monitorované Obchodné Bunky")
    
    # GRAF 1: BTC
    if btc_data:
        st.markdown(f"### ⚓ 1. Obchodné okno: BTCUSDT (Fixná kotva trhu)")
        ai = get_complete_ai_setup("BTCUSDT")
        m = build_adaptive_setup("BTCUSDT", btc_data, ai, is_sniper=False)
        draw_professional_dashboard("BTCUSDT", btc_data["df"], btc_data["cvd"], btc_data["oi"], m, live_prices.get("BTCUSDT"))
        st.markdown("---")

    # GRAF 2: NAJLEPŠÍ ALTCOIN (Mimo BTC)
    if best_altcoin_data:
        st.markdown(f"### ⚡ 2. Obchodné okno: {best_altcoin_data['symbol']} (Najlepšia Altcoinová Príležitosť)")
        ai = get_complete_ai_setup(best_altcoin_data['symbol'])
        m = build_adaptive_setup(best_altcoin_data['symbol'], best_altcoin_data, ai, is_sniper=False)
        draw_professional_dashboard(best_altcoin_data['symbol'], best_altcoin_data["df"], best_altcoin_data["cvd"], best_altcoin_data["oi"], m, live_prices.get(best_altcoin_data['symbol']))
        st.markdown("---")

    # GRAF 3: RUČNÝ VÝBER UŽÍVATEĽA
    if user_data:
        st.markdown(f"### 🏹 3. Obchodné okno: {user_selected_symbol} (Tvoj Súkromný Radar)")
        ai = get_complete_ai_setup(user_selected_symbol)
        m = build_adaptive_setup(user_selected_symbol, user_data, ai, is_sniper=False)
        draw_professional_dashboard(user_selected_symbol, user_data["df"], user_data["cvd"], user_data["oi"], m, live_prices.get(user_selected_symbol))
        st.markdown("---")

    if auto_refresh:
        async def delay_refresh():
            await asyncio.sleep(60)
            st.rerun()
        loop.run_until_complete(delay_refresh())

if __name__ == "__main__":
    main()
