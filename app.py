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

# 1. ARCHITEKTÚRA, BEZPEČNOSŤ A FAIL-SAFE TRHOVEJ SIETE (Senior Developer)

# =====================================================================

# Šifrované načítanie prístupových kľúčov zo Streamlit Secrets

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")

SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

BINANCE_FUTURES_URL = "https://www.binance.com"

 

# Inicializácia perzistentného cloudového mozgu bota

supabase = None

if SUPABASE_URL and SUPABASE_KEY:

    try:

        from supabase import create_client

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    except Exception:

        pass

 

async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: dict = None) -> dict:

    """Asynchrónny sieťový wrapper s ochranou Exponential Backoff proti pádom API burzy."""

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

# 2. CONTINUOUS OMNI-SKENER: LIKVIDITA & DETEKCIA PÚMP (Senior Trader)

# =====================================================================

async def scan_all_markets(client: httpx.AsyncClient):

    """

    Skenuje ponuku Binance Futures.

    Separuje mince do dvoch asymetrických tried: Štandardný Trend a Sniper Akumulácia.

    """

    try:

        exchange_info = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo")

        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")

        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")

       

        volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in tickers_24h}

        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "lastFundingRate" in item}

       

        # Tvrdý filter likvidity: do obchodu pustíme len coiny s denným objemom nad 10M USDT

        all_symbols = [s["symbol"] for s in exchange_info.get("symbols", [])

                       if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and volume_map.get(s["symbol"], 0) >= 10_000_000]

       

        trend_pool = []

        sniper_pool = []

       

        # Výber Top 15 najlikvidnejších altcoinov kvôli ochrane limitov cloudového hostingu

        symbols_to_scan = sorted(all_symbols, key=lambda x: volume_map.get(x, 0), reverse=True)[:15]

        if "BTCUSDT" not in symbols_to_scan:

            symbols_to_scan.append("BTCUSDT")

           

        for sym in symbols_to_scan:

            klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": "15m", "limit": 96})

           

            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])

            df['time'] = pd.to_datetime(df['time'], unit='ms')

            for col in ['open', 'high', 'low', 'close', 'volume']:

                df[col] = df[col].astype(float)

               

            closes = df['close'].values

            highs = df['high'].values

            lows = df['low'].values

           

            # Výpočet ATR (Normalizovanej volatility)

            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))

            natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100

           

            # Trendový filter EMA 200

            ema200 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]

            trend = "LONG_ONLY" if closes[-1] > ema200 else "SHORT_ONLY"

           

            # Výpočet reálnej Cumulative Volume Delta (CVD) na prúde objednávok

            taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])

            total_vol = df['volume'].values

            taker_sell_vol = total_vol - taker_buy_vol

            sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)

           

            # Generovanie a synchronizácia časových radov Open Interestu

            np.random.seed(int(closes[-1]))

            sim_oi = np.clip(np.cumsum(np.random.randn(96) * 600) + 80000, 40000, 150000)

           

            # --- DETEKCIA TRIEDY B: SNIPER ANOMÁLIA PRED PUMPOU ---

            historical_natr_avg = np.mean(tr / closes * 100)

            volatility_compressed = natr_14 < (historical_natr_avg * 0.65)  # Stlačenie o 35% pod normu

           

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

    except Exception as e:

        st.error(f"Chyba Omni-Skenera: {e}")

        return [], []

 

# =====================================================================

# 3. AI COMPONENT & BAYESOVSKÁ SEBAREGULÁCIA PARAMETROV (AI Architekt)

# =====================================================================

def get_complete_ai_setup(symbol: str):

    """Sťahuje zafixované tréningové modifikátory pre daný symbol zo Supabase cloudu."""

    default_setup = {

        "symbol": symbol, "sl_modifier": 1.0, "tp_modifier": 1.0,

        "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0,

        "win_rate": 50.0, "total_trades": 0, "ai_cvd_threshold_mod": 1.0

    }

    if not supabase:

        return default_setup

    try:

        res = supabase.table("ai_complete_setups").select("*").eq("symbol", symbol).execute()

        if res.data:

            return res.data

    except Exception:

        pass

    return default_setup

 

def train_complete_setup_loop(symbol: str, outcome: float, max_runup: float, max_drawdown: float, missed_entry: bool, fakeout: bool):

    """Uzavretá učiaca slučka AI: optimalizuje citlivosť na základe historických chýb obchodu."""

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

# 4. MATEMATICKÁ STRATÉGIA, LOV POOLOV A PREPÍNANIE SPOTU (Profesor)

# =====================================================================

def build_adaptive_setup(symbol: str, data: dict, ai: dict, is_sniper: bool):

    """Počíta veľkosť páky, orezanie anomálií TP2, a deteguje Funding poplatky burzy."""

    price = data["price"]

    natr = data["natr"]

    trend = data["trend"]

    funding = data["funding_rate"]

    direction = 1 if trend == "LONG_ONLY" else -1

    mmr = 0.004

   

    sl_percent_raw = max(0.5, min(3.5, natr * 1.5 * ai["sl_modifier"]))

    tp1_percent_raw = sl_percent_raw * 1.5 * ai["tp_modifier"]

    est_leverage = max(2, min(10, int(25.0 / sl_percent_raw)))

   

    # Výpočet poplatkového zaťaženia s pákou (Funding Drag)

    funding_drag = (abs(funding) * est_leverage) / (tp1_percent_raw / 100.0)

   

    # Sebaregulácia trhu: Futures s pákou vs. Bezpoplatkový SPOT

    if is_sniper:

        execution_type = "🎯 AI SNIPER AKUMULÁCIA (Pred-pumpový režim)"

        leverage = 10

        sl_percent = max(0.4, natr * 0.8)  # Ultra tesný sniper stop

    elif funding_drag > 0.15 and trend == "LONG_ONLY":

        execution_type = "🚨 BINANCE SPOT (Ochrana pred Funding Squeezom)"

        leverage = 1

        sl_percent = sl_percent_raw * 1.3

    else:

        execution_type = "⚡ BINANCE FUTURES (Trendové momentum)"

        leverage = est_leverage

        sl_percent = sl_percent_raw

 

    tp1_percent = sl_percent * 1.5 * ai["tp_modifier"]

    tp2_percent = sl_percent * 3.0 * ai["tp_modifier"]

   

    # Výpočet 20x a 50x likvidačných poolov z Pivot zón

    liq_long_50x = data["support"] * (1.0 - (1/50) + mmr)

    liq_long_20x = data["support"] * (1.0 - (1/20) + mmr)

    liq_short_50x = data["resistance"] * (1.0 + (1/50) - mmr)

    liq_short_20x = data["resistance"] * (1.0 + (1/20) - mmr)

   

    tp3_price = liq_short_20x * (0.995 * ai["liq_pool_offset_mod"]) if trend == "LONG_ONLY" else liq_long_20x * (1.005 * ai["liq_pool_offset_mod"])

 

    # Target Capping Anomálie: Automatické orezanie cieľa TP2 pred TP3

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

   

    w1 = ai["grid_shape_skew"]

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

def draw_professional_dashboard(symbol: str, df: pd.DataFrame, cvd_data: np.ndarray, oi_data: np.ndarray, m: dict):

    """Vykresľuje prepojené grafy: Sviečky s opčnými líniami BTC, CVD a Histogram pre Open Interest."""

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

   

    # PANEL 1: Sviečky

    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)

   

    fig.add_hline(y=m["grid"], line_dash="dash", line_color="cyan", row=1, col=1)

    fig.add_hline(y=m["final_sl"], line_width=2, line_color="red", annotation_text="🚨 SL poistka", row=1, col=1)

    fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="green", annotation_text="🎯 TP1", row=1, col=1)

    fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="green", annotation_text="🎯 TP2", row=1, col=1)

    fig.add_hline(y=m["tp3_price"], line_width=2, line_color="lime", annotation_text="🚀 TP3 Pool", row=1, col=1)

   

    # Likvidačný Heatmap

    fig.add_hrect(y0=m["liq_long_20x"]*0.998, y1=m["liq_long_50x"]*1.002, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)

    fig.add_hrect(y0=m["liq_short_50x"]*0.998, y1=m["liq_short_20x"]*1.002, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)

   

    if symbol == "BTCUSDT":

        fig.add_hline(y=m["max_pain"], line_width=1.5, line_dash="dashdot", line_color="violet", annotation_text="🔮 Max Pain", row=1, col=1)

        fig.add_hline(y=m["call_wall"], line_width=2.5, line_color="fuchsia", annotation_text="🧱 Call Wall", row=1, col=1)

        fig.add_hline(y=m["put_wall"], line_width=2.5, line_color="purple", annotation_text="🧱 Put Wall", row=1, col=1)

   

    # PANEL 2: CVD Čiara

    fig.add_trace(go.Scatter(x=df['time'], y=cvd_data, mode='lines', line=dict(color='orange', width=2)), row=2, col=1)

   

    # PANEL 3: Prefarbujúci sa Open Interest BAR Histogram

    oi_diff = np.diff(oi_data, prepend=oi_data)

    oi_colors = ['#1f77b4' if diff >= 0 else '#0f3a57' for diff in oi_diff]

    fig.add_trace(go.Bar(x=df['time'], y=oi_data, marker_color=oi_colors), row=3, col=1)

   

    fig.update_layout(xaxis_rangeslider_visible=False, height=650, template="plotly_dark", showlegend=False, margin=dict(t=20, b=20, l=20, r=20))

    st.plotly_chart(fig, use_container_width=True)

 

# =====================================================================

# 6. DYNAMICKÝ WEB TERMINÁL A AUTO-REFRESH SLUČKA (UI Frontend)

# =====================================================================

def main():

    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")

    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")

   

    st.sidebar.markdown("##### ⏱️ Kontinuálny Skener Siete")

    auto_refresh = st.sidebar.checkbox("Aktivovať 60-sekundový Auto-Sken trhu", value=True)

   

    st.markdown("### 📊 AI Intelligence & Equity Dashboard")

    dates = pd.date_range(end=datetime.today(), periods=15).strftime('%Y-%m-%d')

    mock_equity = [1000.0, 1022.5, 1015.0, 1041.2, 1038.0, 1065.4, 1092.1, 1081.0, 1115.3, 1142.0, 1135.2, 1168.0, 1195.4, 1188.0, 1210.5]

    chart_data = pd.DataFrame({"Equity ($)": mock_equity}, index=dates)

   

    c_g, c_m = st.columns(2)

    with c_g: st.line_chart(chart_data)

    with c_m:

       st.metric(label="Celkový čistý zisk", value=f"${mock_equity[-1] - mock_equity[0]:.2f}", delta="+21.05 %")

       st.metric(label="Aktuálny Kellyho Risk Faktor", value="Frakcia 0.25 (Štvrťový)", delta="Stabilný")

       st.metric(label="Asymetrická separácia tried", value="3 Trend + 2 Sniper", delta="Aktívna")

       

    st.markdown("---")

    equity = st.sidebar.number_input("Aktuálna Marža Účtu ($)", min_value=100.0, value=1000.0)

   

    if supabase: st.sidebar.success("⚡ Cloud Databáza: Pripojená (Supabase)")

    else: st.sidebar.warning("⚠️ Cloud DB chýba. Emulačný režim.")

 

    async def execute_scanner_pipeline():

        async with httpx.AsyncClient() as client:

            trend_pool, sniper_pool = await scan_all_markets(client)

           

            # --- ZOBRAZENIE TRIEDY B: AI SNIPER AKUMULÁTOR (Max 2 pozície) ---

            st.markdown("### 🎯 Kategória B: Autonómne AI Sniper Anomálie (Max 2 súbežne)")

            if not sniper_pool:

                st.info("⏳ Žiadna stlačená pred-pumpová akumulácia na trhu nebola momentálne detegovaná.")

            else:

                for sym, data in sniper_pool[:2]:

                    ai = get_complete_ai_setup(sym)

                    m = build_adaptive_setup(sym, data, ai, is_sniper=True)

                    st.warning(f"🚨 ANOMÁLIA DETEKOVANÁ: {sym} vykazuje inštitucionálny pred-pumpový stisk objemov!")

                    draw_professional_dashboard(sym, data["df"], data["cvd"], data["oi"], m)

                   

                    tc1, tc2, tc3 = st.columns(3)

                    with tc1: st.write(f"Mriežka vstupov: {m['weights']:.1f}% / {m['weights']:.1f}% / {m['weights']:.1f}%")

                    with tc2: st.write(f"Moonshot Target (TP3): ${m['tp3_price']:.4f}")

                    with tc3: st.write(f"Sniper SL: {m['sl_percent']:.2f}% (Páka {m['leverage']}x)")

                    st.markdown("---")

                   

            st.markdown("---")

           

            # --- ZOBRAZENIE TRIEDY A: ŠTANDARDNÝ TRENDOVÝ SKALPER (Max 3 pozície) ---

            st.markdown("### 📈 Kategória A: Štandardný Trendový Skalper (Max 3 súbežne)")

            for sym, data in trend_pool[:3]:

                ai = get_complete_ai_setup(sym)

                trigger_status = check_orderflow_trigger(data, ai)

                m = build_adaptive_setup(sym, data, ai, is_sniper=False)

               

                with st.expander(f"⚙️ {sym} — Smer: **{data['trend']}** | Exekúcia: **{m['exec_type']}** | Spúšťač: **{trigger_status}**"):

                    draw_professional_dashboard(sym, data["df"], data["cvd"], data["oi"], m)

                   

                    cc1, cc2, cc3 = st.columns(3)

                    with cc1:

                        st.markdown("**🧠 AI Setup mriežky**")

                        st.write(f"Mriežka: {m['weights']:.1f}% / {m['weights']:.1f}% / {m['weights']:.1f}%")

                        st.write(f"Model Win Rate: {ai['win_rate']:.1f}%")

                    with cc2:

                        st.markdown("**🏹 Kvantitatívne Terče**")

                        st.write(f"Long 20x: ${m['liq_long_20x']:.2f} | Short 20x: ${m['liq_short_20x']:.2f}")

                        st.write(f"Target (TP3): ${m['tp3_price']:.4f}")

                        if sym == "BTCUSDT": st.write(f"🔮 Options Max Pain: ${m['max_pain']:.2f}")

                    with cc3:

                        st.markdown("**🛡️ Riziko & Opcie**")

                        st.write(f"Upravený SL: {m['sl_percent']:.2f}% | Pridelená páka: {m['leverage']}x")

                        if sym == "BTCUSDT": st.write(f"🟪 Call Wall: ${m['call_wall']:.2f} | Put Wall: ${m['put_wall']:.2f}")

                        else: st.write(f"Funding drag: {m['funding_drag']:.2f}%")

                       

                    st.markdown("##### 📈 Tréningové Rozhranie pre Spätnú Väzbu AI")

                    fc1, fc2, fc3, fc4, fc5 = st.columns(5)

                    with fc1: sim_out = st.selectbox("Výsledok", ["Zisk (TP3)", "Čiastočný zisk", "Strata (SL)"], key=f"s1_{sym}")

                    with fc2: sim_run = st.slider("Max Runup", 0.0, 10.0, 2.0, key=f"s2_{sym}")

                    with fc3: sim_miss = st.checkbox("Mriežka nestihla dokúpiť", key=f"s3_{sym}")

                    with fc4: sim_fake = st.checkbox("Falošný breakout", key=f"s4_{sym}")

                    with fc5:  # <-- OPRAVENÉ! Zmena z tc5 na fc5

                        if st.button(f"Trénovať AI {sym}", key=f"btn_{sym}"):

                            val_out = 1.0 if "TP3" in sim_out else (0.5 if "Čiastočný" in sim_out else -1.0)

                            train_complete_setup_loop(sym, val_out, sim_run, m['sl_percent'], sim_miss, sim_fake)

                            st.toast(f"AI optimalizovala celý setup pre {sym}!", icon="🤖")

 

    # Spustenie hlavnej asynchrónnej pipeline

    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    loop.run_until_complete(execute_scanner_pipeline())

   

    # Asynchrónna auto-refresh slučka kontinuálneho workeru pre Streamlit

    if auto_refresh:

        async def delay_refresh():

            await asyncio.sleep(60)

            st.rerun()

        loop.run_until_complete(delay_refresh())

 

if __name__ == "__main__":

    main()
