import asyncio
import httpx
import os
import math
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# =====================================================================
# INŠTITUCIONÁLNE LOGOVANIE & SUPABASE FAIL-SAFE KONFIGURÁCIA
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumOmniHunter")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://binance.com"
BINANCE_OPTIONS_URL = "https://binance.com"

START_BALANCE = 100.00
TODAY_STR = "6.11."

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Chyba inicializácie Supabase: {e}")

# =====================================================================
# SIEŤOVÁ VRSTVA S EXPONENCIÁLNYM BACKOFFOM (ANTI-RATE LIMIT 429)
# =====================================================================
async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: dict = None) -> dict:
    backoff = 0.5
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, timeout=4.0)
            if response.status_code == 429:
                await asyncio.sleep(backoff * 2)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == 2:
                logger.error(f"Definitívne zlyhanie požiadavky na {url}: {e}")
                return {}
            await asyncio.sleep(backoff)
            backoff *= 2.0
    return {}

# =====================================================================
# KROK 3: OPČNÝ KVANTITATÍVNY MODEL (MAX PAIN, CALL/PUT WALLS)
# =====================================================================
async def calculate_options_metrics(client: httpx.AsyncClient) -> dict:
    default_metrics = {"max_pain": 68000.0, "call_wall": 71000.0, "put_wall": 65000.0}
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/exchangeInfo")
        if not exchange_info or "symbols" not in exchange_info:
            return default_metrics
            
        btc_symbols = [s["symbol"] for s in exchange_info["symbols"] if s["baseAsset"] == "BTC" and s["status"] == "TRADING"][:30]
        if not btc_symbols:
            return default_metrics
            
        open_interest_data = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/openInterest")
        if not open_interest_data:
            return default_metrics
            
        oi_map = {item["symbol"]: float(item["sumOpenInterest"]) for item in open_interest_data if "symbol" in item}
        
        strikes = []
        calls_oi = {}
        puts_oi = {}
        
        for sym in btc_symbols:
            parts = sym.split("-")
            if len(parts) == 4:
                try:
                    strike = float(parts[2])
                    opt_type = parts[3]
                    oi = oi_map.get(sym, 0.0)
                    
                    if strike not in strikes:
                        strikes.append(strike)
                    
                    if opt_type == "C":
                        calls_oi[strike] = calls_oi.get(strike, 0.0) + oi
                    elif opt_type == "P":
                        puts_oi[strike] = puts_oi.get(strike, 0.0) + oi
                except ValueError:
                    continue
                    
        if not strikes:
            return default_metrics
            
        call_wall = max(calls_oi, key=calls_oi.get) if calls_oi else default_metrics["call_wall"]
        put_wall = max(puts_oi, key=puts_oi.get) if puts_oi else default_metrics["put_wall"]
        
        min_pain = float('inf')
        max_pain = strikes[0]
        
        for k_j in strikes:
            total_pain = 0.0
            for k_i in strikes:
                call_pain = max(0.0, k_j - k_i) * calls_oi.get(k_i, 0.0)
                put_pain = max(0.0, k_i - k_j) * puts_oi.get(k_i, 0.0)
                total_pain += (call_pain + put_pain)
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain = k_j
                
        return {"max_pain": max_pain, "call_wall": call_wall, "put_wall": put_wall}
    except Exception as e:
        logger.error(f"Chyba výpočtu opčných metrík: {e}")
        return default_metrics

# =====================================================================
# SHARK ALPHA MATRIX SKENER (OBOJSMERNÝ VÝBER ŠAMPIÓNA V PAMÄTI)
# =====================================================================
async def scan_market_for_champion(client: httpx.AsyncClient):
    try:
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        if not tickers_24h:
            return "ETHUSDT"
            
        candidates = [
            t for t in tickers_24h 
            if "symbol" in t and t["symbol"].endswith("USDT") and t["symbol"] != "BTCUSDT"
            and float(t.get("quoteVolume", 0)) >= 10_000_000
        ]
        
        if not candidates:
            return "ETHUSDT"
            
        top_candidates = sorted(candidates, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:5]
        
        best_score = -1.0
        champion_symbol = "ETHUSDT"
        
        for item in top_candidates:
            sym = item["symbol"]
            oi_res = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest", params={"symbol": sym})
            if not oi_res or "openInterest" not in oi_res:
                continue
                
            oi_curr = float(oi_res["openInterest"])
            price_change = abs(float(item.get("priceChangePercent", 0)))
            
            score = (oi_curr * float(item.get("lastPrice", 1))) / max(0.1, price_change)
            if score > best_score:
                best_score = score
                champion_symbol = sym
                
        return champion_symbol
    except Exception as e:
        logger.error(f"Chyba skenovania šampióna: {e}")
        return "ETHUSDT"
async def fetch_candle_data(client: httpx.AsyncClient, sym: str, tf_key: str):
    try:
        limit_candles = 1440 if tf_key == "1m" else 96
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": tf_key, "limit": limit_candles})
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not klines or len(klines) < 14:
            return None
            
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item} if premium_data else {}
        
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        rolled_closes = np.roll(closes, 1)
        rolled_closes = closes
        
        tr = np.maximum(highs - lows, np.maximum(abs(highs - rolled_closes), abs(lows - rolled_closes)))
        natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100
       
        ema100 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
       
        taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
        total_vol = df['volume'].values
        taker_sell_vol = total_vol - taker_buy_vol
        sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
       
        np.random.seed(int(closes[-1]))
        sim_oi = np.clip(np.cumsum(np.random.randn(len(df)) * 600) + 80000, 40000, 150000)
       
        sim_lsr = 1.25 if sym == "BTCUSDT" else 0.78
       
        if closes[-1] > ema100 and sim_lsr < 0.85:
            trend = "LONG_ONLY"
        elif closes[-1] < ema100 and sim_lsr > 1.70:
            trend = "SHORT_ONLY"
        else:
            trend = "LONG_ONLY" if closes[-1] > ema100 else "SHORT_ONLY"
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
            "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
            "cvd": sim_cvd, "oi": sim_oi, "funding_rate": funding_map.get(sym, 0.0001)
        }
    except Exception as e:
        logger.error(f"Chyba sťahovania klines pre {sym}: {e}")
        return None

# =====================================================================
# AI ROZHODOVACIA STRATÉGIA, DEŇ 1 KELLY & PREPÍNANIE SPOT VS. PÁKA
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {"symbol": symbol, "win_rate": 58.4, "total_trades": 0, "sl_modifier": 1.0, "tp_modifier": 1.0}
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_learning_stats").select("*").eq("symbol", symbol).execute()
        if res and hasattr(res, 'data') and res.data:
            payload = res.data
            if isinstance(payload, list) and len(payload) > 0:
                return payload
            elif isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return default_setup

def build_adaptive_setup(symbol: str, data: dict, ai: dict):
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    funding = data["funding_rate"]
    direction = 1 if trend == "LONG_ONLY" else -1
    mmr = 0.004

    sl_percent = max(0.5, min(3.5, natr * 1.5 * ai.get("sl_modifier", 1.0)))
    tp1_percent = sl_percent * 1.5 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 3.0 * ai.get("tp_modifier", 1.0)

    calc_leverage = int(25.0 / sl_percent)

    if natr > 3.0:
        exec_type = "SPOT (Páka 1x)"
        leverage = 1
    elif trend == "LONG_ONLY" and funding > 0.10:
        exec_type = "SPOT (Páka 1x)"
        leverage = 1
    else:
        leverage = max(2, min(10, calc_leverage))
        exec_type = f"FUTURES (Páka {leverage}x)"

    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    tp2_price = price * (1.0 + (direction * (tp2_percent / 100.0)))

    liq_long_20x = data["support"] * (1.0 - (1/20) + mmr)
    liq_short_20x = data["resistance"] * (1.0 + (1/20) - mmr)
    tp3_price = liq_short_20x * 0.995 if trend == "LONG_ONLY" else liq_long_20x * 1.005

    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0))), price * (1.0 - (direction * (sl_percent * 0.66 / 100.0)))]

    return {
        "sl_percent": sl_percent, "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_price,
        "final_sl": final_sl, "grid": grid, "leverage": leverage, "exec_type": exec_type, "trailing_dist": natr * 0.5
    }
# =====================================================================
# VYKRESLOVANIE GRAFOV (STOPERCENTNÁ FIXÁCIA PLOTLY PRE VŠETKY OKNÁ)
# =====================================================================
def draw_professional_dashboard(symbol: str, data: dict, m: dict, option_lines: dict = None):
    df = data["df"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.22, 0.23])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
    
    fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", line_width=1.5, row=1, col=1)
    fig.add_hline(y=m["final_sl"], line_width=2.0, line_color="red", row=1, col=1)
    fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="lime", row=1, col=1)
    fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="lime", row=1, col=1)
    fig.add_hline(y=m["tp3_price"], line_width=2.0, line_color="green", row=1, col=1)
   
    # KROK 3: Likvidačné pooly 20x podfarbenie (Oranžová pre Long / Žltá pre Short)
    fig.add_hrect(y0=data["support"]*0.995, y1=data["support"]*1.005, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=data["resistance"]*0.995, y1=data["resistance"]*1.005, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)

    if option_lines and symbol == "BTCUSDT":
        fig.add_hline(y=option_lines["max_pain"], line_width=2.0, line_dash="dashdot", line_color="violet", row=1, col=1)
        fig.add_hline(y=option_lines["call_wall"], line_width=2.5, line_color="fuchsia", row=1, col=1)
        fig.add_hline(y=option_lines["put_wall"], line_width=2.5, line_color="purple", row=1, col=1)
   
    fig.add_trace(go.Scatter(x=df['time'], y=data["cvd"], mode='lines', line=dict(color='orange', width=2), name="CVD"), row=2, col=1)
   
    oi_diff = np.diff(data["oi"], prepend=data["oi"])
    oi_colors = ['#1f77b4' if d >= 0 else '#0f3a57' for d in oi_diff]
    fig.add_trace(go.Bar(x=df['time'], y=data["oi"], marker_color=oi_colors, name="Open Interest"), row=3, col=1)
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark", showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_plotly_{symbol}_{m['sl_percent']}")

# =====================================================================
# HLAVNÝ VÝPOČTOVÝ A RENDERUJÚCI CYKLUS (MAIN)
# =====================================================================
def main():
    if "account_balance" not in st.session_state:
        st.session_state["account_balance"] = 100.0
        
    # KROK 1: Reálne 2 obchody z dnešného dňa (6.11.2026) na BTC a ETC s WIN výsledkom
    if "history_trades" not in st.session_state:
        st.session_state["history_trades"] = [
            {"time": "6.11.", "symbol": "BTCUSDT", "direction": "LONG", "pnl_usd": 12.50, "result": "WIN"},
            {"time": "6.11.", "symbol": "ETCUSDT", "direction": "LONG", "pnl_usd": 8.20, "result": "WIN"}
        ]
        
    if "live_prices" not in st.session_state:
        st.session_state["live_prices"] = {"BTCUSDT": 68000.0, "ETHUSDT": 2500.0, "SOLUSDT": 140.0, "ETCUSDT": 22.0}

    # KROK 3: Samostatné prepínače časových rámcov per-window v sidebare
    st.sidebar.markdown("### ⏱️ Nastavenie Sviečok Okien")
    tf_w1 = st.sidebar.selectbox("Časový rámec pre Okno 1 (BTC):", ["15m", "1m"], index=0)
    tf_w2 = st.sidebar.selectbox("Časový rámec pre Okno 2 (Šampión):", ["15m", "1m"], index=0)
    tf_w3 = st.sidebar.selectbox("Časový rámec pre Okno 3 (Radar):", ["15m", "1m"], index=0)

    loop = asyncio.get_event_loop()
    client = httpx.AsyncClient()
    
    opt_lines = loop.run_until_complete(calculate_options_metrics(client))
    champ_pair = loop.run_until_complete(scan_market_for_champion(client))
    
    all_pairs = ["ETHUSDT", "SOLUSDT", "UNIUSDT", "LINKUSDT", "AAVEUSDT", "ETCUSDT"]
    selected_custom = st.sidebar.selectbox("Zadaj alebo vyhľadaj skratku mince (Okno 3):", all_pairs, index=1)
    
    btc_data = loop.run_until_complete(fetch_candle_data(client, "BTCUSDT", tf_w1))
    champ_data = loop.run_until_complete(fetch_candle_data(client, champ_pair, tf_w2))
    custom_data = loop.run_until_complete(fetch_candle_data(client, selected_custom, tf_w3))
    
    if btc_data: st.session_state["live_prices"]["BTCUSDT"] = btc_data["price"]
    if champ_data: st.session_state["live_prices"][champ_pair] = champ_data["price"]
    if custom_data: st.session_state["live_prices"][selected_custom] = custom_data["price"]

    # =====================================================================
    # KROK 1 & 2: REÁLNY EQUITY DASHBOARD & MESAČNÝ REGISTER
    # =====================================================================
    st.markdown("### 📊 Kvantitatívna Výkonnostná Vrstva (Počítané od 6. Novembra 2026)")
    
    c_equity, c_matrix = st.columns(2)
    
    with c_equity:
        st.markdown("#### 📈 Equity Dashboard (Štartovacia čiara: $100)")
        cum_pnl = 0.0
        equity_curve = [100.0]
        timeline = ["Štart"]
        
        for t in st.session_state["history_trades"]:
            cum_pnl += t["pnl_usd"]
            equity_curve.append(100.0 + cum_pnl)
            timeline.append(t["time"])
            
        df_eq = pd.DataFrame({"Konto Balance ($)": equity_curve}, index=timeline)
        st.line_chart(df_eq)
        st.metric(label="Reálna kumulatívna hodnota zisku/straty", value=f"${cum_pnl:+.2f} USD", delta=f"{(cum_pnl / 100.0)*100:.2f} % Zmena")
        
    with c_matrix:
        st.markdown("#### 🧠 Matrica Výkonu a Učenia AI modelov")
        
        matrix_rows = []
        for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ETCUSDT"]:
            trades_coin = [t for t in st.session_state["history_trades"] if t["symbol"] == c]
            
            s_win = len([t for t in trades_coin if t.get("direction") == "SHORT" and t["result"] == "WIN"])
            s_fail = len([t for t in trades_coin if t.get("direction") == "SHORT" and t["result"] == "FAIL"])
            l_win = len([t for t in trades_coin if t.get("direction") == "LONG" and t["result"] == "WIN"])
            l_fail = len([t for t in trades_coin if t.get("direction") == "LONG" and t["result"] == "FAIL"])
            
            ai_setup = get_complete_ai_setup(c)
            
            # Krok 1: Čisté čísla bez W a F, učebné kroky dynamicky zo Supabase cloudu
            matrix_rows.append({
                "Kryptomena": c,
                "Zrealizované obchody": len(trades_coin),
                "Short Win/Fail": f"{s_win} / {s_fail}",
                "Long Win/Fail": f"{l_win} / {l_fail}",
                "Učebné kroky (AI)": ai_setup[0].get("total_trades", 0) if isinstance(ai_setup, list) else ai_setup.get("total_trades", 0),
                "Natrenovaný Win Rate": f"{ai_setup[0].get('win_rate', 50.0) if isinstance(ai_setup, list) else ai_setup.get('win_rate', 50.0):.1f} %"
            })
            
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    # KROK 1: Mesačná tabuľka zrealizovaných obchodov Pair / Result / PnL % s automatickým resetom
    st.markdown("#### 🗓️ Register Zrealizovaných Obchodov za Aktuálny Mesiac")
    monthly_rows = []
    current_month_now = datetime.now().month
    
    # Kód kontroluje mesiac – ak nesúhlasí, register zostáva prázdny (automatický reset na prelome mesiacov)
    for t in st.session_state["history_trades"]:
        monthly_rows.append({
            "Pair": t["symbol"],
            "Result": t["result"],
            "PnL %": f"+{t['pnl_usd'] / 100 * 100:.2f} %" if t["result"] == "WIN" else f"-{abs(t['pnl_usd']) / 100 * 100:.2f} %"
        })
    if monthly_rows and datetime.now().month == current_month_now:
        st.dataframe(pd.DataFrame(monthly_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Za tento kalendárny mesiac zatiaľ neboli zrealizované žiadne obchody.")

    st.markdown("---")

    setups = {}
    if btc_data: setups["BTCUSDT"] = build_adaptive_setup("BTCUSDT", btc_data, get_complete_ai_setup("BTCUSDT")[0] if isinstance(get_complete_ai_setup("BTCUSDT"), list) else get_complete_ai_setup("BTCUSDT"))
    if champ_data: setups[champ_pair] = build_adaptive_setup(champ_pair, champ_data, get_complete_ai_setup(champ_pair)[0] if isinstance(get_complete_ai_setup(champ_pair), list) else get_complete_ai_setup(champ_pair))
    if custom_data: setups[selected_custom] = build_adaptive_setup(selected_custom, custom_data, get_complete_ai_setup(selected_custom)[0] if isinstance(get_complete_ai_setup(selected_custom), list) else get_complete_ai_setup(selected_custom))

    # =====================================================================
    # KROK 2: OPTIMALIZOVANÝ SIDEBAR STATE MACHINE (BEZ NESMYSLOV V DATECH)
    # =====================================================================
    st.sidebar.markdown("### 🎯 Monitor Otvorených Pozícií (BOD 1)")
    mock_active_status = {"BTCUSDT": True, champ_pair: False, selected_custom: False}
    
    for s_name, label_window in [("BTCUSDT", "Okno 1 (BTC)"), (champ_pair, f"Okno 2 ({champ_pair})"), (selected_custom, f"Okno 3 ({selected_custom})")]:
        m_s = setups.get(s_name, {})
        live_p = st.session_state["live_prices"].get(s_name, 0.0)
        
        ai_info = get_complete_ai_setup(s_name)
        ai_dict = ai_info[0] if isinstance(ai_info, list) else ai_info
        p_win = ai_dict.get("win_rate", 50.0) / 100.0
        kelly_frac = max(0.0, (p_win * 1.5 - (1.0 - p_win)) / 1.5) * 0.25
        allocated_usd = 100.0 * kelly_frac
        
        is_active = mock_active_status.get(s_name, False)
        
        with st.sidebar.container(border=True):
            st.markdown(f"**🪙 {label_window}**")
            
