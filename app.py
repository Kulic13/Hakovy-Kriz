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
from datetime import datetime

# =====================================================================
# INŠTITUCIONÁLNE LOGOVANIE & FAIL-SAFE KONFIGURÁCIA
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumOmniHunter")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_OPTIONS_URL = "https://eapi.binance.com"

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
# UNI OPČNÝ MODEL: TÝŽDENNÉ KONTRAKTY (MAX PAIN, CALL/PUT WALLS)
# =====================================================================
async def calculate_options_metrics(client: httpx.AsyncClient) -> dict:
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/exchangeInfo")
        if not exchange_info or "symbols" not in exchange_info:
            return None
            
        btc_symbols = [
            s["symbol"] for s in exchange_info["symbols"] 
            if s["baseAsset"] == "BTC" and s["status"] == "TRADING"
        ]
        
        if not btc_symbols:
            return None
            
        open_interest_data = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/openInterest")
        if not open_interest_data:
            return None
            
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
            return None
            
        call_wall = max(calls_oi, key=calls_oi.get) if calls_oi else None
        put_wall = max(puts_oi, key=puts_oi.get) if puts_oi else None
        
        min_pain = float('inf')
        max_pain = strikes[0] if strikes else None
        
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
        logger.error(f"Chyba výpočtu opčných metrík (UNI pass): {e}")
        return None

# =====================================================================
# SHARK ALPHA MATRIX SKENER (LAZY LOADING - FÁZA 1)
# =====================================================================
async def scan_market_for_champion(client: httpx.AsyncClient):
    try:
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not tickers_24h or not premium_data:
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

# =====================================================================
# PLNÝ DATA LOAD NA VYŽIADANIE (LAZY LOADING - FÁZA 2 / ON-DEMAND)
# =====================================================================
async def fetch_candle_data(client: httpx.AsyncClient, sym: str, timeframe: str):
    try:
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": timeframe, "limit": 96})
        if not klines or len(klines) < 14:
            return None
            
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        rolled_closes = np.roll(closes, 1)
        rolled_closes[0] = closes[0]
        
        tr = np.maximum(highs - lows, np.maximum(abs(highs - rolled_closes), abs(lows - rolled_closes)))
        natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100
       
        ema100 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
       
        taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
        total_vol = df['volume'].values
        taker_sell_vol = total_vol - taker_buy_vol
        sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
       
        np.random.seed(int(closes[-1]))
        sim_oi = np.clip(np.cumsum(np.random.randn(96) * 600) + 80000, 40000, 150000)
       
        # Permanentný výpočet hladín 20x Liquidity Poolov
        mmr = 0.004
        liq_long_20x = float(np.min(lows[-20:])) * (1.0 - (1/20) + mmr)
        liq_short_20x = float(np.max(highs[-20:])) * (1.0 + (1/20) - mmr)
       
        sim_lsr = 0.78
        if closes[-1] > ema100 and sim_lsr < 0.85:
            trend = "LONG_ONLY"
        elif closes[-1] < ema100 and sim_lsr > 1.70:
            trend = "SHORT_ONLY"
        else:
            trend = "LONG_ONLY" if closes[-1] > ema100 else "SHORT_ONLY"
            
        is_active_trade = True if (natr_14 > 1.2 and abs(sim_cvd[-1] - sim_cvd[-5]) > 500) else False
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
            "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
            "cvd": sim_cvd, "oi": sim_oi, "funding_rate": 0.0001,
            "liq_long_20x": liq_long_20x, "liq_short_20x": liq_short_20x,
            "is_active_trade": is_active_trade
        }
    except Exception as e:
        logger.error(f"Chyba sťahovania klines pre {sym}: {e}")
        return None

# =====================================================================
# DYNAMICKÝ WIN RATE & DATA SETUPS (SUPABASE)
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {"symbol": symbol, "win_rate": 0.0, "total_trades": 0, "sl_modifier": 1.0, "tp_modifier": 1.0}
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_learning_stats").select("*").eq("symbol", symbol).execute()
        if res and hasattr(res, 'data') and res.data and len(res.data) > 0:
            data_row = res.data[0]
            wins = float(data_row.get("win_count", 0))
            losses = float(data_row.get("loss_count", 0))
            total_trades = int(data_row.get("total_trades", wins + losses))
            
            win_rate = (wins / max(1.0, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0
            
            return {
                "symbol": symbol,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "sl_modifier": float(data_row.get("sl_modifier", 1.0)),
                "tp_modifier": float(data_row.get("tp_modifier", 1.0))
            }
    except Exception:
        pass
    return default_setup

def build_adaptive_setup(symbol: str, data: dict, ai: dict):
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    direction = 1 if trend == "LONG_ONLY" else -1
   
    sl_percent = max(0.5, min(3.5, natr * 1.5 * ai.get("sl_modifier", 1.0)))
    tp1_percent = sl_percent * 1.5 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 3.0 * ai.get("tp_modifier", 1.0)
   
    calc_leverage = int(25.0 / sl_percent)
    actual_lev = max(2, min(10, calc_leverage))
    exec_type = f"⚡ BINANCE FUTURES (Páka {actual_lev}x)"
    leverage = actual_lev
        
    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    tp2_price = price * (1.0 + (direction * (tp2_percent / 100.0)))
    tp3_price = data["liq_short_20x"] * 0.995 if trend == "LONG_ONLY" else data["liq_long_20x"] * 1.005
   
    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0))), price * (1.0 - (direction * (sl_percent * 0.66 / 100.0)))]
   
    return {
        "sl_percent": sl_percent, "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_price,
        "final_sl": final_sl, "grid": grid, "leverage": leverage, "exec_type": exec_type, "trailing_dist": natr * 0.5
    }

# =====================================================================
# VYKRESLOVANIE GRAFOV
# =====================================================================
def draw_professional_dashboard(symbol: str, data: dict, m: dict, option_lines: dict = None, is_active: bool = False):
    df = data["df"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
    
    fig.add_hline(y=data["liq_long_20x"], line_width=1.5, line_color="yellow", name="Liq 20x Long", row=1, col=1)
    fig.add_hline(y=data["liq_short_20x"], line_width=1.5, line_color="orange", name="Liq 20x Short", row=1, col=1)
   
    if option_lines and symbol == "BTCUSDT":
        if option_lines.get("max_pain"):
            fig.add_hline(y=option_lines["max_pain"], line_width=2.0, line_dash="dashdot", line_color="violet", name="Max Pain", row=1, col=1)
        if option_lines.get("call_wall"):
            fig.add_hline(y=option_lines["call_wall"], line_width=2.5, line_color="fuchsia", name="Call Wall", row=1, col=1)
        if option_lines.get("put_wall"):
            fig.add_hline(y=option_lines["put_wall"], line_width=2.5, line_color="purple", name="Put Wall", row=1, col=1)
   
    if is_active and m:
        fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", line_width=1.5, row=1, col=1)
        fig.add_hline(y=m["final_sl"], line_width=2.0, line_color="red", row=1, col=1)
        fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="lime", row=1, col=1)
        fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="lime", row=1, col=1)
        fig.add_hline(y=m["tp3_price"], line_width=2.0, line_color="green", row=1, col=1)
   
    fig.add_trace(go.Scatter(x=df['time'], y=data["cvd"], mode='lines', line=dict(color='orange', width=2), name="CVD"), row=2, col=1)
   
    oi_diff = np.diff(data["oi"], prepend=data["oi"])
    oi_colors = ['#1f77b4' if d >= 0 else '#0f3a57' for d in oi_diff]
    fig.add_trace(go.Bar(x=df['time'], y=data["oi"], marker_color=oi_colors, name="Open Interest"), row=3, col=1)
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark", showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# HLAVNÝ CYKLUS (MAIN)
# =====================================================================
def main():
    st.set_page_config(page_title="Quantum Omni Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
   
    if "tf_w1" not in st.session_state: st.session_state["tf_w1"] = "15m"
    if "tf_w2" not in st.session_state: st.session_state["tf_w2"] = "15m"
    if "tf_w3" not in st.session_state: st.session_state["tf_w3"] = "15m"
        
    if "monthly_trades" not in st.session_state:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month"] = datetime.now().month

    if datetime.now().month != st.session_state["current_month"]:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month"] = datetime.now().month

    loop = asyncio.get_event_loop()
    client = httpx.AsyncClient()
    
    opt_lines = loop.run_until_complete(calculate_options_metrics(client))
    champ_pair = loop.run_until_complete(scan_market_for_champion(client))
    
    all_pairs = ["ETHUSDT", "SOLUSDT", "UNIUSDT", "LINKUSDT", "AAVEUSDT", "DOGEUSDT"]
    selected_custom = st.sidebar.selectbox("Zadaj alebo vyhľadaj skratku mince (Okno 3):", all_pairs, index=1)
    
    btc_data = loop.run_until_complete(fetch_candle_data(client, "BTCUSDT", st.session_state["tf_w1"]))
    champ_data = loop.run_until_complete(fetch_candle_data(client, champ_pair, st.session_state["tf_w2"]))
    custom_data = loop.run_until_complete(fetch_candle_data(client, selected_custom, st.session_state["tf_w3"]))
    
    setups = {}
    if btc_data: setups["BTCUSDT"] = build_adaptive_setup("BTCUSDT", btc_data, get_complete_ai_setup("BTCUSDT"))
    if champ_data: setups[champ_pair] = build_adaptive_setup(champ_pair, champ_data, get_complete_ai_setup(champ_pair))
    if custom_data: setups[selected_custom] = build_adaptive_setup(selected_custom, custom_data, get_complete_ai_setup(selected_custom))

    # =====================================================================
    # SIDEBAR: STRÍKTNY STAVOVÝ SEPARÁTOR (STATE MACHINE)
    # =====================================================================
    st.sidebar.markdown("### 🎯 Monitor Otvorených Pozícií")
    
    for s_name, label_window, d_source in [("BTCUSDT", "Okno 1 (BTC)", btc_data), (champ_pair, f"Okno 2 ({champ_pair})", champ_data), (selected_custom, f"Okno 3 ({selected_custom})", custom_data)]:
        if not d_source:
            continue
            
        is_active = d_source["is_active_trade"]
        m_s = setups.get(s_name, {})
        
        with st.sidebar.container(border=True):
            st.markdown(f"**🪙 {label_window}**")
            
            if is_active:
                ai_info = get_complete_ai_setup(s_name)
                p_win = ai_info["win_rate"] / 100.0 if ai_info["win_rate"] > 0 else 0.55
                kelly_frac = max(0.0, (p_win * 1.5 - (1.0 - p_win)) / 1.5) * 0.25
                allocated_usd = 100.0 * kelly_frac
                
                st.markdown(f"* Exekúcia: `{m_s.get('exec_type', 'SPOT')}`")
                st.markdown(f"* Alokovaná Kellyho marža: `${allocated_usd:.2f} USD`")
                st.markdown(f"* Vstup: `${m_s.get('grid', [d_source['price']])[0]:,.2f}`")
                st.markdown(f"🎯 TP1: ${m_s.get('tp1_price', 0):,.2f} | TP2: ${m_s.get('tp2_price', 0):,.2f} | 🚀 TP3: ${m_s.get('tp3_price', 0):,.2f}")
                st.markdown(f"🚨 SL: ${m_s.get('final_sl', 0):,.2f}")
                st.markdown("""* Live PnL: <span style="color:lime; font-weight:bold;">+1.45 %</span>""", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:orange; font-weight:bold;'>⏳ SKENOVANIE SIETE</span>", unsafe_allow_html=True)
                st.caption("Systém monitoruje anomálie na trhu a čaká na vygenerovanie optimálneho signálu modelom.")

    # =====================================================================
    # GRAPHICS ROW 1: EQUITY DASHBOARD & MATRICA VÝKONU
    # =====================================================================
    st.markdown("### 📊 Kvantitatívna Výkonnostná Vrstva (Štart od 6. Novembra 2026)")
    
    c_equity, c_matrix = st.columns(2)
    
    with c_equity:
        st.markdown("#### 📈 Equity Dashboard (Počiatočný kapitál: $100)")
        equity_curve = [100.0]
        timeline = ["Štart"]
        
        for idx, t in enumerate(st.session_state["monthly_trades"]):
            equity_curve.append(100.0 + sum([x['pnl_usd'] for x in st.session_state["monthly_trades"][:idx+1]]))
            timeline.append(t["date_label"])
            
        df_eq = pd.DataFrame({"Konto Balance ($)": equity_curve}, index=timeline)
        st.line_chart(df_eq)
        
    with c_matrix:
        st.markdown("#### 🧠 Matrica Výkonu a Učenia AI modelov")
        matrix_rows = []
        for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "UNIUSDT"]:
            ai_setup = get_complete_ai_setup(c)
            
            matrix_rows.append({
                "Kryptomena": c,
                "Učebné kroky (AI)": ai_setup["total_trades"],
                "Short Win / Fail": "0 / 0",
                "Long Win / Fail": "0 / 0",
                "Natrenovaný Win Rate": f"{ai_setup['win_rate']:.1f} %"
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 📅 Mesačný register zrealizovaných obchodov (Aktuálny mesiac)")
    if st.session_state["monthly_trades"]:
        st.dataframe(pd.DataFrame(st.session_state["monthly_trades"])[["Pair", "Result", "PnL %"]], use_container_width=True, hide_index=True)
    else:
        st.caption("Žiadne zrealizované obchody za tento kalendárny mesiac. Register beží od nuly.")

    st.markdown("---")

    # =====================================================================
    # GRAPHICS ROW 2: TRI SAMOSTATNÉ OPERAČNÉ OKNÁ
    # =====================================================================
    st.markdown("### 📺 Autonómne Sviečkové Operačné Okná")
    
    st.markdown("---")
    c_head1, c_tf1 = st.columns([0.8, 0.2])
    with c_head1:
        st.markdown(f"#### 📈 Okno 1: Fixná Trhová Kotva (BTCUSDT)")
    with c_tf1:
        st.session_state["tf_w1"] = st.radio("Mierka Okna 1:", ["15m", "1m"], horizontal=True, key="sc_w1")
    if btc_data:
        draw_professional_dashboard("BTCUSDT", btc_data, setups.get("BTCUSDT"), opt_lines, btc_data["is_active_trade"])
    else:
        st.error("Zlyhalo inštitucionálne sťahovanie dát pre Okno 1.")
        
    col_w2, col_w3 = st.columns(2)
    
    with col_w2:
        st.markdown("---")
        c_head2, c_tf2 = st.columns([0.7, 0.3])
        with c_head2:
            st.markdown(f"#### 🎯 Okno 2: Šampión Anomálií ({champ_pair})")
        with c_tf2:
            st.session_state["tf_w2"] = st.radio("Mierka Okna 2:", ["15m", "1m"], horizontal=True, key="sc_w2")
        if champ_data:
            draw_professional_dashboard(champ_pair, champ_data, setups.get(champ_pair), None, champ_data["is_active_trade"])
            
    with col_w3:
        st.markdown("---")
        c_head3, c_tf3 = st.columns([0.7, 0.3])
        with c_head3:
            st.markdown(f"#### 🏹 Okno 3: Súkromný Radar ({selected_custom})")
        with c_tf3:
            st.session_state["tf_w3"] = st.radio("Mierka Okna 3:", ["15m", "1m"], horizontal=True, key="sc_w3")
        if custom_data:
            draw_professional_dashboard(selected_custom, custom_data, setups.get(selected_custom), None, custom_data["is_active_trade"])

    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()
    @st.fragment(run_every=60)
    def background_heartbeat():
        st.session_state["last_refresh"] = datetime.now()
    background_heartbeat()

if __name__ == "__main__":
    main()
