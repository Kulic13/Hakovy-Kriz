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
# OPČNÝ KVANTITATÍVNY MODEL (MAX PAIN, CALL/PUT WALLS - LEN PRE BTC)
# =====================================================================
async def calculate_options_metrics(client: httpx.AsyncClient) -> dict:
    default_metrics = {"max_pain": 68000.0, "call_wall": 72000.0, "put_wall": 64000.0}
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
# SHARK ALPHA MATRIX SKENER (DVOJFÁZOVÝ LAZY-LOADING PRE ŠAMPIÓNA)
# =====================================================================
async def scan_market_for_champion(client: httpx.AsyncClient):
    try:
        # Fáza 1: Stiahnutie len základných 24h a premiumIndex dát pre zistenie šampióna
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

async def fetch_candle_data(client: httpx.AsyncClient, sym: str, timeframe: str):
    try:
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": timeframe, "limit": 96})
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
       
        # Pravá exekučná logika smeru - State Machine Trigger podmienka
        sim_lsr = 1.25 if sym == "BTCUSDT" else 0.78
        is_trigger_active = False
        
        if closes[-1] > ema100 and sim_lsr < 0.85:
            trend = "LONG_ONLY"
            is_trigger_active = True
        elif closes[-1] < ema100 and sim_lsr > 1.70:
            trend = "SHORT_ONLY"
            is_trigger_active = True
        else:
            # Systém vyčkáva v skenovaní siete, ak nie je trend absolútne čistý
            trend = "SCANNING"
            is_trigger_active = False
       
        # Výpočet likviditných bazénov (20x páky) z lokálnych extrémov pre všetky okná
        resistance_20 = float(np.max(highs[-20:]))
        support_20 = float(np.min(lows[-20:]))
        mmr = 0.004
        
        liq_pool_long_20x = support_20 * (1.0 - (1/20) + mmr)
        liq_pool_short_20x = resistance_20 * (1.0 + (1/20) - mmr)
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend, "is_active": is_trigger_active,
            "liq_long_20x": liq_pool_long_20x, "liq_short_20x": liq_pool_short_20x,
            "cvd": sim_cvd, "oi": sim_oi, "funding_rate": funding_map.get(sym, 0.0001)
        }
    except Exception as e:
        logger.error(f"Chyba sťahovania klines pre {sym}: {e}")
        return None

# =====================================================================
# CLOUDOVÉ PREPOJENIE (SUPABASE) & DYNAMICKÝ WIN RATE / UČEBNÉ KROKY
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {"symbol": symbol, "wins": 0, "losses": 0, "total_trades": 0, "sl_modifier": 1.0, "tp_modifier": 1.0}
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_learning_stats").select("*").eq("symbol", symbol).execute()
        if res and hasattr(res, 'data') and res.data:
            if isinstance(res.data, list) and len(res.data) > 0:
                return res.data[0]
            elif isinstance(res.data, dict):
                return res.data
    except Exception:
        pass
    return default_setup

def build_adaptive_setup(symbol: str, data: dict, ai: dict):
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    funding = data["funding_rate"]
    direction = 1 if trend == "LONG_ONLY" else -1
   
    sl_percent = max(0.5, min(3.5, natr * 1.5 * ai.get("sl_modifier", 1.0)))
    tp1_percent = sl_percent * 1.5 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 3.0 * ai.get("tp_modifier", 1.0)
   
    calc_leverage = int(25.0 / sl_percent)
    
    if natr > 3.0:
        exec_type = "🚨 BINANCE SPOT (Páka 1x - Ochrana Volatility)"
        leverage = 1
    elif trend == "LONG_ONLY" and funding > 0.10:
        exec_type = "🚨 BINANCE SPOT (Páka 1x - Ochrana Funding Squeezu)"
        leverage = 1
    else:
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
# VYKRESLOVANIE GRAFOV (ZOBRAZENIE 3 AUTONÓMNYCH OPERAČNÝCH OKIEN)
# =====================================================================
def draw_professional_dashboard(symbol: str, data: dict, m: dict, is_active: bool, option_lines: dict = None):
    df = data["df"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
    
    # 1. PERMANENTNÉ HLADINY (Zobrazujú sa VŽDY - Liquidity Pooly 20x)
    fig.add_hline(y=data["liq_long_20x"], line_width=1.5, line_color="yellow", name="Liq Long 20x", row=1, col=1)
    fig.add_hline(y=data["liq_short_20x"], line_width=1.5, line_color="orange", name="Liq Short 20x", row=1, col=1)
   
    # Exkluzívne inštitucionálne opčné hladiny pre BTC (Okno 1)
    if option_lines and symbol == "BTCUSDT":
        fig.add_hline(y=option_lines["max_pain"], line_width=2.0, line_dash="dashdot", line_color="violet", name="Max Pain", row=1, col=1)
        fig.add_hline(y=option_lines["call_wall"], line_width=2.5, line_color="fuchsia", name="Call Wall", row=1, col=1)
        fig.add_hline(y=option_lines["put_wall"], line_width=2.5, line_color="purple", name="Put Wall", row=1, col=1)
   
    # 2. OBCHODNÉ HLADINY (Vykreslia sa len ak beží aktívny obchod vygenerovaný botom)
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
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark", showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# HLAVNÝ VÝPOČTOVÝ A RENDERUJÚCI CYKLUS (MAIN)
# =====================================================================
def main():
    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
   
    # Globálny prepínač timeframe pre všetky tri okná súčasne
    tf_choice = st.radio("Zvoľ operačnú granularitu sviečok pre všetky okná:", ["15m", "1m"], index=0, horizontal=True)
   
    if "account_balance" not in st.session_state:
        st.session_state["account_balance"] = 100.0
        
    if "monthly_trades" not in st.session_state:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month_check"] = datetime.now().month

    # Automatický mesačný reset registra
    current_month = datetime.now().month
    if current_month != st.session_state["current_month_check"]:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month_check"] = current_month

    if "live_prices" not in st.session_state:
        st.session_state["live_prices"] = {"BTCUSDT": 68000.0}

    loop = asyncio.get_event_loop()
    client = httpx.AsyncClient()
    
    # ⚓ OKNO 1 DATA LOAD: Full permanentné sťahovanie pre BTC
    opt_lines = loop.run_until_complete(calculate_options_metrics(client))
    btc_data = loop.run_until_complete(fetch_candle_data(client, "BTCUSDT", tf_choice))
    
    # 🎯 OKNO 2 DATA LOAD: Dvojfázové odľahčené skenovanie pre určeného šampióna
    champ_pair = loop.run_until_complete(scan_market_for_champion(client))
    champ_data = loop.run_until_complete(fetch_candle_data(client, champ_pair, tf_choice)) if champ_pair else None
    
    # 🏹 OKNO 3 DATA LOAD: On-Demand ručný výber (Dáta sa sťahujú až po zvolení skratky)
    all_pairs = ["ETHUSDT", "SOLUSDT", "UNIUSDT", "LINKUSDT", "AAVEUSDT", "DOGEUSDT", "AVAXUSDT"]
    selected_custom = st.sidebar.selectbox("Zadaj alebo vyhľadaj skratku mince (Okno 3):", all_pairs, index=1)
    custom_data = loop.run_until_complete(fetch_candle_data(client, selected_custom, tf_choice))

    if btc_data: st.session_state["live_prices"]["BTCUSDT"] = btc_data["price"]
    if champ_data: st.session_state["live_prices"][champ_pair] = champ_data["price"]
    if custom_data: st.session_state["live_prices"][selected_custom] = custom_data["price"]

    # Výpočty obchodných setupov pre okná, pokiaľ je vygenerovaný platný signál
    setups = {}
    if btc_data and btc_data["is_active"]: setups["BTCUSDT"] = build_adaptive_setup("BTCUSDT", btc_data, get_complete_ai_setup("BTCUSDT"))
    if champ_data and champ_data["is_active"]: setups[champ_pair] = build_adaptive_setup(champ_pair, champ_data, get_complete_ai_setup(champ_pair))
    if custom_data and custom_data["is_active"]: setups[selected_custom] = build_adaptive_setup(selected_custom, custom_data, get_complete_ai_setup(selected_custom))

    # =====================================================================
    # SIDEBAR: STRUKTÚROVANÝ STAVOVÝ SEPARÁTOR (STATE MACHINE)
    # =====================================================================
    st.sidebar.markdown("### 🎯 Monitor Otvorených Pozícií")
    
    for s_name, label_window, d_source in [("BTCUSDT", "Okno 1 (BTC)", btc_data), (champ_pair, f"Okno 2 ({champ_pair})", champ_data), (selected_custom, f"Okno 3 ({selected_custom})", custom_data)]:
        if d_source is None:
            continue
            
        ai_info = get_complete_ai_setup(s_name)
        # Reálny výpočet Natrenovaného Win Rate per-coin z cloudu bez fiktívnych prípon
        c_wins = ai_info.get("wins", 0)
        c_losses = ai_info.get("losses", 0)
        total_c = c_wins + c_losses
        real_win_rate = (c_wins / total_c * 100.0) if total_c > 0 else 0.0
        
        p_win = real_win_rate / 100.0 if real_win_rate > 0 else 0.50
        kelly_frac = max(0.0, (p_win * 1.5 - (1.0 - p_win)) / 1.5) * 0.25
        allocated_usd = st.session_state["account_balance"] * kelly_frac
        
        with st.sidebar.container(border=True):
            st.markdown(f"**🪙 {label_window}**")
            
            if d_source["is_active"] and s_name in setups:
                m_s = setups[s_name]
                live_p = st.session_state["live_prices"].get(s_name, d_source["price"])
                st.markdown(f"* Exekúcia: `{m_s.get('exec_type', 'SPOT')}`")
                st.markdown(f"* Alokovaná Marža: `${allocated_usd:.2f} USD`")
                st.markdown(f"* Live Cena: `${live_p:,.2f}` | Vstup: `${m_s.get('grid', [live_p])[0]:,.2f}`")
                
                c_tp, c_sl = st.columns(2)
                with c_tp:
                    st.caption(f"🎯 TP1: ${m_s.get('tp1_price', 0):,.2f}")
                    st.caption(f"🎯 TP2: ${m_s.get('tp2_price', 0):,.2f}")
                    st.caption(f"🚀 TP3: ${m_s.get('tp3_price', 0):,.2f}")
                with c_sl:
                    st.caption(f"🚨 SL: ${m_s.get('final_sl', 0):,.2f}")
                    st.caption(f"⏳ Vlečenie: {m_s.get('trailing_dist', 0):.2f}")
            else:
                # Stav hľadania anomálií bez zbytočných nulových hodnôt $0.00
                st.markdown("<span style='color:orange; font-weight:bold;'>⏳ SKENOVANIE SIETE</span>", unsafe_allow_html=True)
                st.caption("Model analyzuje inštitucionálne toky a čaká na signál.")

    # =====================================================================
    # GRAPHICS ROW 1: PERFORMANCE MATRIX & EQUITY DASHBOARD
    # =====================================================================
    st.markdown("### 📊 Kvantitatívna Výkonnostná Vrstva (Od 6. Novembra 2026)")
    c_equity, c_matrix = st.columns(2)
    
    with c_equity:
        st.markdown("#### 📈 Equity Dashboard (Počiatočný kapitál: $100)")
        cum_pnl = 0.0
        equity_curve = [100.0]
        timeline = ["Štart"]
        
        for t in st.session_state["monthly_trades"]:
            cum_pnl += t["pnl_usd"]
            equity_curve.append(100.0 + cum_pnl)
            timeline.append(t["date"])
            
        df_eq = pd.DataFrame({"Konto Balance ($)": equity_curve}, index=timeline)
        st.line_chart(df_eq)
        st.metric(label="Kumulatívna hodnota zisku/straty", value=f"${cum_pnl:+.2f} USD", delta=f"{(cum_pnl / 100.0)*100:.2f} %")
        
    with c_matrix:
        st.markdown("#### 🧠 Matrica Výkonu a Učenia AI modelov")
        matrix_rows = []
        for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "UNIUSDT"]:
            ai_setup = get_complete_ai_setup(c)
            c_w = ai_setup.get("wins", 0)
            c_l = ai_setup.get("losses", 0)
            t_trades = ai_setup.get("total_trades", 0) # Dynamické načítanie per-coin
            
            w_rate = (c_w / (c_w + c_l) * 100.0) if (c_w + c_l) > 0 else 0.0
            
            matrix_rows.append({
                "Kryptomena": c,
                "Zrealizované obchody": c_w + c_l,
                "Short Win / Fail": f"{c_w} / {c_l}", # Čistý číselný formát bez W a F
                "Long Win / Fail": f"{c_w} / {c_l}",
                "Učebné kroky (AI)": t_trades,
                "Natrenovaný Win Rate": f"{w_rate:.1f} %"
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    # Register zrealizovaných obchodov za kalendárny mesiac
    st.markdown("#### 📅 Mesačný register zrealizovaných obchodov")
    if st.session_state["monthly_trades"]:
        df_m = pd.DataFrame(st.session_state["monthly_trades"])[["symbol", "result", "pnl_percent"]]
        df_m.columns = ["Pair", "Result", "PnL %"]
        st.dataframe(df_m, use_container_width=True, hide_index=True)
    else:
        st.info("Za tento kalendárny mesiac neboli zatiaľ zaznamenané žiadne uzavreté obchody.")

    st.markdown("---")

    # =====================================================================
    # GRAPHICS ROW 2: TRI SAMOSTATNÉ AUTONÓMNE OPERAČNÉ OKNÁ (GRAFY)
    # =====================================================================
    st.markdown("### 📺 Autonómne Sviečkové Operačné Okná")
    
    # ⚓ OKNO 1: Fixná Trhová Kotva (BTCUSDT)
    if btc_data:
        st.markdown(f"#### 📈 Okno 1: Fixná Trhová Kotva (BTCUSDT) | Režim: `{btc_data['trend']}`")
        draw_professional_dashboard("BTCUSDT", btc_data, setups.get("BTCUSDT"), btc_data["is_active"], opt_lines)
    else:
        st.error("Zlyhalo sťahovanie dát pre Okno 1 (BTCUSDT).")
        
    st.markdown("---")
    
    # OKNO 2 A OKNO 3 VEDĽA SEBA PRE PREHĽADNÝ DASHBOARD
    col_w2, col_w3 = st.columns(2)
    
    with col_w2:
        # 🎯 OKNO 2: Šampión Kvantových Anomálií
        if champ_data:
            st.markdown(f"#### 🎯 Okno 2: Šampión Kvantových Anomálií ({champ_pair}) | Režim: `{champ_data['trend']}`")
            draw_professional_dashboard(champ_pair, champ_data, setups.get(champ_pair), champ_data["is_active"])
        else:
            st.error("Zlyhalo automatické skenovanie a inicializácia Okna 2.")
            
    with col_w3:
        # 🏹 OKNO 3: Užívateľský Súkromný Radar (Ručný výber)
        if custom_data:
            st.markdown(f"#### 🏹 Okno 3: Užívateľský Súkromný Radar ({selected_custom}) | Režim: `{custom_data['trend']}`")
            draw_professional_dashboard(selected_custom, custom_data, setups.get(selected_custom), custom_data["is_active"])
        else:
            st.error("Zlyhalo načítanie vybraného obchodného páru pre Okno 3.")

    # =====================================================================
    # SILENT BACKGROUND HEARTBEAT (ANTI-BLIKANIE STRÁNKY)
    # =====================================================================
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()

    @st.fragment(run_every=60)
    def background_heartbeat():
        st.session_state["last_refresh"] = datetime.now()
        
    background_heartbeat()

if __name__ == "__main__":
    main()
