#!/usr/bin/env python3
"""
Pre-Market + Intraday Trade Bot
Phase 1 (7:00–9:30 AM ET): broad pre-market news sweep → BUY signals
Phase 2 (9:30 AM–3:30 PM ET): intraday major-catalyst scanner → BUY signals
Position monitor runs all day → auto-sells on take-profit, stop-loss, or EOD
Uses Groq (free Llama 3) — no Anthropic credits needed.
"""

import os
import json
import time
import sqlite3
import hashlib
import threading
import feedparser
import requests
from groq import Groq
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv(Path(__file__).parent / ".env")

# ── Config (edit in .env) ──────────────────────────────────────────────────────
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "5.0"))
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT",   "2.0"))
TRADE_VALUE     = float(os.getenv("TRADE_VALUE",    "25.0"))
POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL",      "60"))
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL",      "")

ET              = ZoneInfo("America/New_York")
PREMARKET_START = (7,  0)   # begin scraping
MARKET_OPEN     = (9, 30)   # pre-market ends, intraday begins
INTRADAY_END    = (15, 30)  # stop new buys 30 min before close
EOD_CLOSE       = (15, 55)  # close all positions

FEEDS = {
    "Wall Street Journal":  "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "NY Times Business":    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "CNBC":                 "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch":          "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Guardian Business":    "https://www.theguardian.com/uk/business/rss",
    "Yahoo Finance":        "https://finance.yahoo.com/news/rssindex",
    "Bloomberg Markets":    "https://feeds.bloomberg.com/markets/news.rss",
}

RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

SEEN_PATH = Path(__file__).parent / "seen_articles.json"
DB_PATH   = Path(__file__).parent / "trades.db"

ALLOWED_ASSETS = """
- US-listed individual stocks (e.g. AAPL, NVDA, XOM)
- US-listed ETFs (e.g. SPY, QQQ, XLE, XLF, IWM)
- Gold: GLD or IAU only (no futures)
- Oil: USO or XLE only (no futures)
Do NOT recommend crypto, bonds, forex, or futures.
"""

# Pre-market prompt: broad sweep, any clearly bullish story
PREMARKET_PROMPT = f"""You are a pre-market financial analyst (7:00–9:30 AM ET).
Assess whether this headline is a HIGH-confidence BUY opportunity for a US-listed stock or ETF.

Tradeable assets:
{ALLOWED_ASSETS}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
    "ticker": "SYMBOL",
    "action": "BUY" | "SKIP",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "one sentence"
}}

Rules:
- Return BUY/HIGH only when news is clearly and directly bullish for a specific ticker.
- Prefer ETFs for macro themes. Use individual stocks only for company-specific news.
- If uncertain, negative, or not market-relevant: return SKIP.
"""

# Intraday prompt: stricter — only major market-shifting catalysts
INTRADAY_PROMPT = f"""You are an intraday trading analyst monitoring live market news (9:30 AM–3:30 PM ET).
Your job is to identify MAJOR market-shifting catalysts only. The bar is much higher than pre-market.

A qualifying catalyst must be ONE of:
- Federal Reserve statement, rate decision, or surprise commentary
- Major earnings surprise (significant beat or miss vs. expectations)
- Surprise economic data release (CPI, jobs, GDP, PPI — materially off consensus)
- Geopolitical shock with direct and immediate market impact
- Large unexpected M&A announcement (hostile takeover, merger of equals)
- Regulatory decision with major financial impact (FDA approval/rejection, FTC ruling)

Do NOT trigger on: analyst upgrades/downgrades, routine company news, market commentary, opinion pieces, or anything that was already known pre-market.

Tradeable assets:
{ALLOWED_ASSETS}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
    "ticker": "SYMBOL",
    "action": "BUY" | "SKIP",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "catalyst_type": "FED" | "EARNINGS" | "ECON_DATA" | "GEOPOLITICAL" | "M&A" | "REGULATORY" | "NONE",
    "reasoning": "one sentence"
}}

Rules:
- Return BUY/HIGH ONLY for genuine major catalysts. When in doubt, SKIP.
- Do not recommend a ticker already held as an open position.
"""

# ── Discord ────────────────────────────────────────────────────────────────────

def discord(embed: dict):
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"embeds": [embed]}, timeout=8).raise_for_status()
    except Exception as e:
        print(f"  [Discord WARN] {e}")

def notify_buy(ticker: str, qty: float, price: float, headline: str, reasoning: str, phase: str, catalyst_type: str = ""):
    phase_label = "📰 Pre-Market Signal" if phase == "premarket" else f"⚡ Intraday Catalyst — {catalyst_type}"
    discord({
        "title":     f"🟢 BUY {ticker} — Paper Trade Executed",
        "color":     0x00C805,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer":    {"text": phase_label},
        "fields": [
            {"name": "Ticker",    "value": f"`{ticker}`",                                              "inline": True},
            {"name": "Price",     "value": f"${price:.2f}",                                            "inline": True},
            {"name": "Invested",  "value": f"${TRADE_VALUE:.2f}",                                      "inline": True},
            {"name": "Shares",    "value": f"{qty:.6f}",                                               "inline": True},
            {"name": "TP Target", "value": f"${price*(1+TAKE_PROFIT_PCT/100):.2f} (+{TAKE_PROFIT_PCT}%)", "inline": True},
            {"name": "SL Target", "value": f"${price*(1-STOP_LOSS_PCT/100):.2f} (-{STOP_LOSS_PCT}%)",    "inline": True},
            {"name": "Reasoning", "value": reasoning,       "inline": False},
            {"name": "Headline",  "value": headline[:200],  "inline": False},
        ],
    })

def notify_sell(ticker: str, price: float, pnl_pct: float, pnl_dollar: float, reason: str):
    labels = {
        "TAKE_PROFIT": "🎯 Take Profit Hit",
        "STOP_LOSS":   "🛑 Stop Loss Hit",
        "EOD":         "🕓 End of Day Close",
    }
    win   = pnl_pct >= 0
    color = 0x00C805 if win else 0xFF3B30
    discord({
        "title":     f"{'📈' if win else '📉'} SELL {ticker} — {labels.get(reason, reason)}",
        "color":     color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fields": [
            {"name": "Ticker",     "value": f"`{ticker}`",              "inline": True},
            {"name": "Exit Price", "value": f"${price:.2f}",            "inline": True},
            {"name": "Result",     "value": "WIN" if win else "LOSS",   "inline": True},
            {"name": "Return",     "value": f"{pnl_pct:+.2f}%",        "inline": True},
            {"name": "P&L",        "value": f"${pnl_dollar:+.2f}",     "inline": True},
            {"name": "Reason",     "value": labels.get(reason, reason), "inline": True},
        ],
    })

# ── Database ───────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT NOT NULL,
            phase         TEXT,
            opened_at     TEXT,
            closed_at     TEXT,
            source        TEXT,
            headline      TEXT,
            ticker        TEXT NOT NULL,
            qty           REAL,
            entry_price   REAL,
            exit_price    REAL,
            pnl_pct       REAL,
            pnl_dollar    REAL,
            close_reason  TEXT,
            alpaca_buy_id TEXT,
            reasoning     TEXT,
            catalyst_type TEXT,
            status        TEXT DEFAULT 'OPEN'
        )
    """)
    conn.commit()
    conn.close()

def log_open(article: dict, rec: dict, qty: float, entry_price: float, order_id: str, phase: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO trades
            (date, phase, opened_at, source, headline, ticker, qty, entry_price, alpaca_buy_id, reasoning, catalyst_type, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        datetime.now(ET).strftime("%Y-%m-%d"),
        phase,
        datetime.now(timezone.utc).isoformat(),
        article["source"],
        article["title"],
        rec["ticker"],
        qty,
        entry_price,
        order_id,
        rec.get("reasoning", ""),
        rec.get("catalyst_type", ""),
    ))
    conn.commit()
    conn.close()

def log_close(ticker: str, exit_price: float, reason: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, entry_price, qty FROM trades WHERE ticker=? AND status='OPEN' ORDER BY opened_at DESC LIMIT 1",
        (ticker,)
    ).fetchone()
    if row:
        trade_id, entry_price, qty = row
        pnl_pct    = (exit_price - entry_price) / entry_price * 100
        pnl_dollar = qty * (exit_price - entry_price)
        conn.execute("""
            UPDATE trades SET closed_at=?, exit_price=?, pnl_pct=?, pnl_dollar=?, close_reason=?, status='CLOSED'
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), exit_price, pnl_pct, pnl_dollar, reason, trade_id))
        conn.commit()
    conn.close()

# ── Time helpers ───────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(ET)

def in_premarket_window() -> bool:
    t = now_et()
    s = t.replace(hour=PREMARKET_START[0], minute=PREMARKET_START[1], second=0, microsecond=0)
    e = t.replace(hour=MARKET_OPEN[0],     minute=MARKET_OPEN[1],     second=0, microsecond=0)
    return s <= t < e

def in_intraday_window() -> bool:
    t = now_et()
    s = t.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],  second=0, microsecond=0)
    e = t.replace(hour=INTRADAY_END[0], minute=INTRADAY_END[1], second=0, microsecond=0)
    return s <= t < e

def is_eod() -> bool:
    t = now_et()
    return t.hour > EOD_CLOSE[0] or (t.hour == EOD_CLOSE[0] and t.minute >= EOD_CLOSE[1])

def secs_until(hour: int, minute: int) -> float:
    t      = now_et()
    target = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
    diff   = (target - t).total_seconds()
    return max(diff, 0)

# ── RSS helpers ────────────────────────────────────────────────────────────────

def load_seen() -> set:
    if SEEN_PATH.exists():
        return set(json.loads(SEEN_PATH.read_text()))
    return set()

def save_seen(seen: set):
    SEEN_PATH.write_text(json.dumps(list(seen)))

def article_id(entry) -> str:
    key = entry.get("id") or entry.get("link") or entry.get("title") or ""
    return hashlib.sha256(key.encode()).hexdigest()

def fetch_feeds() -> list[dict]:
    articles = []
    for source, url in FEEDS.items():
        try:
            r    = requests.get(url, headers=RSS_HEADERS, timeout=10)
            r.raise_for_status()
            feed = feedparser.parse(r.content)
            for entry in feed.entries:
                articles.append({
                    "id":     article_id(entry),
                    "source": source,
                    "title":  entry.get("title", "").strip(),
                    "link":   entry.get("link", ""),
                })
        except Exception as e:
            print(f"  [WARN] {source}: {e}")
    return articles

# ── Groq recommendation ────────────────────────────────────────────────────────

def get_recommendation(groq_client: Groq, headline: str, source: str,
                       session_tickers: set, system_prompt: str) -> dict:
    already_held = ", ".join(session_tickers) if session_tickers else "none"
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"Source: {source}\n"
                f"Headline: {headline}\n"
                f"Tickers already held (do not recommend): {already_held}"
            )},
        ],
        max_tokens=150,
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

# ── Alpaca trading ─────────────────────────────────────────────────────────────

def place_buy(alpaca: TradingClient, ticker: str) -> tuple[float, float, str] | None:
    try:
        order = alpaca.submit_order(MarketOrderRequest(
            symbol=ticker,
            notional=TRADE_VALUE,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        time.sleep(2)
        filled = alpaca.get_order_by_id(str(order.id))
        return float(filled.filled_qty or 0), float(filled.filled_avg_price or 0), str(order.id)
    except Exception as e:
        print(f"  [ERROR] Buy {ticker}: {e}")
        return None

def place_sell(alpaca: TradingClient, ticker: str, qty: float) -> float | None:
    try:
        order = alpaca.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        time.sleep(2)
        filled = alpaca.get_order_by_id(str(order.id))
        return float(filled.filled_avg_price or 0)
    except Exception as e:
        print(f"  [ERROR] Sell {ticker}: {e}")
        return None

def open_position_tickers(alpaca: TradingClient) -> set:
    try:
        return {p.symbol for p in alpaca.get_all_positions()}
    except Exception:
        return set()

# ── Position monitor ───────────────────────────────────────────────────────────

def monitor_positions(alpaca: TradingClient):
    print(f"  [Monitor] Started — TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}% | EOD: {EOD_CLOSE[0]}:{EOD_CLOSE[1]:02d} ET")
    while True:
        try:
            if is_eod():
                positions = alpaca.get_all_positions()
                if positions:
                    print(f"\n[{now_et():%H:%M:%S}] EOD — closing {len(positions)} position(s)…")
                    for pos in positions:
                        ticker = pos.symbol
                        qty    = float(pos.qty)
                        pct    = float(pos.unrealized_plpc) * 100
                        price  = place_sell(alpaca, ticker, qty)
                        if price:
                            pnl_dollar = qty * (price - float(pos.avg_entry_price))
                            log_close(ticker, price, "EOD")
                            notify_sell(ticker, price, pct, pnl_dollar, "EOD")
                            print(f"  [EOD] {ticker}: {pct:+.2f}%")
                print(f"[{now_et():%H:%M:%S}] EOD complete. Monitor shutting down.")
                break

            for pos in alpaca.get_all_positions():
                ticker = pos.symbol
                qty    = float(pos.qty)
                pct    = float(pos.unrealized_plpc) * 100

                if pct >= TAKE_PROFIT_PCT:
                    print(f"\n  [TP] {ticker} @ {pct:+.2f}% — selling…")
                    price = place_sell(alpaca, ticker, qty)
                    if price:
                        pnl_dollar = qty * (price - float(pos.avg_entry_price))
                        log_close(ticker, price, "TAKE_PROFIT")
                        notify_sell(ticker, price, pct, pnl_dollar, "TAKE_PROFIT")
                        print(f"  [SOLD] {ticker} @ ${price:.2f} | {pct:+.2f}%")

                elif pct <= -STOP_LOSS_PCT:
                    print(f"\n  [SL] {ticker} @ {pct:+.2f}% — selling…")
                    price = place_sell(alpaca, ticker, qty)
                    if price:
                        pnl_dollar = qty * (price - float(pos.avg_entry_price))
                        log_close(ticker, price, "STOP_LOSS")
                        notify_sell(ticker, price, pct, pnl_dollar, "STOP_LOSS")
                        print(f"  [SOLD] {ticker} @ ${price:.2f} | {pct:+.2f}%")

        except Exception as e:
            print(f"  [Monitor ERROR] {e}")

        time.sleep(30)

# ── Shared trade loop ──────────────────────────────────────────────────────────

def run_scan_loop(groq_client, alpaca, seen, session_tickers, system_prompt, phase, window_fn):
    label = "Pre-Market" if phase == "premarket" else "Intraday"
    print(f"[{now_et():%H:%M:%S}] ── {label} scanner active ──")

    while window_fn():
        print(f"[{now_et():%H:%M:%S}] Polling…")
        articles = fetch_feeds()
        new_ones = [a for a in articles if a["id"] not in seen and a["title"]]
        print(f"  {len(new_ones)} new article(s)")

        # For intraday also check live positions so we don't double-buy
        held = open_position_tickers(alpaca) if phase == "intraday" else set()
        skip_tickers = session_tickers | held

        for article in new_ones:
            try:
                rec = get_recommendation(groq_client, article["title"], article["source"],
                                         skip_tickers, system_prompt)

                if rec.get("action") != "BUY" or rec.get("confidence") != "HIGH":
                    print(f"  → Skip ({rec.get('action','?')}/{rec.get('confidence','?')}): {article['title'][:60]}")
                    seen.add(article["id"])
                    time.sleep(0.5)
                    continue

                ticker = rec["ticker"].upper().strip()
                if ticker in skip_tickers:
                    print(f"  → Skip (already holding {ticker})")
                    seen.add(article["id"])
                    continue

                catalyst = rec.get("catalyst_type", "")
                print(f"  → {label} BUY {ticker}{f' [{catalyst}]' if catalyst else ''} | {article['title'][:55]}")
                result = place_buy(alpaca, ticker)
                if result:
                    qty, price, order_id = result
                    log_open(article, rec, qty, price, order_id, phase)
                    session_tickers.add(ticker)
                    notify_buy(ticker, qty, price, article["title"], rec.get("reasoning", ""), phase, catalyst)
                    print(f"  ✓ Bought {qty:.6f} {ticker} @ ${price:.2f}")

            except Exception as e:
                print(f"  [ERROR] {article['title'][:55]}: {e}")

            seen.add(article["id"])
            time.sleep(1)

        save_seen(seen)
        print(f"  Sleeping {POLL_INTERVAL}s…")
        time.sleep(POLL_INTERVAL)

    print(f"[{now_et():%H:%M:%S}] {label} window closed.")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_db()

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    alpaca      = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )

    acct = alpaca.get_account()
    print(f"[{now_et():%H:%M:%S}] Pre-Market + Intraday Trade Bot")
    print(f"  Paper account: ${float(acct.cash):,.2f} cash")
    print(f"  TP: +{TAKE_PROFIT_PCT}%  |  SL: -{STOP_LOSS_PCT}%  |  ${TRADE_VALUE}/trade")
    print(f"  Phase 1 (Pre-Market): {PREMARKET_START[0]}:00–{MARKET_OPEN[0]}:{MARKET_OPEN[1]:02d} ET")
    print(f"  Phase 2 (Intraday):   {MARKET_OPEN[0]}:{MARKET_OPEN[1]:02d}–{INTRADAY_END[0]}:{INTRADAY_END[1]:02d} ET\n")

    monitor_thread = threading.Thread(target=monitor_positions, args=(alpaca,), daemon=True)
    monitor_thread.start()

    seen            = load_seen()
    session_tickers = set()

    # ── Phase 1: Pre-Market ────────────────────────────────────────────────────
    wait = secs_until(*PREMARKET_START)
    if wait > 0:
        print(f"  Waiting {int(wait//60)}m {int(wait%60)}s until 7:00 AM ET…")
        time.sleep(wait)

    if in_premarket_window():
        run_scan_loop(groq_client, alpaca, seen, session_tickers,
                      PREMARKET_PROMPT, "premarket", in_premarket_window)

    # ── Phase 2: Intraday ──────────────────────────────────────────────────────
    wait = secs_until(*MARKET_OPEN)
    if wait > 0:
        print(f"  Waiting {int(wait//60)}m {int(wait%60)}s until market open (9:30 AM ET)…")
        time.sleep(wait)

    if in_intraday_window():
        run_scan_loop(groq_client, alpaca, seen, session_tickers,
                      INTRADAY_PROMPT, "intraday", in_intraday_window)

    # ── Hold until EOD monitor finishes ───────────────────────────────────────
    print(f"[{now_et():%H:%M:%S}] No more new buys. Monitoring {len(session_tickers)} position(s) until 3:55 PM ET…")
    monitor_thread.join()
    print(f"[{now_et():%H:%M:%S}] Bot complete for today.")

if __name__ == "__main__":
    main()
