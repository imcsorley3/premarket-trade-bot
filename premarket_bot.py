#!/usr/bin/env python3
"""
Pre-Market Trade Bot
Scrapes financial news 7:00–9:30 AM ET, generates HIGH-confidence BUY signals
via Groq (free Llama 3), and executes paper trades on Alpaca.
Monitors positions and auto-sells at take-profit or stop-loss thresholds.
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

# ── Config (edit these in .env) ────────────────────────────────────────────────
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "5.0"))  # sell at +5%
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT",   "2.0"))  # sell at -2%
TRADE_VALUE     = float(os.getenv("TRADE_VALUE",    "25.0"))  # $ per trade
POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL",      "60"))  # seconds

ET           = ZoneInfo("America/New_York")
SCRAPE_START = (7,  0)   # 7:00 AM ET
SCRAPE_END   = (9, 30)   # 9:30 AM ET  (market open)
EOD_CLOSE    = (15, 55)  # 3:55 PM ET

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

# Assets the bot is allowed to trade
ALLOWED_ASSET_TYPES = """
- US-listed individual stocks (e.g. AAPL, NVDA, XOM)
- US-listed ETFs (e.g. SPY, QQQ, XLE, XLF, IWM)
- Gold ETFs ONLY: GLD or IAU (do NOT use GC=F or futures)
- Oil ETFs ONLY: USO or XLE (do NOT use CL=F or futures)
Do NOT recommend crypto, bonds, forex, or futures contracts.
"""

SYSTEM_PROMPT = f"""You are a pre-market financial analyst reviewing headlines from 7–9:30 AM ET.
Assess whether each headline is a HIGH-confidence BUY opportunity.

Tradeable assets:
{ALLOWED_ASSET_TYPES}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
    "ticker": "SYMBOL",
    "action": "BUY" | "SKIP",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "one sentence"
}}

Rules:
- Return action=BUY with confidence=HIGH only when news is clearly and directly bullish for a specific ticker.
- Prefer ETFs for broad macro themes. Use individual stocks only for company-specific news.
- If uncertain, negative, or not market-relevant: return action=SKIP.
- Do not recommend tickers that are not liquid US-listed stocks or ETFs.
"""

# ── Database ───────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT NOT NULL,
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
            status        TEXT DEFAULT 'OPEN'
        )
    """)
    conn.commit()
    conn.close()

def log_open(article: dict, rec: dict, qty: float, entry_price: float, order_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO trades (date, opened_at, source, headline, ticker, qty, entry_price, alpaca_buy_id, reasoning, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        datetime.now(ET).strftime("%Y-%m-%d"),
        datetime.now(timezone.utc).isoformat(),
        article["source"],
        article["title"],
        rec["ticker"],
        qty,
        entry_price,
        order_id,
        rec.get("reasoning", ""),
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

def in_scrape_window() -> bool:
    t = now_et()
    start = t.replace(hour=SCRAPE_START[0], minute=SCRAPE_START[1], second=0, microsecond=0)
    end   = t.replace(hour=SCRAPE_END[0],   minute=SCRAPE_END[1],   second=0, microsecond=0)
    return start <= t < end

def is_eod() -> bool:
    t = now_et()
    return t.hour > EOD_CLOSE[0] or (t.hour == EOD_CLOSE[0] and t.minute >= EOD_CLOSE[1])

def secs_until_start() -> float:
    t     = now_et()
    start = t.replace(hour=SCRAPE_START[0], minute=SCRAPE_START[1], second=0, microsecond=0)
    if t >= start:
        return 0
    return (start - t).total_seconds()

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

def get_recommendation(groq_client: Groq, headline: str, source: str, session_tickers: set) -> dict:
    already_bought = ", ".join(session_tickers) if session_tickers else "none"
    user_msg = (
        f"Source: {source}\n"
        f"Headline: {headline}\n"
        f"Tickers already bought this session (do not recommend again): {already_bought}"
    )
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
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

def get_quote_price(alpaca: TradingClient, ticker: str) -> float | None:
    try:
        asset = alpaca.get_asset(ticker)
        if not asset.tradable:
            return None
        # Use latest trade price from data API
        url = f"https://data.sandbox.alpaca.markets/v2/stocks/{ticker}/trades/latest"
        headers = {
            "APCA-API-KEY-ID":     os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            return float(r.json()["trade"]["p"])
    except Exception:
        pass
    return None

def place_buy(alpaca: TradingClient, ticker: str) -> tuple[float, float, str] | None:
    try:
        order = alpaca.submit_order(MarketOrderRequest(
            symbol=ticker,
            notional=TRADE_VALUE,          # fractional dollar amount
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        # Allow order to fill
        time.sleep(2)
        filled = alpaca.get_order_by_id(str(order.id))
        qty    = float(filled.filled_qty   or 0)
        price  = float(filled.filled_avg_price or 0)
        return qty, price, str(order.id)
    except Exception as e:
        print(f"  [ERROR] Buy order for {ticker} failed: {e}")
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
        print(f"  [ERROR] Sell order for {ticker} failed: {e}")
        return None

# ── Position monitor (runs in background thread) ───────────────────────────────

def monitor_positions(alpaca: TradingClient):
    print(f"  [Monitor] Position monitor started (TP: +{TAKE_PROFIT_PCT}% | SL: -{STOP_LOSS_PCT}%)")
    while True:
        try:
            if is_eod():
                positions = alpaca.get_all_positions()
                if positions:
                    print(f"\n[{now_et():%H:%M:%S}] EOD — closing {len(positions)} open position(s)…")
                    for pos in positions:
                        ticker = pos.symbol
                        qty    = float(pos.qty)
                        price  = place_sell(alpaca, ticker, qty)
                        if price:
                            log_close(ticker, price, "EOD")
                            pct = float(pos.unrealized_plpc) * 100
                            print(f"  [EOD CLOSE] {ticker}: {pct:+.2f}%")
                print(f"[{now_et():%H:%M:%S}] EOD complete. Monitor shutting down.")
                break

            positions = alpaca.get_all_positions()
            for pos in positions:
                ticker   = pos.symbol
                qty      = float(pos.qty)
                pct      = float(pos.unrealized_plpc) * 100

                if pct >= TAKE_PROFIT_PCT:
                    print(f"\n  [TAKE PROFIT] {ticker} @ {pct:+.2f}% — selling…")
                    price = place_sell(alpaca, ticker, qty)
                    if price:
                        log_close(ticker, price, "TAKE_PROFIT")
                        print(f"  [SOLD] {ticker} @ ${price:.2f} | +{pct:.2f}%")

                elif pct <= -STOP_LOSS_PCT:
                    print(f"\n  [STOP LOSS] {ticker} @ {pct:+.2f}% — selling…")
                    price = place_sell(alpaca, ticker, qty)
                    if price:
                        log_close(ticker, price, "STOP_LOSS")
                        print(f"  [SOLD] {ticker} @ ${price:.2f} | {pct:.2f}%")

        except Exception as e:
            print(f"  [Monitor ERROR] {e}")

        time.sleep(30)

# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    init_db()

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    alpaca      = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )

    acct = alpaca.get_account()
    print(f"[{now_et():%H:%M:%S}] Pre-Market Trade Bot")
    print(f"  Paper account: ${float(acct.cash):,.2f} cash available")
    print(f"  Take-profit: +{TAKE_PROFIT_PCT}%  |  Stop-loss: -{STOP_LOSS_PCT}%  |  Per trade: ${TRADE_VALUE}")
    print(f"  Scrape window: {SCRAPE_START[0]}:{SCRAPE_START[1]:02d} – {SCRAPE_END[0]}:{SCRAPE_END[1]:02d} ET\n")

    # Start position monitor in background
    monitor_thread = threading.Thread(target=monitor_positions, args=(alpaca,), daemon=True)
    monitor_thread.start()

    # Wait for scrape window if not already in it
    wait = secs_until_start()
    if wait > 0:
        print(f"  Waiting {int(wait//60)}m {int(wait%60)}s until 7:00 AM ET…")
        time.sleep(wait)

    seen            = load_seen()
    session_tickers = set()  # avoid double-buying same ticker in one session

    print(f"[{now_et():%H:%M:%S}] Scrape window open — polling every {POLL_INTERVAL}s until 9:30 AM ET\n")

    while in_scrape_window():
        ts = f"[{now_et():%H:%M:%S}]"
        print(f"{ts} Polling feeds…")

        articles = fetch_feeds()
        new_ones = [a for a in articles if a["id"] not in seen and a["title"]]
        print(f"  {len(new_ones)} new article(s)")

        for article in new_ones:
            try:
                rec = get_recommendation(groq_client, article["title"], article["source"], session_tickers)

                if rec.get("action") != "BUY" or rec.get("confidence") != "HIGH":
                    print(f"  → Skip ({rec.get('action','?')}/{rec.get('confidence','?')}): {article['title'][:60]}")
                    seen.add(article["id"])
                    time.sleep(0.5)
                    continue

                ticker = rec["ticker"].upper().strip()

                if ticker in session_tickers:
                    print(f"  → Skip (already bought {ticker} this session)")
                    seen.add(article["id"])
                    continue

                print(f"  → BUY {ticker} | {article['title'][:60]}")
                result = place_buy(alpaca, ticker)
                if result:
                    qty, price, order_id = result
                    log_open(article, rec, qty, price, order_id)
                    session_tickers.add(ticker)
                    print(f"  ✓ Bought {qty:.4f} {ticker} @ ${price:.2f} (${TRADE_VALUE} notional)")

            except Exception as e:
                print(f"  [ERROR] {article['title'][:60]}: {e}")

            seen.add(article["id"])
            time.sleep(1)

        save_seen(seen)
        print(f"  Sleeping {POLL_INTERVAL}s…")
        time.sleep(POLL_INTERVAL)

    print(f"\n[{now_et():%H:%M:%S}] Scrape window closed (9:30 AM ET). No more new buys.")
    print(f"  Positions open: {len(session_tickers)} | Monitor running until EOD (3:55 PM ET)…")

    # Keep process alive until monitor thread finishes at EOD
    monitor_thread.join()
    print(f"[{now_et():%H:%M:%S}] Bot complete for today.")

if __name__ == "__main__":
    main()
