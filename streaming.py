"""
Real-time trade tick streaming manager for PreTrade.

Supports:
- `demo`: Zero-key background thread simulating realistic geometric Brownian motion ticks
- `finnhub`: Live Finnhub WebSocket (`wss://ws.finnhub.io?token=...`) trade stream
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from market_data import fetch_quote

logger = logging.getLogger(__name__)


class StreamManager:
    """Thread-safe background trade tick streaming manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws_app: Any = None
        self._provider: str = "demo"
        self._status: str = "idle"
        self._error: str | None = None
        self._symbols: list[str] = []
        self._base_prices: dict[str, float] = {}
        self._latest: dict[str, dict[str, Any]] = {}

    def start(
        self,
        provider: str = "demo",
        symbols: Sequence[str] = ("AAPL",),
        api_key: str | None = None,
        seed_prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Start streaming trade ticks for the given symbols."""
        self.stop()
        clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not clean_symbols:
            clean_symbols = ["AAPL"]

        prov = str(provider).strip().lower()
        if prov not in ("demo", "finnhub"):
            raise ValueError("provider must be 'demo' or 'finnhub'.")

        if prov == "finnhub" and not (api_key and str(api_key).strip()):
            with self._lock:
                self._provider = prov
                self._symbols = clean_symbols
                self._status = "error"
                self._error = "Missing Finnhub API key"
            return self.snapshot()

        bases: dict[str, float] = {}
        for sym in clean_symbols:
            if seed_prices and sym in seed_prices and seed_prices[sym] > 0:
                bases[sym] = float(seed_prices[sym])
            else:
                try:
                    q = fetch_quote(sym)
                    bases[sym] = float(q["price"])
                except Exception:
                    bases[sym] = 100.0

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            self._provider = prov
            self._symbols = clean_symbols
            self._base_prices = dict(bases)
            self._error = None
            self._status = "streaming" if prov == "demo" else "connecting"
            self._latest = {
                sym: {
                    "symbol": sym,
                    "price": round(px, 2),
                    "change_pct": 0.0,
                    "volume": 100,
                    "source": prov,
                    "iso": now_iso,
                }
                for sym, px in bases.items()
            }

        self._stop_event.clear()
        if prov == "demo":
            self._thread = threading.Thread(target=self._run_demo_loop, daemon=True)
            self._thread.start()
        else:
            self._thread = threading.Thread(
                target=self._run_finnhub_loop, args=(str(api_key).strip(),), daemon=True
            )
            self._thread.start()

        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        """Stop the active background stream thread."""
        self._stop_event.set()
        ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception as exc:
                logger.debug("WebSocket operation ignored: %s", exc)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._thread = None
            self._ws_app = None
            if self._status not in ("idle", "error"):
                self._status = "stopped"
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of current stream state and latest ticks."""
        with self._lock:
            return {
                "status": self._status,
                "provider": self._provider,
                "error": self._error,
                "symbols": list(self._symbols),
                "latest": {k: dict(v) for k, v in self._latest.items()},
            }

    def _record_tick(self, symbol: str, price: float, volume: float = 100.0, source: str = "demo") -> None:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            base = self._base_prices.get(symbol, price) or price
            chg_pct = ((price - base) / base * 100.0) if base > 0 else 0.0
            self._latest[symbol] = {
                "symbol": symbol,
                "price": round(float(price), 2),
                "change_pct": round(float(chg_pct), 3),
                "volume": float(volume),
                "source": source,
                "iso": now_iso,
            }
            self._status = "streaming"

    def _run_demo_loop(self) -> None:
        rng = np.random.default_rng()
        while not self._stop_event.is_set():
            with self._lock:
                symbols = list(self._symbols)
                current_map = {s: float(self._latest.get(s, {}).get("price", 100.0)) for s in symbols}
            for sym in symbols:
                if self._stop_event.is_set():
                    break
                cur_px = current_map.get(sym, 100.0)
                shock = float(rng.normal(0.0, 0.0012))
                new_px = max(0.50, cur_px * (1.0 + shock))
                vol = int(rng.integers(25, 600))
                self._record_tick(sym, new_px, volume=vol, source="demo")
            self._stop_event.wait(0.8)

    def _run_finnhub_loop(self, api_key: str) -> None:
        try:
            import websocket
        except ImportError:
            with self._lock:
                self._status = "error"
                self._error = "websocket-client package is not installed"
            return

        url = f"wss://ws.finnhub.io?token={api_key}"

        def on_open(ws: Any) -> None:
            with self._lock:
                self._status = "streaming"
                syms = list(self._symbols)
            for sym in syms:
                try:
                    ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                except Exception as exc:
                    logger.debug("WebSocket subscribe ignored: %s", exc)

        def on_message(_ws: Any, message: str) -> None:
            try:
                payload = json.loads(message)
                if payload.get("type") == "trade":
                    for trade in payload.get("data") or []:
                        sym = str(trade.get("s") or "").upper()
                        px = float(trade.get("p") or 0.0)
                        vol = float(trade.get("v") or 0.0)
                        if sym and px > 0:
                            self._record_tick(sym, px, volume=vol, source="finnhub")
                elif payload.get("type") == "error":
                    with self._lock:
                        self._status = "error"
                        self._error = str(payload.get("msg") or "Finnhub stream error")
            except Exception as exc:
                logger.debug("WebSocket operation ignored: %s", exc)

        def on_error(_ws: Any, error: Any) -> None:
            with self._lock:
                if not self._stop_event.is_set():
                    self._status = "error"
                    self._error = str(error)

        def on_close(_ws: Any, _code: Any, _msg: Any) -> None:
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"

        try:
            ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            with self._lock:
                self._ws_app = ws_app
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            with self._lock:
                self._status = "error"
                self._error = str(exc)


_GLOBAL_MANAGER = StreamManager()


def get_manager() -> StreamManager:
    """Return the process-wide singleton StreamManager instance."""
    return _GLOBAL_MANAGER


def start_stream(
    provider: str = "demo",
    symbols: Sequence[str] = ("AAPL",),
    api_key: str | None = None,
    seed_prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Start the singleton trade stream and return its initial snapshot."""
    return _GLOBAL_MANAGER.start(
        provider=provider,
        symbols=symbols,
        api_key=api_key,
        seed_prices=seed_prices,
    )


def stop_stream() -> dict[str, Any]:
    """Stop the singleton trade stream and return its final snapshot."""
    return _GLOBAL_MANAGER.stop()


def stream_snapshot() -> dict[str, Any]:
    """Return the current state snapshot from the singleton StreamManager."""
    return _GLOBAL_MANAGER.snapshot()


def apply_stream_price_to_quote(
    quote: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlay the latest streamed tick price onto a quote dictionary if available."""
    updated = dict(quote)
    snap = snapshot if snapshot is not None else stream_snapshot()
    sym = str(updated.get("symbol") or "").upper()
    tick = (snap.get("latest") or {}).get(sym)
    if tick and tick.get("price"):
        px = float(tick["price"])
        prev = float(updated.get("prev_close") or px)
        updated["price"] = round(px, 2)
        updated["change_pct"] = round(((px - prev) / prev * 100.0) if prev > 0 else 0.0, 2)
        updated["source"] = f"stream:{tick.get('source', 'live')}"
    return updated
