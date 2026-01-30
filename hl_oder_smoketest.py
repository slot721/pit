#!/usr/bin/env python3
from __future__ import annotations

# scripts/hl_oder_smoketest.py
import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, cast

import ccxt
import requests

"""
# dry run
/opt/dxrr-venv/bin/python -u scripts/hl_order_smoketest.py   --user 0x52....   --coin BTC --usd 10 --lev 1


#live order
/opt/dxrr-venv/bin/python -u scripts/hl_order_smoketest.py \
  --user 0x52...  \
  --coin BTC --side short --usd 10 --lev 1 \
  --live


"""

def load_env_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

# try your canonical path(s)
load_env_file("/opt/dxrr/asterdex-trader/.env")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

import os, re
import requests
from typing import Any

ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

ExpectState = Literal["FILLED", "OPEN", "ACK"]  # ACK = got an order id, no stronger assertion

@dataclass(frozen=True)
class OrderDiag:
    oid: str
    filled: float
    remaining: Optional[float]
    average: Optional[float]
    raw_has_resting: bool

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def hl_order_diag(o: Optional[dict]) -> OrderDiag:
    o = o or {}
    info = o.get("info") or {}
    filled_info = info.get("filled") or {}
    resting_info = info.get("resting") or {}

    oid = str(o.get("id") or filled_info.get("oid") or resting_info.get("oid") or "").strip()

    # CCXT top-level first
    filled = _safe_float(o.get("filled"))
    if filled is None:
        # HL sometimes puts fill size in info.filled.totalSz
        filled = _safe_float(filled_info.get("totalSz")) or 0.0

    remaining = _safe_float(o.get("remaining"))
    average = _safe_float(o.get("average"))
    raw_has_resting = bool(resting_info)

    return OrderDiag(
        oid=oid,
        filled=float(filled or 0.0),
        remaining=remaining,
        average=average,
        raw_has_resting=raw_has_resting,
    )

def assert_hl_order(
    o: Optional[dict],
    *,
    name: str,
    expect: ExpectState,
) -> tuple[bool, str, OrderDiag]:
    d = hl_order_diag(o)

    if not d.oid:
        return False, f"[{name}] no order id in response", d

    if expect == "ACK":
        return True, f"[{name}] ack oid={d.oid}", d

    if expect == "FILLED":
        if d.filled > 0.0:
            return True, f"[{name}] filled={d.filled} oid={d.oid}", d
        if d.raw_has_resting:
            return False, f"[{name}] not filled (resting) oid={d.oid}", d
        return False, f"[{name}] not filled (filled=0) oid={d.oid}", d

    if expect == "OPEN":
        # For SL/TP triggers: immediate fill is suspicious (it would close you instantly)
        if d.filled > 0.0:
            return False, f"[{name}] unexpectedly filled immediately filled={d.filled} oid={d.oid}", d
        return True, f"[{name}] accepted/open (filled=0) oid={d.oid}", d

    return False, f"[{name}] unknown expect={expect}", d

def require_user(args_user: str | None) -> str:
    u = (args_user or "").strip()
    if not u:
        u = (os.getenv("HYPERLIQUID_WALLET_ADDRESS", "") or "").strip()
    if not u:
        raise SystemExit("Missing --user and env HYPERLIQUID_WALLET_ADDRESS is empty")
    if not ADDR_RE.match(u):
        raise SystemExit(f"Invalid user address: {u!r} (expected 0x + 40 hex chars)")
    return u

def load_markets_retry(ex: ccxt.Exchange, *, attempts: int = 5, base_sleep: float = 0.5) -> None:
    last = None
    for i in range(attempts):
        try:
            ex.load_markets()
            return
        except (ccxt.ExchangeNotAvailable, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise RuntimeError(f"[HL] load_markets failed after {attempts} attempts: {type(last).__name__}: {last}")


def post_info(payload: dict, timeout: int = 10) -> Any:
    r = requests.post(HL_INFO_URL, json=payload, timeout=timeout)
    if r.status_code >= 400:
        # critical for debugging 422/500
        raise RuntimeError(f"/info HTTP {r.status_code} body={r.text[:500]} payload={payload}")
    return r.json()

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()



def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_all_mids(timeout: int = 10) -> Dict[str, float]:
    j = post_info({"type": "allMids"}, timeout=timeout)
    # returns dict coin->str price
    out: Dict[str, float] = {}
    if isinstance(j, dict):
        for k, v in j.items():
            out[str(k).upper()] = as_float(v, 0.0)
    return out


def get_user_positions(user: str, timeout: int = 10) -> List[dict]:
    # ✅ HERE: first line in the function (before parsing)
    ch = post_info({"type": "clearinghouseState", "user": user}, timeout=timeout)
    aps = ch.get("assetPositions") or []
    out: List[dict] = []
    for wrap in aps:
        pos = (wrap or {}).get("position") or {}
        c = str(pos.get("coin") or "").upper()
        szi = as_float(pos.get("szi"), 0.0)
        if c and abs(szi) > 0:
            out.append(pos)
    return out


def get_frontend_open_orders(user: str, timeout: int = 10) -> List[dict]:
    j = post_info({"type": "frontendOpenOrders", "user": user}, timeout=timeout)
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        return j.get("orders") or []
    return []


def fmt_json(x: Any) -> str:
    try:
        return json.dumps(x, indent=2, sort_keys=True)
    except Exception:
        return repr(x)

def safe_hl_load_markets(ex: ccxt.Exchange, *, tries: int = 5) -> None:
    last = None
    for i in range(tries):
        try:
            ex.load_markets()
            return
        except (ccxt.ExchangeNotAvailable, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last = e
            time.sleep(min(2.0 * (i + 1), 8.0))
    raise RuntimeError(f"HL load_markets failed after {tries} tries: {last}")

def make_ccxt_hl(*, user_addr: str) -> ccxt.Exchange:
    agent_addr = (os.getenv("HYPERLIQUID_API_WALLET_ADDRESS") or "").strip()
    pk = (os.getenv("HYPERLIQUID_PRIVATE_KEY") or "").strip()
    if not agent_addr or not pk:
        raise SystemExit("Missing HYPERLIQUID_API_WALLET_ADDRESS / HYPERLIQUID_PRIVATE_KEY")

    print(f"[HL] user={user_addr} agent={agent_addr}")

    config: dict[str, object] = {
        "walletAddress": user_addr,   # ✅ user owns position
        "privateKey": pk,             # ✅ agent signs
        "enableRateLimit": True,
        "timeout": 10000,
        "options": {
            "defaultType": "swap",
            "defaultSlippage": 0.005,
            "fetchMarkets": {"hip3": {"dex": ["hyperliquid"]}},
            "broker": "",
            "skipApproveBuilderFee": True,
        },
    }
    ex = ccxt.hyperliquid(config)  # type: ignore[arg-type]
    ex.verbose = False
    return ex



def pick_ccxt_symbol(ex: ccxt.Exchange, coin: str) -> str:
    """
    HL symbols in ccxt can vary by version. Try a few.
    Prefer coin/USDC:USDC, then coin/USDC, then coin itself.
    """
    coin = coin.upper().strip()
    candidates = [
        f"{coin}/USDC:USDC",
        f"{coin}/USDC",
        coin,
    ]
    markets = getattr(ex, "markets", {}) or {}
    for s in candidates:
        if s in markets:
            return s
    # last resort: return first candidate; ccxt will error and show allowed symbols
    return candidates[0]

OrderSide = Literal["buy", "sell"]

def _req_str(x: Any, *, name: str) -> str:
    if x is None:
        raise ValueError(f"{name} is None")
    s = str(x).strip()
    if not s:
        raise ValueError(f"{name} is empty")
    return s

def _req_float(x: Any, *, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        f = float(x)
    except Exception as e:
        raise ValueError(f"{name} not float-convertible: {x!r}") from e
    if f <= 0:
        raise ValueError(f"{name} must be > 0, got {f}")
    return f


def place_market_entry_precise(
    ex: Any,                 # ccxt.Exchange typing is messy; Any avoids pylance false-positives
    symbol: str,
    side: OrderSide,
    amount: float,
    *,
    price_hint: float,
    params: Optional[dict[str, Any]] = None,
) -> dict:
    if amount <= 0:
        raise ValueError(f"bad amount={amount}")
    if price_hint <= 0:
        raise ValueError(f"bad price_hint={price_hint}")

    # ccxt precision helpers return strings (and stubs often say Unknown|None)
    amt_s = _req_str(ex.amount_to_precision(symbol, amount), name="amount_to_precision")
    px_s  = _req_str(ex.price_to_precision(symbol, price_hint), name="price_to_precision")

    amt = _req_float(amt_s, name="amount_precise")
    px  = _req_float(px_s,  name="price_precise")

    return ex.create_order(symbol, "market", side, amt, px, params or {})

def best_effort_cancel_all(ex: ccxt.Exchange, symbol: str) -> int:
    try:
        oo = ex.fetch_open_orders(symbol)
    except Exception:
        oo = []
    n = 0
    for r in oo:
        oid = r.get("id")
        if not oid:
            continue
        try:
            ex.cancel_order(oid, symbol)
            n += 1
        except Exception:
            pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True, help="e.g. BTC, ETH, SOL")
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--usd", type=float, default=10.0, help="Approx notional in USD (USDC)")
    ap.add_argument("--lev", type=int, default=1, help="Target leverage")
    ap.add_argument("--sl-pct", type=float, default=0.004, help="Stop loss pct (e.g. 0.004 = 0.4%)")
    ap.add_argument("--tp-pct", type=float, default=0.006, help="Take profit pct")
    ap.add_argument("--live", action="store_true", help="Actually place orders")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--move-sl", action="store_true", help="After placing exits, cancel+recreate SL at tighter price")
    ap.add_argument("--user", default="", help="0x... wallet (or set HYPERLIQUID_API_WALLET_ADDRESS)")
    args = ap.parse_args()

    user_addr = require_user(args.user)  # canonical
    # optional fallback for convenience:
    if not user_addr:
        user_addr = (os.getenv("HYPERLIQUID_PUBLIC_WALLET_ADDRESS") or "").strip()

    coin = args.coin.strip().upper()
    side_txt = args.side
    side_ccxt = "buy" if side_txt == "long" else "sell"
    leverage = args.lev
    wallet = (os.getenv("HYPERLIQUID_API_WALLET_ADDRESS") or "").strip()
    user_addr = (os.getenv("HYPERLIQUID_PUBLIC_WALLET_ADDRESS") or "").strip()
    if not wallet:
        raise SystemExit("Missing HYPERLIQUID_API_WALLET_ADDRESS")

    mids = get_all_mids(timeout=args.timeout)
    px = mids.get(coin, 0.0)
    if px <= 0:
        raise SystemExit(f"Could not get mid for {coin} from allMids")

    amount = float(args.usd) / float(px)
    # guard: tiny sizes can get rejected; you can bump usd if needed
    if amount <= 0:
        raise SystemExit("Computed amount <= 0")

    # Build prices
    if side_txt == "long":
        sl = px * (1.0 - args.sl_pct)
        tp = px * (1.0 + args.tp_pct)
    else:
        sl = px * (1.0 + args.sl_pct)
        tp = px * (1.0 - args.tp_pct)

    print(f"[CTX] ts={_utc()} coin={coin} mid={px:.6f} usd={args.usd} amount≈{amount:.10f} side={side_txt} lev={args.lev}")
    print(f"[CTX] SL={sl:.6f} TP={tp:.6f}")

    # Show current state (before)
    pos0 = get_user_positions(user_addr, timeout=args.timeout)
    foo0 = get_frontend_open_orders(user_addr, timeout=args.timeout)
    print(f"[BEFORE] positions_nonzero={len(pos0)} open_orders={len(foo0)}")
    if args.show_all:
        print("positions=", fmt_json(pos0))
        print("orders=", fmt_json(foo0))

    if not args.live:
        print("[DRY] --live not set, stopping before trading calls.")
        return 0

    user_addr = require_user(args.user)
    ex = make_ccxt_hl(user_addr=user_addr)
    load_markets_retry(ex)

    sym_ccxt = f"{coin}/USDC:USDC"   # e.g. BTC/USDC:USDC
    m = ex.market(sym_ccxt)          # asserts it exists in loaded markets
    print("[MARKET]", m["symbol"], "id=", m.get("id"))

    sym_ccxt = pick_ccxt_symbol(ex, coin)  # pick_ccxt_symbol should NOT call load_markets anymore

    print(f"[CCXT] symbol={sym_ccxt}")

    # -------------------
    # 1 Entry
    # -------------------
    def _entry():
        return place_market_entry_precise(ex, sym_ccxt, side_ccxt, float(amount), price_hint=float(px), params=None)

    entry = hl_with_slippage_retry(ex, _entry, name="ENTRY")
    print("entry=", fmt_json(entry))

    filled_sz, msg = assert_entry_filled(entry, sym_ccxt, ex)
    print(msg)
    if filled_sz <= 0:
        return 0

    exit_qty = filled_sz
    exit_side = "buy" if side_ccxt == "sell" else "sell"


    # 2) Inspect exchange truth after entry (position + orders)
    time.sleep(1.0)
    pos1 = get_user_positions(user_addr, timeout=args.timeout)
    foo1 = get_frontend_open_orders(user_addr, timeout=args.timeout)
    print(f"[AFTER ENTRY] positions_nonzero={len(pos1)} open_orders={len(foo1)}")
    # right after foo1 is loaded
    if foo1:
        print(f"[EXITS] open_orders={len(foo1)} detected; cancelling before placing SL/TP")
        best_effort_cancel_all(ex, sym_ccxt)
        time.sleep(0.5)

    if args.show_all:
        print("positions=", fmt_json(pos1))
        print("orders=", fmt_json(foo1))

    px2 = float(get_all_mids(timeout=args.timeout).get(coin, 0.0) or px)
    mid_now = px2   # <-- define it
    exit_side = "buy" if side_ccxt == "sell" else "sell"

    # -------------------
    # 3) TP / SL
    # -------------------
    if foo1:
        print(f"[EXITS] open_orders={len(foo1)} detected; cancelling before placing SL/TP")
        best_effort_cancel_all(ex, sym_ccxt)
        time.sleep(0.5)
    
    sl_o = place_reduce_only_trigger_market(
        ex, sym_ccxt, exit_side, exit_qty,
        trigger_price=sl,
        price_hint=sl,
    )
    tp_o = place_reduce_only_trigger_market(
        ex, sym_ccxt, exit_side, exit_qty,
        trigger_price=tp,
        price_hint=tp,
    )
    
    sl_oid, tp_oid = _oid(sl_o), _oid(tp_o)
    print(f"[SL] oid={sl_oid} filled={_filled(sl_o)}")
    print(f"[TP] oid={tp_oid} filled={_filled(tp_o)}")
    
    want = {x for x in (sl_oid, tp_oid) if x}
    ok_vis = wait_open_orders_contain(user_addr=user_addr, want_oids=want, timeout_s=2.5)
    print(f"[EXITS] visible_in_open_orders={ok_vis} want={sorted(list(want))}")

    # 4) Optionally "move SL": cancel existing SL/TP, then recreate tighter SL
    if args.move_sl:
        cancelled = best_effort_cancel_all(ex, sym_ccxt)
        print(f"[MOVE SL] cancelled open orders n={cancelled}")
        time.sleep(0.5)

        px2 = float(get_all_mids(timeout=args.timeout).get(coin, 0.0) or px)
        if side_txt == "long":
            sl2 = px2 * (1.0 - max(0.001, args.sl_pct / 2.0))
            exit_side2 = "sell"
        else:
            sl2 = px2 * (1.0 + max(0.001, args.sl_pct / 2.0))
            exit_side2 = "buy"

        print(f"[MOVE SL] new SL={sl2:.6f} mid={px2:.6f} qty={exit_qty}")

        try:
            o_sl2 = place_reduce_only_trigger_market(
                ex, sym_ccxt, exit_side2, exit_qty,
                trigger_price=sl2,
                price_hint=sl2,
            )
            print("[MOVE SL] SL2 order=", fmt_json(o_sl2))
        except Exception as e:
            print(f"[MOVE SL] SL2 create failed: {type(e).__name__}: {str(e)[:400]}")

    # -------------------
    # 5) Flatten: reduce-only market opposite side
    # -------------------
    print("[FLATTEN] reduce-only market close")
    mid_flat = float(get_all_mids(timeout=args.timeout).get(coin, 0.0) or px)
    flat = flatten_reduce_only(ex, sym_ccxt, exit_side, exit_qty, mid_now=mid_flat)
    ok_f, msg_f, _ = assert_hl_order(flat, name="FLATTEN", expect="FILLED")
    print(msg_f)

    best_effort_cancel_all(ex, sym_ccxt)
    time.sleep(0.5)
    pos2 = get_user_positions(user_addr, timeout=args.timeout)
    foo2 = get_frontend_open_orders(user_addr, timeout=args.timeout)
    print(f"[AFTER FLATTEN] positions_nonzero={len(pos2)} open_orders={len(foo2)}")
    if args.show_all:
        print("positions=", fmt_json(pos2))
        print("orders=", fmt_json(foo2))
    
    # after flatten (cleanup)
    best_effort_cancel_all(ex, sym_ccxt)
    time.sleep(0.5)
    return 0

T = TypeVar("T")

def hl_with_slippage_retry(
    ex: Any,                      # <-- avoid ccxt stub nonsense
    fn: Callable[[], T],
    *,
    name: str,
    slippages: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> T:
    last_err: Optional[Exception] = None

    for s in slippages:
        try:
            # CCXT stubs type options poorly; cast fixes Pylance.
            opts = cast(dict[str, Any], getattr(ex, "options", {}))
            opts["defaultSlippage"] = float(s)
            ex.options = opts  # keep it explicit/safe

            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"[{name}] try slippage={s:.3%} failed: {type(e).__name__}: {msg[:220]}")

    # make Pylance + runtime happy
    raise RuntimeError(f"[{name}] all slippage retries failed") from last_err

def place_reduce_only_trigger_market(
    ex: ccxt.Exchange,
    symbol: str,
    side: OrderSide,     # enforce buy/sell for pylance
    qty: float,
    *,
    trigger_price: float,
    price_hint: float,   # recommend price_hint=trigger_price
) -> dict:
    def _do():
        params = {
            "reduceOnly": True,
            "triggerPrice": float(trigger_price),
        }
        # reuse the same precision-safe path as entry
        return place_market_entry_precise(
            ex,
            symbol,
            side,
            float(qty),
            price_hint=float(price_hint),
            params=params,
        )

    return hl_with_slippage_retry(ex, _do, name="TRIGGER_MKT")

def wait_open_orders_contain(
    *,
    user_addr: str,
    want_oids: set[str],
    timeout_s: float = 2.0,
    poll_s: float = 0.2,
    timeout_api: float = 10.0,
) -> bool:
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        oo = get_frontend_open_orders(user_addr, timeout=timeout_api)
        have = {str(r.get("oid") or r.get("id") or "").strip() for r in (oo or [])}
        if want_oids.issubset(have):
            return True
        time.sleep(poll_s)
    return False

def _oid(o: dict | None) -> str:
    o = o or {}
    oid = str(o.get("id") or "").strip()
    if oid:
        return oid
    info = o.get("info") or {}
    return str((info.get("filled") or {}).get("oid") or (info.get("resting") or {}).get("oid") or "").strip()

def _filled(o: dict | None) -> float:
    o = o or {}
    try:
        x = o.get("filled")
        if x is not None:
            return float(x)
    except Exception:
        pass
    info = o.get("info") or {}
    try:
        return float((info.get("filled") or {}).get("totalSz") or 0.0)
    except Exception:
        return 0.0

def assert_entry_filled(entry: dict, symbol: str, ex: ccxt.Exchange) -> tuple[float, str]:
    f = _filled(entry)
    oid = _oid(entry)
    if f > 0:
        return f, f"[ENTRY] filled={f} oid={oid}"
    # treat as resting/unfilled => cancel
    if oid:
        try:
            ex.cancel_order(oid, symbol)
            return 0.0, f"[ENTRY] not filled (resting). cancelled oid={oid}"
        except Exception as e:
            return 0.0, f"[ENTRY] not filled; cancel failed oid={oid}: {type(e).__name__}: {str(e)[:160]}"
    return 0.0, "[ENTRY] not filled and no oid returned"

def flatten_reduce_only(
    ex: ccxt.Exchange,
    symbol: str,
    close_side: OrderSide,
    qty: float,
    *,
    mid_now: float,
) -> dict:
    def _do():
        return place_market_entry_precise(
            ex,
            symbol,
            close_side,
            float(qty),
            price_hint=float(mid_now),
            params={"reduceOnly": True},
        )
    return hl_with_slippage_retry(ex, _do, name="FLATTEN")

if __name__ == "__main__":
    raise SystemExit(main())
