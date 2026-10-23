"""Append-only, hash-chained paper-money ledger.

Layout under the ledger root (default `ledger/`):

  bets.jsonl         one row per bet placed
  settlements.jsonl  one row per settled bet, referencing the bet's hash
  snapshots/         raw API responses kept as evidence, named by sha256

Each jsonl file is its own chain. Every row carries `prev_hash` (the
previous row's `hash`, or 64 zeros for the first row) and `hash`, the
sha256 of the row's canonical JSON with `hash` removed. Editing, deleting
or reordering any row breaks every hash after it.

Each bet also records `settlements_seen` and `settlements_head` (the
settlements chain length and last hash at placement), which anchors the
settlements chain to the bets chain: truncating settlements that an
earlier bet already saw is detected even when no cash moved (a loss).

Bankroll is cash on hand: start, minus stakes as bets are placed, plus
returns (stake x odds on a win, stake on a void) as they settle.

Verify: uv run worldcup-verify-ledger [--dir ledger]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from worldcup.betting import engine

START_BANKROLL = 1000.0
GENESIS_HASH = "0" * 64
BETS_FILE = "bets.jsonl"
SETTLEMENTS_FILE = "settlements.jsonl"
SNAPSHOT_DIR = "snapshots"
SELECTIONS = ("home", "draw", "away")
MONEY_TOL = 0.005

BET_FIELDS = (
    "placed_at", "event_id", "kickoff", "home", "away", "selection",
    "odds", "stake", "model_p", "implied_p", "edge", "bankroll_before",
)
SETTLEMENT_FIELDS = (
    "settled_at", "bet_hash", "event_id", "result", "outcome",
    "closing_odds", "clv", "clv_raw", "returns", "pnl", "bankroll_after",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def row_hash(row: dict) -> str:
    body = {k: v for k, v in row.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must carry a timezone: {ts!r}")
    return dt


def _cents(x: float) -> float:
    return round(x + 0.0, 2)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name}:{n}: bad JSON ({e})") from e
    return rows


class Ledger:
    def __init__(self, root: str | Path = "ledger", start_bankroll: float = START_BANKROLL):
        self.root = Path(root)
        self.start_bankroll = start_bankroll

    @property
    def bets_path(self) -> Path:
        return self.root / BETS_FILE

    @property
    def settlements_path(self) -> Path:
        return self.root / SETTLEMENTS_FILE

    @property
    def snapshot_dir(self) -> Path:
        return self.root / SNAPSHOT_DIR

    def bets(self) -> list[dict]:
        return read_rows(self.bets_path)

    def settlements(self) -> list[dict]:
        return read_rows(self.settlements_path)

    def bankroll(self) -> float:
        cash = self.start_bankroll
        cash -= sum(b["stake"] for b in self.bets())
        cash += sum(s["returns"] for s in self.settlements())
        return _cents(cash)

    def open_bets(self) -> list[dict]:
        settled = {s["bet_hash"] for s in self.settlements()}
        return [b for b in self.bets() if b["hash"] not in settled]

    def _append(self, path: Path, row: dict) -> dict:
        rows = read_rows(path)
        row = dict(row)
        row["prev_hash"] = rows[-1]["hash"] if rows else GENESIS_HASH
        row["hash"] = row_hash(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
        return row

    def save_snapshot(self, raw: bytes | str, label: str) -> dict:
        """Store a raw API response; returns a reference to put on a bet or settlement."""
        data = raw.encode("utf-8") if isinstance(raw, str) else raw
        digest = hashlib.sha256(data).hexdigest()
        name = f"{digest[:16]}-{label}.json"
        path = self.snapshot_dir / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {"file": name, "sha256": digest}

    def place_bet(
        self,
        *,
        event_id: str,
        kickoff: str,
        home: str,
        away: str,
        selection: int,
        odds: Sequence[float],
        stake: float,
        model_p: Sequence[float],
        placed_at: str | None = None,
        snapshot: dict | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Record a bet on `selection` (0 home, 1 draw, 2 away) at the quoted 1X2 `odds`."""
        placed_at = placed_at or now_utc()
        if _parse_ts(placed_at) >= _parse_ts(kickoff):
            raise ValueError(f"bet placed at/after kickoff: {event_id}")
        if any(b["event_id"] == event_id for b in self.bets()):
            raise ValueError(f"already bet on event {event_id}")
        bankroll = self.bankroll()
        stake = _cents(stake)
        if stake <= 0 or stake > bankroll + MONEY_TOL:
            raise ValueError(f"stake {stake} outside (0, bankroll {bankroll}]")
        implied, overround = engine.devig_proportional(odds)
        sets = self.settlements()
        row = {
            "type": "bet",
            "placed_at": placed_at,
            "event_id": event_id,
            "kickoff": kickoff,
            "home": home,
            "away": away,
            "selection": SELECTIONS[selection],
            "odds": float(odds[selection]),
            "odds_1x2": [float(o) for o in odds],
            "overround": round(overround, 6),
            "stake": stake,
            "model_p": round(float(model_p[selection]), 6),
            "model_p_1x2": [round(float(p), 6) for p in model_p],
            "implied_p": round(implied[selection], 6),
            "edge": round(engine.edge(model_p[selection], odds[selection]), 6),
            "bankroll_before": bankroll,
            "settlements_seen": len(sets),
            "settlements_head": sets[-1]["hash"] if sets else GENESIS_HASH,
        }
        if snapshot:
            row["snapshot"] = snapshot
        if extra:
            row["extra"] = extra
        return self._append(self.bets_path, row)

    def settle(
        self,
        bet_hash: str,
        outcome: int | None,
        *,
        closing_odds: Sequence[float] | None = None,
        settled_at: str | None = None,
        snapshot: dict | None = None,
    ) -> dict:
        """Settle a bet. `outcome` is 0/1/2 (home/draw/away after 90 min) or None for void."""
        bet = next((b for b in self.bets() if b["hash"] == bet_hash), None)
        if bet is None:
            raise ValueError(f"no bet with hash {bet_hash}")
        if any(s["bet_hash"] == bet_hash for s in self.settlements()):
            raise ValueError(f"bet {bet_hash[:12]} already settled")
        settled_at = settled_at or now_utc()
        if _parse_ts(settled_at) < _parse_ts(bet["placed_at"]):
            raise ValueError("settlement precedes bet")
        sel = SELECTIONS.index(bet["selection"])
        if outcome is None:
            result, returns = "void", bet["stake"]
        elif outcome == sel:
            result, returns = "won", bet["stake"] * bet["odds"]
        else:
            result, returns = "lost", 0.0
        returns = _cents(returns)
        row = {
            "type": "settlement",
            "settled_at": settled_at,
            "bet_hash": bet_hash,
            "event_id": bet["event_id"],
            "result": result,
            "outcome": None if outcome is None else SELECTIONS[outcome],
            "closing_odds": [float(o) for o in closing_odds] if closing_odds else None,
            "clv": round(engine.clv(bet["odds"], closing_odds, sel), 6) if closing_odds else None,
            "clv_raw": round(engine.clv_raw(bet["odds"], closing_odds, sel), 6) if closing_odds else None,
            "returns": returns,
            "pnl": _cents(returns - bet["stake"]),
            "bankroll_after": _cents(self.bankroll() + returns),
        }
        if snapshot:
            row["snapshot"] = snapshot
        return self._append(self.settlements_path, row)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    n_bets: int = 0
    n_settled: int = 0
    bankroll: float = START_BANKROLL
    staked: float = 0.0
    pnl: float = 0.0
    clvs: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_chain(name: str, rows: list[dict], errors: list[str]) -> None:
    prev = GENESIS_HASH
    for i, row in enumerate(rows, 1):
        if row.get("prev_hash") != prev:
            errors.append(f"{name}:{i}: prev_hash does not match previous row")
        if row.get("hash") != row_hash(row):
            errors.append(f"{name}:{i}: hash mismatch (row edited)")
        prev = row.get("hash")


def _check_snapshot(led: Ledger, where: str, row: dict, errors: list[str]) -> None:
    snap = row.get("snapshot")
    if not snap:
        return
    path = led.snapshot_dir / snap.get("file", "")
    if not path.is_file():
        errors.append(f"{where}: snapshot missing ({snap.get('file')})")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != snap.get("sha256"):
        errors.append(f"{where}: snapshot sha256 mismatch ({snap['file']})")


def verify(root: str | Path = "ledger", start_bankroll: float = START_BANKROLL) -> Report:
    led = Ledger(root, start_bankroll)
    rep = Report(bankroll=start_bankroll)
    errors = rep.errors
    try:
        bets, sets = led.bets(), led.settlements()
    except ValueError as e:
        errors.append(str(e))
        return rep

    _check_chain(BETS_FILE, bets, errors)
    _check_chain(SETTLEMENTS_FILE, sets, errors)

    by_hash: dict[str, dict] = {}
    seen_events: set[str] = set()
    for i, b in enumerate(bets, 1):
        where = f"{BETS_FILE}:{i}"
        missing = [k for k in BET_FIELDS if k not in b]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue
        if b["event_id"] in seen_events:
            errors.append(f"{where}: second bet on event {b['event_id']}")
        seen_events.add(b["event_id"])
        try:
            if _parse_ts(b["placed_at"]) >= _parse_ts(b["kickoff"]):
                errors.append(f"{where}: placed at/after kickoff")
        except ValueError as e:
            errors.append(f"{where}: {e}")
        seen = b.get("settlements_seen")
        if seen is not None:
            head = sets[seen - 1].get("hash") if 0 < seen <= len(sets) else (GENESIS_HASH if seen == 0 else None)
            if head != b.get("settlements_head"):
                errors.append(f"{where}: settlements chain no longer matches the head this bet saw")
        if b["selection"] not in SELECTIONS:
            errors.append(f"{where}: bad selection {b['selection']!r}")
        by_hash[b["hash"]] = b

    settled: set[str] = set()
    for i, s in enumerate(sets, 1):
        where = f"{SETTLEMENTS_FILE}:{i}"
        missing = [k for k in SETTLEMENT_FIELDS if k not in s]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue
        bet = by_hash.get(s["bet_hash"])
        if bet is None:
            errors.append(f"{where}: references unknown bet {s['bet_hash'][:12]}")
            continue
        if s["bet_hash"] in settled:
            errors.append(f"{where}: bet {s['bet_hash'][:12]} settled twice")
        settled.add(s["bet_hash"])
        want = {"won": _cents(bet["stake"] * bet["odds"]), "lost": 0.0, "void": bet["stake"]}.get(s["result"])
        if want is None:
            errors.append(f"{where}: bad result {s['result']!r}")
        elif abs(s["returns"] - want) > MONEY_TOL:
            errors.append(f"{where}: returns {s['returns']} should be {want}")
        if s["result"] != "void" and (s["outcome"] == bet["selection"]) != (s["result"] == "won"):
            errors.append(f"{where}: result {s['result']} inconsistent with outcome {s['outcome']}")
        if s.get("clv") is not None:
            rep.clvs.append(s["clv"])

    for i, row in enumerate(bets, 1):
        _check_snapshot(led, f"{BETS_FILE}:{i}", row, errors)
    for i, row in enumerate(sets, 1):
        _check_snapshot(led, f"{SETTLEMENTS_FILE}:{i}", row, errors)

    # Replay cash in time order (settlements first on a tie, as the bot settles before betting).
    events = [(b.get("placed_at", ""), 1, i, b) for i, b in enumerate(bets) if b.get("hash") in by_hash]
    events += [(s.get("settled_at", ""), 0, i, s) for i, s in enumerate(sets) if s.get("bet_hash") in by_hash]
    try:
        events.sort(key=lambda e: (_parse_ts(e[0]), e[1], e[2]))
    except ValueError as e:
        errors.append(f"timestamps: {e}")
        return rep
    cash = start_bankroll
    for ts, kind, i, row in events:
        if kind == 1:
            if abs(row["bankroll_before"] - cash) > MONEY_TOL:
                errors.append(f"{BETS_FILE}:{i + 1}: bankroll_before {row['bankroll_before']} but replay gives {_cents(cash)}")
            cash -= row["stake"]
            rep.staked += row["stake"]
        else:
            cash += row["returns"]
            rep.pnl += row["pnl"]
            if abs(row["bankroll_after"] - cash) > MONEY_TOL:
                errors.append(f"{SETTLEMENTS_FILE}:{i + 1}: bankroll_after {row['bankroll_after']} but replay gives {_cents(cash)}")

    rep.n_bets, rep.n_settled = len(bets), len(settled)
    rep.bankroll = _cents(cash)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the hash-chained betting ledger.")
    ap.add_argument("--dir", default="ledger", help="ledger root (default: ledger)")
    ap.add_argument("--start", type=float, default=START_BANKROLL, help="starting bankroll")
    args = ap.parse_args(argv)
    rep = verify(args.dir, args.start)
    clv = f"{sum(rep.clvs) / len(rep.clvs):+.2%}" if rep.clvs else "n/a"
    print(
        f"{args.dir}: {rep.n_bets} bets, {rep.n_settled} settled, "
        f"staked ${rep.staked:.2f}, settled P&L ${rep.pnl:+.2f}, "
        f"bankroll ${rep.bankroll:.2f}, mean CLV {clv}"
    )
    if rep.ok:
        print("OK: hash chains and bankroll replay verified")
        return 0
    for e in rep.errors:
        print(f"FAIL {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
