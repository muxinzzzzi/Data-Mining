from __future__ import annotations

import pandas as pd


def build_main_strategy_attribution(strategy_daily: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    df = strategy_daily.loc[strategy_daily["strategy"] == strategy_name].sort_values("date").copy()
    if df.empty:
        raise ValueError(f"No strategy rows found for attribution: {strategy_name}")
    position_gap = 1.0 - df["final_position"]
    df["missed_upside"] = (position_gap.clip(lower=0.0) * df["buy_hold_return"].clip(lower=0.0)).fillna(0.0)
    df["avoided_downside"] = (position_gap.clip(lower=0.0) * (-df["buy_hold_return"].clip(upper=0.0))).fillna(0.0)
    df["gross_position_timing_contribution"] = ((df["final_position"] - 1.0) * df["buy_hold_return"]).fillna(0.0)
    df["net_position_timing_contribution"] = df["gross_position_timing_contribution"] - df["transaction_cost"].fillna(0.0)
    df["defensive_day"] = df["final_position"] < 0.995
    return df[
        [
            "date",
            "trade_date",
            "strategy",
            "final_position",
            "signal",
            "buy_hold_return",
            "strategy_return",
            "active_return",
            "turnover",
            "transaction_cost",
            "missed_upside",
            "avoided_downside",
            "gross_position_timing_contribution",
            "net_position_timing_contribution",
            "defensive_day",
            "pred_big_up_prob",
            "pred_big_down_prob",
            "pred_tail_score",
        ]
    ]


def write_attribution_summary(attribution: pd.DataFrame, path) -> None:
    missed_upside = float(attribution["missed_upside"].sum())
    avoided_downside = float(attribution["avoided_downside"].sum())
    transaction_cost = float(attribution["transaction_cost"].sum())
    net_contribution = float(attribution["net_position_timing_contribution"].sum())
    defensive_days = attribution.loc[attribution["defensive_day"]].copy()
    best = defensive_days.sort_values("net_position_timing_contribution", ascending=False).head(10)
    worst = defensive_days.sort_values("net_position_timing_contribution", ascending=True).head(10)

    def _table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No defensive days."
        cols = ["date", "final_position", "buy_hold_return", "net_position_timing_contribution", "pred_big_up_prob", "pred_big_down_prob"]
        out = df[cols].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for row in out.itertuples(index=False):
            values = []
            for value in row:
                values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    text = f"""# Main Strategy Attribution

## Summary
- Missed upside from reduced exposure on positive-return days: {missed_upside:.6f}
- Avoided downside from reduced exposure on negative-return days: {avoided_downside:.6f}
- Transaction cost: {transaction_cost:.6f}
- Net position-timing contribution: {net_contribution:.6f}

## Top 10 Best Defensive Days
{_table(best)}

## Top 10 Worst Defensive Days
{_table(worst)}
"""
    path.write_text(text, encoding="utf-8")
