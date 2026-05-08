import pandas as pd
import nfl_data_py as nfl

GAME_TYPE_MAP = {
    "REG": "regular",
    "WC": "wildcard",
    "DIV": "divisional",
    "CON": "conference",
    "SB": "superbowl",
}

def load_schedules(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_schedules(seasons)
    df["game_type"] = df["game_type"].map(GAME_TYPE_MAP).fillna("regular")
    df["game_id"] = df["game_id"].astype(str)
    return df

def load_games(seasons: list[int]) -> pd.DataFrame:
    df = load_schedules(seasons)
    completed = df[df["home_score"].notna()].copy()
    completed["home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    return completed

def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_weekly_data(seasons)

def get_team_recent_form(schedules: pd.DataFrame, team: str,
                          season: int, week: int, n: int = 4) -> dict:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) &
        (schedules["week"] < week) &
        (schedules["home_score"].notna())
    )
    recent = schedules[mask].tail(n)
    if recent.empty:
        return {"win_pct": 0.5, "avg_point_diff": 0.0, "games_played": 0}

    wins = 0
    point_diffs = []
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            wins += int(row["home_score"] > row["away_score"])
            point_diffs.append(row["home_score"] - row["away_score"])
        else:
            wins += int(row["away_score"] > row["home_score"])
            point_diffs.append(row["away_score"] - row["home_score"])

    return {
        "win_pct": wins / len(recent),
        "avg_point_diff": sum(point_diffs) / len(point_diffs),
        "games_played": len(recent),
    }

def get_rest_days(schedules: pd.DataFrame, team: str,
                   season: int, week: int) -> int:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) &
        (schedules["week"] < week)
    )
    prior = schedules[mask].tail(1)
    if prior.empty:
        return 14  # assume bye week rest at start of season
    last_date = pd.to_datetime(prior["gameday"].values[0])
    current = schedules[
        (((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
         (schedules["season"] == season) & (schedules["week"] == week))
    ]
    if current.empty:
        return 7
    game_date = pd.to_datetime(current["gameday"].values[0])
    return max(1, (game_date - last_date).days)
