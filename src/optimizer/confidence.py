def get_point_range(n_games: int, game_type: str = "regular") -> tuple[int, int]:
    # game_type differentiation not yet implemented; all types use 1..n
    return (1, n_games)


def assign_confidence_points(
    games: list[dict],
    point_range: tuple[int, int],
    uncertainty_threshold: float = 0.03,
) -> list[dict]:
    n = len(games)
    if n == 0:
        return []

    sorted_games = sorted(games, key=lambda g: g["win_probability"], reverse=True)

    # Always assign 1..n; highest probability gets n points (rearrangement inequality)
    point_assignments = list(range(n, 0, -1))

    result = []
    for i, game in enumerate(sorted_games):
        pts = point_assignments[i]

        is_uncertain = False
        if i > 0:
            prev_prob = sorted_games[i - 1]["win_probability"]
            if abs(prev_prob - game["win_probability"]) <= uncertainty_threshold:
                is_uncertain = True
        if i < len(sorted_games) - 1:
            next_prob = sorted_games[i + 1]["win_probability"]
            if abs(next_prob - game["win_probability"]) <= uncertainty_threshold:
                is_uncertain = True

        result.append({
            **game,
            "confidence_points": pts,
            "is_uncertain": is_uncertain,
        })

    return result
