def get_point_range(n_games: int, game_type: str = "regular") -> tuple[int, int]:
    return (1, n_games)


def assign_confidence_points(
    games: list[dict],
    point_range: tuple[int, int],
    uncertainty_threshold: float = 0.03,
) -> list[dict]:
    min_pts, max_pts = point_range
    n = len(games)
    points = list(range(min_pts, max_pts + 1))
    if len(points) < n:
        points = list(range(1, n + 1))

    sorted_games = sorted(games, key=lambda g: g["win_probability"], reverse=True)

    # highest probability gets highest points (rearrangement inequality)
    point_assignments = list(reversed(points[-n:]))

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
