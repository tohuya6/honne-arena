from app.points import best_per, dynamic_value, rank_value


def test_first_solve_is_full_value():
    assert dynamic_value(500, 100, 10, 0) == 500
    assert dynamic_value(500, 100, 10, 1) == 500


def test_value_falls_to_floor_after_decay_solves():
    values = [dynamic_value(500, 100, 10, n) for n in range(1, 15)]
    assert values == sorted(values, reverse=True)
    assert dynamic_value(500, 100, 10, 11) == 100   # n-1 = decay hits the floor
    assert dynamic_value(500, 100, 10, 50) == 100


def test_zero_decay_means_static_points():
    assert dynamic_value(300, 100, 0, 40) == 300


def test_rank_value_linear():
    assert rank_value(500, 100, 1, 5) == 500
    assert rank_value(500, 100, 5, 5) == 100
    assert rank_value(500, 100, 3, 5) == 300
    assert rank_value(500, 100, 1, 1) == 500


def test_best_per_keeps_best_and_breaks_ties_by_time():
    rows = [
        {"id": 1, "player_id": "a", "score": 0.2, "created_at": "1"},
        {"id": 2, "player_id": "a", "score": 0.5, "created_at": "2"},
        {"id": 3, "player_id": "b", "score": 0.5, "created_at": "3"},
    ]
    best = best_per(rows, "player_id", higher_is_better=True)
    assert [(r["player_id"], r["score"]) for r in best] == [("a", 0.5), ("b", 0.5)]
    low = best_per(rows, "player_id", higher_is_better=False)
    assert [(r["player_id"], r["score"]) for r in low] == [("a", 0.2), ("b", 0.5)]
