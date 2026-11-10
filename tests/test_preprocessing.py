"""Tests for notebook-compatible preprocessing."""

from pathlib import Path

from src.preprocessing import load_data, prepare_linear_data, prepare_neural_data


DATA_PATH = Path(__file__).parents[1] / "data" / "music-mental.csv"


def test_linear_preprocessing_matches_notebook_filters() -> None:
    """Linear preprocessing removes metadata and applies notebook filters."""
    cleaned = prepare_linear_data(load_data(DATA_PATH))

    assert len(cleaned) == 529
    assert "Timestamp" not in cleaned.columns
    assert "Permissions" not in cleaned.columns
    assert "Foreign languages" not in cleaned.columns
    assert cleaned["Age"].between(15, 70).all()
    assert cleaned["Hours per day"].between(1, 10, inclusive="left").all()
    assert cleaned["BPM"].between(20, 500, inclusive="neither").all()


def test_neural_preprocessing_encodes_genre_and_targets() -> None:
    """Neural preprocessing returns encoded genres and complete rows."""
    prepared, encoder = prepare_neural_data(load_data(DATA_PATH))

    assert len(prepared) == 616
    assert "Timestamp" not in prepared.columns
    assert "Permissions" not in prepared.columns
    assert prepared["Fav genre"].dtype.kind in "iu"
    assert len(encoder.classes_) == 16
    assert prepared[["Anxiety", "Depression", "Insomnia", "OCD"]].notna().all().all()

