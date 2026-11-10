"""Streamlit interface for predictions and SHAP explanations."""

import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.evaluate import _pipeline_shap_values
from src.preprocessing import MENTAL_HEALTH_TARGETS

from app.prediction import load_target_models, predict_scores

st.set_page_config(
    page_title="Musichology Mental Health Predictor",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 Musichology Mental Health Predictor")
st.warning(
    "Disclaimer: This is not a diagnostic tool. Predictions are estimates from "
    "self-reported survey data and should not guide medical decisions."
)
st.write("Enter listening habits to estimate four self-reported scores from 0 to 10.")


def _input_values() -> dict[str, object]:
    """Collect the core listening-habit inputs shown in the UI."""
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0)
        hours = st.number_input(
            "Hours per day",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
        )
        genre = st.text_input("Favorite genre", value="Rock")
        service = st.text_input("Primary streaming service", value="Spotify")
        bpm = st.number_input("Typical BPM", min_value=0.0, max_value=400.0, value=120.0)
        music_effects = st.selectbox(
            "Perceived effect of music",
            ["Improve", "No effect", "Worsen"],
        )
        submitted = st.form_submit_button("Predict")
    if not submitted:
        return {}
    return {
        "Age": age,
        "Hours per day": hours,
        "Fav genre": genre,
        "Primary streaming service": service,
        "BPM": bpm,
        "Music effects": music_effects,
    }


values = _input_values()
if values:
    try:
        scores, feature_frame = predict_scores(values)
        models, _ = load_target_models()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
    else:
        columns = st.columns(len(MENTAL_HEALTH_TARGETS))
        for column, target in zip(columns, MENTAL_HEALTH_TARGETS):
            column.metric(target, f"{scores[target]:.2f} / 10")

        st.subheader("Why did the model make these predictions?")
        st.caption(
            "SHAP values show how each encoded feature moved a prediction relative "
            "to that target model's baseline; they do not establish causation."
        )
        for target in MENTAL_HEALTH_TARGETS:
            with st.expander(f"{target} explanation", expanded=True):
                explanation, shap_values, feature_names, transformed = (
                    _pipeline_shap_values(models[target], feature_frame)
                )
                waterfall = shap.Explanation(
                    values=shap_values[0],
                    base_values=explanation.base_values[0],
                    data=transformed[0],
                    feature_names=feature_names,
                )
                figure = plt.figure()
                shap.plots.waterfall(waterfall, max_display=12, show=False)
                st.pyplot(figure, clear_figure=True)
