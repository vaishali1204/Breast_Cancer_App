
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
)
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
)

st.title("🎗️ Breast Cancer Classification")
st.write(
    "Enter the 30 features of a breast cell sample to get a prediction "
    "from the trained CNN model."
)

st.warning(
    "Educational demonstration only. This model is not a medical diagnostic "
    "tool and must not be used to make healthcare decisions."
)

# --------------------------------------------------
# Project files
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = BASE_DIR / "model_weights.h5"
MEAN_PATH = BASE_DIR / "scaler_mean.npy"
SCALE_PATH = BASE_DIR / "scaler_scale.npy"


# --------------------------------------------------
# Rebuild the CNN architecture used during training
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    inputs = Input(shape=(30, 1))

    x = Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = MaxPooling1D(pool_size=2)(x)
    flat = Flatten()(x)

    dense1 = Dense(64, activation="relu")(flat)
    concat1 = Concatenate()([flat, dense1])

    dense2 = Dense(64, activation="relu")(concat1)
    concat2 = Concatenate()([concat1, dense2])

    dense3 = Dense(64, activation="relu")(concat2)
    concat3 = Concatenate()([concat2, dense3])

    dense4 = Dense(64, activation="relu")(concat3)
    x = Dropout(0.3)(dense4)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {WEIGHTS_PATH.name}")

    if not MEAN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MEAN_PATH.name}")

    if not SCALE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {SCALE_PATH.name}")

    model.load_weights(str(WEIGHTS_PATH))

    scaler_mean = np.load(MEAN_PATH)
    scaler_scale = np.load(SCALE_PATH)

    if len(scaler_mean) != 30 or len(scaler_scale) != 30:
        raise ValueError("Scaler files must each contain exactly 30 values.")

    if np.any(scaler_scale == 0):
        raise ValueError("A scaler value is zero. Check the saved scaler files.")

    return model, scaler_mean, scaler_scale


# --------------------------------------------------
# Feature names in the original dataset order
# --------------------------------------------------
feature_groups = {
    "Mean Features": [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
    ],
    "Standard Error Features": [
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
    ],
    "Worst Features": [
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ],
}

feature_names = [
    feature
    for group in feature_groups.values()
    for feature in group
]

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
try:
    model, scaler_mean, scaler_scale = load_artifacts()
    st.success("Trained model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Could not load the model or scaler: {e}")
    st.info(
        "Check that model_weights.h5, scaler_mean.npy, and "
        "scaler_scale.npy are in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter Sample Measurements")

st.caption(
    "Enter measurements from a real sample in the same units and feature "
    "order used during training. Do not leave fields empty."
)

with st.form("prediction_form"):
    entered_values = {}

    for group_name, group_features in feature_groups.items():
        st.markdown(f"#### {group_name}")
        columns = st.columns(2)

        for index, feature in enumerate(group_features):
            with columns[index % 2]:
                entered_values[feature] = st.number_input(
                    label=feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=None,
                    placeholder="Enter measurement",
                    step=0.01,
                    format="%.6f",
                    key=feature,
                )

    submitted = st.form_submit_button(
        "Predict Classification",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    missing_features = [
        feature
        for feature in feature_names
        if entered_values[feature] is None
    ]

    if missing_features:
        st.error(
            f"Please enter all 30 measurements. "
            f"{len(missing_features)} field(s) are missing."
        )

    else:
        try:
            # Preserve the original training feature order.
            sample = np.array(
                [entered_values[name] for name in feature_names],
                dtype=np.float32,
            ).reshape(1, 30)

            # Apply the same standardization used during training.
            sample_scaled = (sample - scaler_mean) / scaler_scale

            # CNN input shape: (batch_size, 30 features, 1 channel)
            sample_reshaped = sample_scaled.reshape(1, 30, 1)

            prediction = float(
                model.predict(sample_reshaped, verbose=0)[0][0]
            )

            st.subheader("Model Output")

            st.metric(
                label="Sigmoid Output",
                value=f"{prediction:.6f}",
            )

            st.progress(min(max(prediction, 0.0), 1.0))

            st.caption(
                "The sigmoid output ranges from 0 to 1. Its interpretation "
                "depends on the label mapping and calibration of the trained model."
            )

            # Assumes the training labels were encoded as:
            # Benign = 0, Malignant = 1.
            if prediction >= 0.5:
                st.error("Model class: Malignant (M)")
            else:
                st.success("Model class: Benign (B)")

            st.caption(
                "This is the model's classification, not a clinical diagnosis. "
                "A qualified healthcare professional must interpret any "
                "real medical findings."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
