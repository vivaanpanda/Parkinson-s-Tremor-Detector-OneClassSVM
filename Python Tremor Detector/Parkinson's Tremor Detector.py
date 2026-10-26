import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from pathlib import Path


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
st.title("Parkinson's Classifier", text_alignment="center")
st.divider()
st.markdown("This OneClassSVM-based ML Classification tool requires the input of two CSV's -- each containing rest tremor and procedural tremor linear accelerometer data -- to theoretically check for Parkinson's. This tool is most useful if MRI's or other neurological scanning technology is inaccessible.")
st.markdown("Be aware: :color[this is only meant to be used as an experimental classification tool based on available Parkinson's research, which is fairly limited at the moment.]{foreground='rgba(255,0,0,1)'} If you believe you or anyone else has Parkinson's, please contact a medical professional in your area.")
required_col = ["time", "ax", "ay", "az"]
base_dir = Path(__file__).resolve().parent
path = base_dir / "Training Data" / "ALAMEDA_PD_tremor_dataset.csv"

def calculate_frequency(analysis, tremor_type):
    dominant_freq = 0
    max_power = 0
    pc1_rms = 0
    pc1_std = 0
    zero_crossings = 0
    zero_crossings_rate = 0
    analysis = analysis[required_col]
    if "ax" in analysis.columns and "time" in analysis.columns and "ay" in analysis.columns and "az" in analysis.columns:
        time = analysis["time"].values
        ax_data = analysis["ax"].values
        ax_data = signal.detrend(ax_data, axis = -1, type= 'linear', bp =0, overwrite_data = False)
        ay_data = analysis["ay"].values
        ay_data = signal.detrend(ay_data, axis = -1, type= 'linear', bp =0, overwrite_data = False)
        az_data = analysis["az"].values
        az_data = signal.detrend(az_data, axis = -1, type= 'linear', bp =0, overwrite_data = False)
        PCA_data = np.column_stack((ax_data, ay_data, az_data))
        pca = PCA(n_components=1)
        principal_component_1 = pca.fit_transform(PCA_data).flatten()
        #Tests multiple frequencies in hertz
        freqs_hz = np.linspace(2.5, 12.5, 1000)
        #Tests those same frequencies in angular_freqs, which are the acutal input in cos/sin graphs to get a baseline of time, t.
        angular_freqs = 2 * np.pi * freqs_hz

        # Compute Lomb-Scargle periodogram
        power = signal.lombscargle(time, principal_component_1, angular_freqs, normalize=False)
        #The highest power calculated using teh Lomb-Scargle periodogram is used to return the value of the index 
        max_power_raw = np.max(power)
        max_power = np.sqrt(max_power_raw) ##this is a limit of the code -- sqrt is just a method to "trim down" the raw power so that the range of values from the training data matches the calculated ones.
        tremor_threshold = 1.0
    # Display raw data safely
        if max_power > tremor_threshold:
                max_value_index = np.argmax(power) #argmax() returns index of maximum number within array.
                dominant_freq = freqs_hz[max_value_index]
        else:
                dominant_freq = 0
        pc1_std = np.std(principal_component_1)  
        pc1_rms = np.sqrt(
            np.mean(principal_component_1**2)
        )
        zero_crossings = np.sum(
            np.diff(np.sign(principal_component_1)) != 0
        )
        zero_crossings_rate = zero_crossings / len(principal_component_1)

    return dominant_freq, max_power, pc1_rms, pc1_std, zero_crossings_rate


tr_data = pd.read_csv(path)
train_columns = [
    "PC1_fft_dom_freq",
    "PC1_fft_max",
    "PC1_fft_rms",
    "PC1_fft_std_dev",
    "PC1_zero_cross_rt",
]

def cleandata_postural(df):
  # Keep only windows where clinical tremor was actually present
  df_filtered = df[df["Postural_tremor"] == 1].copy()
  return df_filtered[train_columns].dropna()

def cleandata_rest(df):
  # Keep only windows where clinical tremor was actually present
  df_filtered = df[df["Rest_tremor"] == 1].copy()
  return df_filtered[train_columns].dropna()


tr_data_postural = cleandata_postural(tr_data)
tr_data_rest = cleandata_rest(tr_data)
@st.cache_resource
def train_model(tremor_data):
    scaler = StandardScaler()

    X_train = tremor_data[train_columns].values
    X_train_scaled = scaler.fit_transform(X_train)

    model = OneClassSVM(kernel = 'rbf', nu=0.01, gamma = 0.01)
    trained_model = model.fit(X_train_scaled)
    trained_model.predict(X_train_scaled)
    return model, scaler
postural_model, postural_scaler = train_model(tr_data_postural)
rest_model, rest_scaler = train_model(tr_data_rest)

##Training Data: dom_Freq, max_power, pc1_std/pc1_rms, zcr; PC1_fft_dom_freq, PC1_fft_max, PC1_fft_rms, PC1_fft_std_dev, PC1_zero_cross_rt

def run_model(df, tremor_type):
    scalers = {
        "postural": postural_scaler,
        "rest": rest_scaler
    }
    models = {
        "postural": postural_model,
        "rest": rest_model
    }
    scaled_data = scalers.get(tremor_type).transform(df)
    return models.get(tremor_type).predict(scaled_data)[0]




tab1, tab2 = st.tabs(["Tool", "Other"])

with st.sidebar:
    with tab2:
        st.title("Supplementary Information")
        st.subheader("Research Reference:")
        st.subheader("Citation Document:")
        st.subheader("Data Acquisition Methodology:")
        st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ol, [data-testid="stMarkdownContainer"] ul {
    padding-left:40px;
}
</style>
1. Acquire raw tremor time-series data utilizing the **Physics Toolbox Sensor Suite** application across appropriate mobile deployment platforms:
''', unsafe_allow_html=True)
        with st.container(horizontal=True):
            st.link_button("Google Play", "https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite&hl=en_US&pli=1")
            st.link_button("App Store", "https://apps.apple.com/us/app/physics-toolbox-sensor-suite/id1128914250")
        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ol, [data-testid="stMarkdownContainer"] ul {
            padding-left:40px;
        }
        </style>
        2. Verify that the exported dataset structurally contains the mandatory parameters: `time` (temporal tracking), and `ax`, `ay`, `az` (triaxial linear acceleration components).
        ''', unsafe_allow_html=True)

    with tab1:
        postural_tremor_uploaded_file = st.file_uploader("Upload the postural tremor file")
        rest_tremor_uploaded_file = st.file_uploader("Upload the rest tremor file")
        if postural_tremor_uploaded_file is not None and rest_tremor_uploaded_file is not None:
            df_postural = pd.read_csv(postural_tremor_uploaded_file, comment = '#')
            df_rest = pd.read_csv(rest_tremor_uploaded_file, comment = '#')
            df_postural.columns = df_postural.columns.str.strip().str.lower()
            df_rest.columns = df_rest.columns.str.strip().str.lower()
            if set(required_col).issubset(df_postural.columns) and set(required_col).issubset(df_rest.columns):
                df_postural = df_postural[required_col]
                df_rest = df_rest[required_col]
            elif df_postural.shape[1] >= 4 and df_rest.shape[1] >= 4:
                df_postural.columns = required_col
                df_rest.columns = required_col

            else:
                st.text("Your data has invalid columns; please use 'time', 'ax', 'ay', and 'az'")
                st.stop()
            
            p_tremor_data = np.array(calculate_frequency(df_postural, "postural")).reshape(1, -1)
            r_tremor_data = np.array(calculate_frequency(df_rest, "rest")).reshape(1, -1)
            if p_tremor_data[0,0] == 0 and r_tremor_data[0,0] == 0:
                st.markdown("Both of your files show that the patient :red[doesn't] have any tremor")
            else:
                raw_prediction_p = run_model(p_tremor_data, "postural")
                raw_prediction_r = run_model(r_tremor_data, "rest")
                st.markdown(f"The calculated values are {p_tremor_data[0, 0]:.2f} Hz and {r_tremor_data[0, 0]:.2f} Hz")
                st.markdown(f"The calculated power values are {p_tremor_data[0,1]:.2f} and {r_tremor_data[0,1]:.2f}")
                if raw_prediction_r == 1 or raw_prediction_p == 1:
                    st.header("Results:")
                    st.markdown(":red[Parkinson's]")
                else:
                    st.header("Results:")
                    st.markdown(":green[Not] Parkinson's")
        else:
            st.text("Please upload a file")