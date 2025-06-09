import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Welding Spectrum Explorer", layout="wide")
st.title("🔍 Welding Spectrum Explorer")

# --- Sidebar Configuration ---
st.sidebar.header("Visualization Controls")

num_rows_to_plot = st.sidebar.slider("Number of signal rows to overlay in line plot:", 1, 50, 10)
show_mean_spectrum = st.sidebar.checkbox("Show Mean Spectrum Comparison", value=True)
show_difference_plot = st.sidebar.checkbox("Show Difference Plot", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
show_pca = st.sidebar.checkbox("Show PCA Projection", value=False)

uploaded_ok = st.sidebar.file_uploader("Upload OK Welding CSV", type="csv")
uploaded_nok = st.sidebar.file_uploader("Upload NOK Welding CSV", type="csv")

# --- Helper Functions ---
def load_spectral_csv(file):
    df = pd.read_csv(file, header=None)
    wavelengths = df.iloc[0].astype(float).values
    data = df.iloc[1:].astype(float).reset_index(drop=True)
    return wavelengths, data

def plot_line_overlay(wavelengths, data, label, num_lines):
    sampled = data.sample(min(num_lines, len(data)), random_state=42)
    for i in range(len(sampled)):
        plt.plot(wavelengths, sampled.iloc[i], alpha=0.5, label=f"{label} {i+1}" if i == 0 else None)

def plot_heatmap(data, wavelengths, title):
    plt.figure(figsize=(12, 5))
    sns.heatmap(data.values, cmap="viridis", cbar_kws={'label': 'Signal Intensity'})
    plt.title(title)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Signal Index")
    plt.xticks(ticks=np.linspace(0, len(wavelengths)-1, 10), labels=np.round(np.linspace(wavelengths[0], wavelengths[-1], 10)).astype(int))
    st.pyplot(plt.gcf())

def plot_pca(ok_data, nok_data):
    combined = pd.concat([ok_data, nok_data], ignore_index=True)
    labels = np.array(["OK"] * len(ok_data) + ["NOK"] * len(nok_data))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)

    df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    df_pca["Label"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Label", alpha=0.7)
    plt.title("PCA Projection of OK vs NOK Spectra")
    st.pyplot(plt.gcf())

# --- Main Content ---
if uploaded_ok and uploaded_nok:
    wavelengths_ok, data_ok = load_spectral_csv(uploaded_ok)
    wavelengths_nok, data_nok = load_spectral_csv(uploaded_nok)

    if not np.allclose(wavelengths_ok, wavelengths_nok, atol=1e-6):
        st.error("Wavelengths differ between OK and NOK files. Please verify matching first rows.")
        st.write("First 10 from OK:", wavelengths_ok[:10])
        st.write("First 10 from NOK:", wavelengths_nok[:10])
        st.stop()

    wavelengths = wavelengths_ok

    st.subheader("Line Overlay Plot of Spectral Data")
    plt.figure(figsize=(12, 6))
    plot_line_overlay(wavelengths, data_ok, "OK", num_rows_to_plot)
    plot_line_overlay(wavelengths, data_nok, "NOK", num_rows_to_plot)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Signal Intensity")
    plt.legend()
    st.pyplot(plt.gcf())

    if show_mean_spectrum:
        st.subheader("Mean Spectrum Comparison")
        mean_ok = data_ok.mean(axis=0)
        mean_nok = data_nok.mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, mean_ok, label="OK Mean", color="green")
        plt.plot(wavelengths, mean_nok, label="NOK Mean", color="red")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean Signal Intensity")
        plt.legend()
        st.pyplot(plt.gcf())

    if show_difference_plot:
        st.subheader("Difference Plot (NOK - OK)")
        diff = data_nok.mean(axis=0) - data_ok.mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, diff, label="Difference", color="purple")
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Difference in Mean Intensity")
        plt.legend()
        st.pyplot(plt.gcf())

    if show_heatmap:
        st.subheader("Heatmaps")
        plot_heatmap(data_ok, wavelengths, "OK Welding Heatmap")
        plot_heatmap(data_nok, wavelengths, "NOK Welding Heatmap")

    if show_pca:
        st.subheader("PCA Dimensionality Reduction")
        plot_pca(data_ok, data_nok)

else:
    st.info("Please upload both OK and NOK welding CSV files to begin analysis.")
