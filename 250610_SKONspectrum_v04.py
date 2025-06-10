import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Welding Spectrum Explorer", layout="wide")
st.title("🔍 Welding Spectrum Explorer")

# --- Sidebar Configuration ---
st.sidebar.header("Visualization Controls")

num_rows_to_plot = st.sidebar.slider("Number of signal rows to overlay in line plot:", 1, 150, 10)
show_mean_spectrum = st.sidebar.checkbox("Show Mean Spectrum Comparison", value=True)
show_difference_plot = st.sidebar.checkbox("Show Difference Plot", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
show_pca = st.sidebar.checkbox("Show PCA Projection", value=False)
show_band_integration = st.sidebar.checkbox("Show Band Integration Tool", value=False)

band_start = st.sidebar.number_input("Band Start Wavelength (nm)", value=900.0)
band_end = st.sidebar.number_input("Band End Wavelength (nm)", value=1100.0)

uploaded_ok = st.sidebar.file_uploader("Upload OK Welding CSV", type="csv")
uploaded_nok = st.sidebar.file_uploader("Upload NOK Welding CSV", type="csv")

# --- Helper Functions ---
def load_spectral_csv(file):
    df = pd.read_csv(file, header=None)
    wavelengths = df.iloc[0].astype(str).str.strip().astype(float).values
    data = df.iloc[1:].astype(float).reset_index(drop=True)
    return wavelengths, data

def plot_line_overlay_plotly(wavelengths, data_ok, data_nok, num_lines):
    fig = go.Figure()
    for i in range(min(num_lines, len(data_ok))):
        fig.add_trace(go.Scatter(x=wavelengths, y=data_ok.iloc[i], mode='lines', name=f'OK {i+1}', line=dict(color='green', width=1)))
    for i in range(min(num_lines, len(data_nok))):
        fig.add_trace(go.Scatter(x=wavelengths, y=data_nok.iloc[i], mode='lines', name=f'NOK {i+1}', line=dict(color='red', width=1)))
    fig.update_layout(title="Line Overlay Plot", xaxis_title="Wavelength (nm)", yaxis_title="Signal Intensity")
    st.plotly_chart(fig, use_container_width=True)

def plot_mean_comparison_plotly(wavelengths, mean_ok, mean_nok):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=mean_ok, mode='lines', name='OK Mean', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=wavelengths, y=mean_nok, mode='lines', name='NOK Mean', line=dict(color='red')))
    fig.update_layout(title="Mean Spectrum Comparison", xaxis_title="Wavelength (nm)", yaxis_title="Mean Signal Intensity")
    st.plotly_chart(fig, use_container_width=True)

def plot_difference_plotly(wavelengths, mean_ok, mean_nok):
    diff = mean_nok - mean_ok
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=diff, mode='lines', name='Difference', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=wavelengths, y=[0]*len(wavelengths), mode='lines', line=dict(color='black', dash='dash'), showlegend=False))
    fig.update_layout(title="Difference Plot (NOK - OK)", xaxis_title="Wavelength (nm)", yaxis_title="Difference in Mean Intensity")
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_plotly(data, wavelengths, title):
    fig = px.imshow(data.values, aspect='auto', color_continuous_scale='Viridis',
                    labels=dict(x="Wavelength Index", y="Signal Index", color="Intensity"),
                    x=wavelengths)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_band_integration(wavelengths, data_ok, data_nok, band_start, band_end):
    band_mask = (wavelengths >= band_start) & (wavelengths <= band_end)
    auc_ok = data_ok.iloc[:, band_mask].sum(axis=1)
    auc_nok = data_nok.iloc[:, band_mask].sum(axis=1)

    df_auc = pd.DataFrame({
        "AUC": np.concatenate([auc_ok, auc_nok]),
        "Label": ["OK"]*len(auc_ok) + ["NOK"]*len(auc_nok)
    })
    fig = px.violin(df_auc, y="AUC", color="Label", box=True, points="all")
    fig.update_layout(title=f"AUC Comparison for Band {band_start}-{band_end} nm")
    st.plotly_chart(fig, use_container_width=True)

def plot_pca_with_selection(ok_data, nok_data, wavelengths):
    combined = pd.concat([ok_data, nok_data], ignore_index=True)
    labels = np.array(["OK"] * len(ok_data) + ["NOK"] * len(nok_data))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined.fillna(0))

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)

    df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    df_pca["Label"] = labels
    df_pca["Index"] = df_pca.index

    fig = px.scatter(df_pca, x="PC1", y="PC2", color="Label", hover_data=["Index"])
    st.plotly_chart(fig, use_container_width=True)

    selected_indices = st.multiselect("Select rows to visualize their spectra:", options=df_pca["Index"].tolist())
    if selected_indices:
        fig_spectra = go.Figure()
        for idx in selected_indices:
            row = combined.iloc[idx]
            label = labels[idx]
            fig_spectra.add_trace(go.Scatter(x=wavelengths, y=row, mode='lines', name=f"{label} {idx}"))
        fig_spectra.update_layout(title="Spectra of Selected PCA Points", xaxis_title="Wavelength (nm)", yaxis_title="Signal Intensity")
        st.plotly_chart(fig_spectra, use_container_width=True)

def show_single_weld_plot(wavelengths, data, label):
    idx = st.sidebar.slider(f"Select a specific {label} row to inspect:", 0, len(data)-1, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavelengths, y=data.iloc[idx], mode='lines', name=f"{label} Row {idx}"))
    fig.update_layout(title=f"Single Welding Spectrum from {label} Data (Row {idx})", xaxis_title="Wavelength (nm)", yaxis_title="Signal Intensity")
    st.plotly_chart(fig, use_container_width=True)

# --- Main Content ---
if uploaded_ok and uploaded_nok:
    wavelengths_ok, data_ok = load_spectral_csv(uploaded_ok)
    wavelengths_nok, data_nok = load_spectral_csv(uploaded_nok)

    if not np.allclose(wavelengths_ok, wavelengths_nok, rtol=0, atol=1e-5):
        st.warning("Wavelengths are very slightly different between files, likely due to formatting. Proceeding with analysis anyway.")

    wavelengths = wavelengths_ok

    st.subheader("Line Overlay Plot of Spectral Data")
    plot_line_overlay_plotly(wavelengths, data_ok, data_nok, num_rows_to_plot)

    if show_mean_spectrum:
        st.subheader("Mean Spectrum Comparison")
        mean_ok = data_ok.mean(axis=0)
        mean_nok = data_nok.mean(axis=0)
        plot_mean_comparison_plotly(wavelengths, mean_ok, mean_nok)

    if show_difference_plot:
        st.subheader("Difference Plot (NOK - OK)")
        plot_difference_plotly(wavelengths, mean_ok, mean_nok)

    if show_heatmap:
        st.subheader("Heatmaps")
        plot_heatmap_plotly(data_ok, wavelengths, "OK Welding Heatmap")
        plot_heatmap_plotly(data_nok, wavelengths, "NOK Welding Heatmap")

    if show_pca:
        st.subheader("PCA Dimensionality Reduction with Interactive Selection")
        try:
            plot_pca_with_selection(data_ok, data_nok, wavelengths)
        except Exception as e:
            st.error(f"PCA failed: {e}")

    if show_band_integration:
        st.subheader("AUC over Custom Wavelength Band")
        plot_band_integration(wavelengths, data_ok, data_nok, band_start, band_end)

    st.subheader("🔍 Explore Individual Weldings")
    show_single_weld_plot(wavelengths, data_ok, "OK")
    show_single_weld_plot(wavelengths, data_nok, "NOK")

else:
    st.info("Please upload both OK and NOK welding CSV files to begin analysis.")
