import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Welding Spectrum Explorer", layout="wide")
st.title("🔍 Welding Spectrum Explorer")

# --- Sidebar Configuration ---
st.sidebar.header("Visualization Controls")

num_rows_to_plot = st.sidebar.slider("Number of signal rows to overlay in line plot:", 1, 150, 10)
show_mean_spectrum = st.sidebar.checkbox("Show Mean Spectrum Comparison", value=True)
show_difference_plot = st.sidebar.checkbox("Show Difference Plot", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
show_time_series = st.sidebar.checkbox("Explore Signal Evolution Over Time", value=False)
show_classic_spectrogram = st.sidebar.checkbox("Show Classic Spectrogram (2D Heatmap)", value=False)


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

def filter_band(wavelengths, data, band_start, band_end):
    band_mask = (wavelengths >= band_start) & (wavelengths <= band_end)
    return wavelengths[band_mask], data.iloc[:, band_mask]

def plot_line_overlay_plotly(wavelengths, data_ok, data_nok, num_lines):
    fig = go.Figure()
    for i in range(min(num_lines, len(data_ok))):
        fig.add_trace(go.Scatter(x=wavelengths, y=data_ok.iloc[i], mode='lines', name=f'OK {i+1}', line=dict(color='green', width=1)))
    for i in range(min(num_lines, len(data_nok))):
        fig.add_trace(go.Scatter(x=wavelengths, y=data_nok.iloc[i], mode='lines', name=f'NOK {i+1}', line=dict(color='red', width=1)))
    fig.update_layout(title="Line Overlay Plot", xaxis_title="Wavelength (nm)", yaxis_title="Signal Intensity")
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_combined_lines(wavelengths, data_ok, data_nok):
    fig = go.Figure()
    for i in range(len(data_ok)):
        fig.add_trace(go.Scatter3d(x=wavelengths, y=[i]*len(wavelengths), z=data_ok.iloc[i], mode='lines', line=dict(color='green'), name=f'OK {i}'))
    for i in range(len(data_nok)):
        fig.add_trace(go.Scatter3d(x=wavelengths, y=[i]*len(wavelengths), z=data_nok.iloc[i], mode='lines', line=dict(color='red'), name=f'NOK {i}'))
    fig.update_layout(title="3D Line Plot: OK and NOK Combined", scene=dict(xaxis_title='Wavelength', yaxis_title='Time Index', zaxis_title='Intensity'))
    st.plotly_chart(fig, use_container_width=True)

def plot_surface(data, wavelengths, title):
    fig = go.Figure(data=[go.Surface(z=data.values, x=wavelengths, y=np.arange(len(data)))])
    fig.update_layout(title=title, scene=dict(xaxis_title='Wavelength', yaxis_title='Time Index', zaxis_title='Intensity'))
    st.plotly_chart(fig, use_container_width=True)

def plot_difference_surface(data_ok, data_nok, wavelengths):
    min_len = min(len(data_ok), len(data_nok))
    diff = data_nok.iloc[:min_len].values - data_ok.iloc[:min_len].values
    fig = go.Figure(data=[go.Surface(z=diff, x=wavelengths, y=np.arange(min_len))])
    fig.update_layout(title="Difference Surface (NOK - OK)", scene=dict(xaxis_title='Wavelength', yaxis_title='Time Index', zaxis_title='Intensity Difference'))
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
                    labels=dict(x="Wavelength (nm)", y="Time Index", color="Intensity"),
                    x=wavelengths)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series(data_ok, data_nok, wavelengths, selected_wavelengths):
    fig = go.Figure()
    for wl in selected_wavelengths:
        idx = np.argmin(np.abs(wavelengths - wl))
        fig.add_trace(go.Scatter(y=data_ok.iloc[:, idx], mode='lines', name=f"OK {wl} nm", line=dict(color='green')))
        fig.add_trace(go.Scatter(y=data_nok.iloc[:, idx], mode='lines', name=f"NOK {wl} nm", line=dict(color='red')))
    fig.update_layout(title="Signal Intensity Over Time", xaxis_title="Time Index", yaxis_title="Intensity")
    st.plotly_chart(fig, use_container_width=True)

# --- Main Content ---
if uploaded_ok and uploaded_nok:
    wavelengths_ok, data_ok_raw = load_spectral_csv(uploaded_ok)
    wavelengths_nok, data_nok_raw = load_spectral_csv(uploaded_nok)

    if not np.allclose(wavelengths_ok, wavelengths_nok, rtol=0, atol=1e-5):
        st.warning("Wavelengths are very slightly different between files, likely due to formatting. Proceeding with analysis anyway.")

    # Apply global band filtering
    wavelengths, data_ok = filter_band(wavelengths_ok, data_ok_raw, band_start, band_end)
    _, data_nok = filter_band(wavelengths_nok, data_nok_raw, band_start, band_end)

    st.subheader("Line Overlay Plot of Spectral Data")
    plot_line_overlay_plotly(wavelengths, data_ok, data_nok, num_rows_to_plot)

    st.subheader("3D Combined Line Plot: OK and NOK")
    plot_3d_combined_lines(wavelengths, data_ok, data_nok)

    st.subheader("3D Surface Plot - OK")
    plot_surface(data_ok, wavelengths, "Surface - OK")

    st.subheader("3D Surface Plot - NOK")
    plot_surface(data_nok, wavelengths, "Surface - NOK")

    st.subheader("3D Difference Surface (NOK - OK)")
    plot_difference_surface(data_ok, data_nok, wavelengths)

    if show_mean_spectrum:
        st.subheader("Mean Spectrum Comparison")
        mean_ok = data_ok.mean(axis=0)
        mean_nok = data_nok.mean(axis=0)
        plot_mean_comparison_plotly(wavelengths, mean_ok, mean_nok)

    if show_difference_plot:
        st.subheader("Difference Plot (NOK - OK)")
        plot_difference_plotly(wavelengths, mean_ok, mean_nok)

    if show_heatmap:
        st.subheader("Wavelength-Time Heatmaps")
        plot_heatmap_plotly(data_ok, wavelengths, "OK Welding Heatmap")
        plot_heatmap_plotly(data_nok, wavelengths, "NOK Welding Heatmap")

    if show_time_series:
        st.subheader("Signal Evolution Over Time at Specific Wavelengths")
        unique_wavelengths = list(np.round(wavelengths, 2))
        selected_wavelengths = st.multiselect("Select Wavelengths to Track Over Time", options=unique_wavelengths, default=[unique_wavelengths[0]])
        plot_time_series(data_ok, data_nok, wavelengths, selected_wavelengths)

    if show_classic_spectrogram:
        st.subheader("📊 Classic Spectrogram Style (2D Heatmap)")
    
        combined = pd.concat([data_ok, data_nok], axis=0).reset_index(drop=True)
    
        fig = px.imshow(
            combined.values,
            labels=dict(x="Wavelength Index", y="Sample Index", color="Intensity"),
            x=wavelengths,
            y=np.arange(len(combined)),
            color_continuous_scale="Turbo"
        )
        fig.update_layout(
            title="Combined Spectrogram (OK and NOK)",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Time Index",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload both OK and NOK welding CSV files to begin analysis.")
