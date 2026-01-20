import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import io

# Seiten-Config
st.set_page_config(layout="wide", page_title="Isohypsen Generator")
st.title("🚀 Grundwassergleichenplan (FG1–FG8)")

# ------------------------------------------------------------------
# 1. DATENBANK (Koordinaten & ROK)
# ------------------------------------------------------------------
brunnen_db = {
    'FG1': {'x': 382673.38885,  'y': 5642833.20171,  'rok_elev': 397.692},
    'FG2': {'x': 382680.94665,  'y': 5642838.43778,  'rok_elev': 397.122},
    'FG3': {'x': 382692.407626, 'y': 5642844.986956, 'rok_elev': 396.881},
    'FG4': {'x': 382675.633003, 'y': 5642838.224092, 'rok_elev': 397.382},
    'FG5': {'x': 382674.85057,  'y': 5642835.96159,  'rok_elev': 397.54},
    'FG6': {'x': 382670.90617,  'y': 5642838.16415,  'rok_elev': 397.413},
    'FG7': {'x': 382673.258185, 'y': 5642838.300039, 'rok_elev': 396.604},
    'FG8': {'x': 382678.273087, 'y': 5642838.375616, 'rok_elev': 396.524},
}

# Globale Parameter
Gx, Gy = 382673.258185, 5642838.300039
terrain_elevation = 396.64

st.caption("Eingabe: BOK-Abstich [m]. 0 = Brunnen ignorieren.")

# ------------------------------------------------------------------
# 2. INPUTS
# ------------------------------------------------------------------
inputs = {}
cols = st.columns(4)
keys = ['FG1','FG2','FG3','FG4','FG5','FG6','FG7','FG8']

for i, fg in enumerate(keys):
    with cols[i % 4]:
        # Default 0.0 -> Wird ignoriert
        inputs[fg] = st.number_input(f"{fg}", min_value=0.0, value=0.0, step=0.01, key=fg)

# Optional: Manueller Kontur-Step (falls leer -> Auto)
contour_step_manual = st.number_input("Konturintervall [m] (0 = Auto)", min_value=0.0, value=0.0, step=0.01)

# ------------------------------------------------------------------
# 3. BERECHNUNG & PLOT
# ------------------------------------------------------------------
if st.button("🚀 Gleichenplan erstellen", use_container_width=True):
    # Nur Brunnen mit Wert > 0 filtern
    data = {k: v for k, v in inputs.items() if v and v > 0}

    if len(data) < 3:
        st.error("⚠️ Bitte mindestens 3 Brunnen mit Werten > 0 eingeben.")
    else:
        # DataFrame
        df = pd.DataFrame(list(data.items()), columns=['name', 'bok_abstich'])

        # Mapping aus DB
        df['x'] = df['name'].map(lambda n: brunnen_db[n]['x'])
        df['y'] = df['name'].map(lambda n: brunnen_db[n]['y'])
        df['rok_elev'] = df['name'].map(lambda n: brunnen_db[n]['rok_elev'])

        # Head & Local Coords
        df['head'] = df['rok_elev'] - df['bok_abstich']
        df['head_rel'] = df['head'] - terrain_elevation
        df['x_local'] = df['x'] - Gx
        df['y_local'] = df['y'] - Gy

        # Anzeige Tabelle
        st.subheader("Datenübersicht")
        st.dataframe(df[['name','bok_abstich','rok_elev','head','head_rel']].round(4), use_container_width=True)

        # ---------------------------------------------------------
        # RBF INTERPOLATION (Extrapolation möglich!)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        # Koordinaten für RBF (Training)
        coords = np.column_stack([df['x_local'], df['y_local']])
        values = df['head_rel'].values

        # RBF Interpolator erstellen (Thin Plate Spline = robust & glatt)
        rbf = RBFInterpolator(coords, values, kernel='thin_plate_spline')

        # Grid für Konturen (mit Extrapolation-Padding)
        padding = 5.0 # Meter Rand um Brunnen
        x_min, x_max = df['x_local'].min() - padding, df['x_local'].max() + padding
        y_min, y_max = df['y_local'].min() - padding, df['y_local'].max() + padding
        
        # Feinheit des Grids
        grid_res = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), 
                             np.linspace(y_min, y_max, grid_res))
        
        # Vorhersage auf Grid
        coords_grid = np.column_stack([xx.ravel(), yy.ravel()])
        zz = rbf(coords_grid).reshape(xx.shape)

        # ---------------------------------------------------------
        # LEVELS AUTOMATIK
        # ---------------------------------------------------------
        zmin, zmax = zz.min(), zz.max() # Min/Max im Grid (inkl Extrapolation)
        range_z = zmax - zmin
        
        if contour_step_manual > 0:
            step = contour_step_manual
        else:
            # Auto-Step: Wenn Range klein (<0.2m), nimm 0.01m, sonst gröber
            if range_z < 0.2:
                step = 0.01
            elif range_z < 0.5:
                step = 0.02
            else:
                step = 0.05
        
        # Levels exakt runden
        start = np.floor(zmin / step) * step
        stop  = np.ceil(zmax / step) * step
        levels = np.arange(start, stop + step/2, step)

        # ---------------------------------------------------------
        # PLOTTING
        # ---------------------------------------------------------
        # Konturen
        cs = ax.contour(xx, yy, zz, levels=levels, colors='black', linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=11, fmt='%.2f')

        # Brunnenpunkte
        ax.plot(df['x_local'], df['y_local'], 'ro', markersize=12, zorder=5, label='Brunnen')

        # Labels
        for _, row in df.iterrows():
            ax.annotate(f"{row['name']}\n{row['head_rel']:.3f}", 
                        (row['x_local'], row['y_local']), xytext=(5,5),
                        textcoords='offset points', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_title(f'Gleichenplan (RBF Extrapolation, n={len(df)})')
        ax.set_xlabel('x_local [m]')
        ax.set_ylabel('y_local [m]')
        ax.grid(True, alpha=0.3)
        ax.axis('equal') # Wichtig!
        
        st.pyplot(fig)

        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        st.download_button("💾 PNG speichern", buf.getvalue(), "isohypsen_rbf.png", "image/png")

st.caption("FG1–FG8 | Update via GitHub | RBF Interpolation")
