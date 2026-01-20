import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, CubicTriInterpolator
import io

st.set_page_config(layout="wide")
st.title("🚀 Grundwassergleichenplan (FG1–FG8)")

# 1. Datenbank (Koordinaten & ROK)
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

# 2. Globale Parameter
Gx, Gy = 382673.258185, 5642838.300039
terrain_elevation = 396.64

st.caption("Eingabe: BOK-Abstich [m]. Wenn 0 → Brunnen wird ignoriert.")

# 3. Eingabemaske
inputs = {}
cols = st.columns(4) # Layout für Inputs optimiert
keys = ['FG1','FG2','FG3','FG4','FG5','FG6','FG7','FG8']
for i, fg in enumerate(keys):
    with cols[i % 4]:
        inputs[fg] = st.number_input(f"{fg} BOK [m]", min_value=0.0, value=0.0, step=0.01, key=fg)

contour_step = st.number_input("Konturintervall [m]", min_value=0.01, value=0.05, step=0.01)

# 4. Berechnung & Plot
if st.button("🚀 Gleichenplan erstellen", use_container_width=True):
    # Nur Brunnen mit Wert > 0 filtern
    data = {k: v for k, v in inputs.items() if v and v > 0}

    if len(data) < 3:
        st.error("⚠️ Bitte mindestens 3 Brunnen mit Werten > 0 eingeben (für Triangulation).")
    else:
        # DataFrame aufbauen
        df = pd.DataFrame(list(data.items()), columns=['name', 'bok_abstich'])

        # Daten aus DB mappen
        df['x'] = df['name'].map(lambda n: brunnen_db[n]['x'])
        df['y'] = df['name'].map(lambda n: brunnen_db[n]['y'])
        df['rok_elev'] = df['name'].map(lambda n: brunnen_db[n]['rok_elev'])

        # Berechnungen
        df['head'] = df['rok_elev'] - df['bok_abstich']
        df['head_rel'] = df['head'] - terrain_elevation
        df['x_local'] = df['x'] - Gx
        df['y_local'] = df['y'] - Gy

        # Tabelle anzeigen
        st.subheader("Datenübersicht")
        st.dataframe(df[['name','bok_abstich','rok_elev','head','head_rel']].round(4), use_container_width=True)

        # ---------------------------------------------------------
        # PLOT START
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        # A) Triangulation & Gradienten
        triang = Triangulation(df['x_local'], df['y_local'])
        
        # Interpolator für Gradientenberechnung (Fließrichtung)
        # geom='min_E' minimiert Energie, oft glatter als Standard
        tci = CubicTriInterpolator(triang, df['head_rel'], kind='min_E') 
        
        # Gradient an den Brunnenpositionen berechnen
        # Fließrichtung ist negativ zum Gradienten (-dh/dx, -dh/dy)
        gx, gy = tci.gradient(df['x_local'], df['y_local'])
        flow_u, flow_v = -gx, -gy

        # B) Konturlinien
        zmin = df['head_rel'].min()
        zmax = df['head_rel'].max()
        # Levels sauber runden
        start = np.floor(zmin / contour_step) * contour_step
        stop  = np.ceil(zmax / contour_step) * contour_step
        levels = np.arange(start, stop + contour_step/2, contour_step)

        cs = ax.tricontour(triang, df['head_rel'], levels=levels, colors='black', linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=11, fmt='%.2f')

        # C) Brunnenpunkte
        ax.plot(df['x_local'], df['y_local'], 'ro', markersize=12, zorder=5, label='Brunnen')

        # D) Labels
        for _, row in df.iterrows():
            ax.annotate(f"{row['name']}\n{row['head_rel']:.3f}",
                        (row['x_local'], row['y_local']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # E) Fließrichtungspfeile (Quiver) an den Brunnen
        # Normierung der Pfeile für gleichmäßige Darstellung (optional)
        # Hier zeigen wir die echte magnitude an
        ax.quiver(df['x_local'], df['y_local'], flow_u, flow_v,
                  color='darkgreen', width=0.005, scale=None, scale_units='xy', angles='xy',
                  label='Fließrichtung (lokal)')

        ax.set_xlabel('x_local [m]')
        ax.set_ylabel('y_local [m]')
        ax.set_title(f'Grundwassergleichenplan & Fließrichtung (n={len(df)})')
        ax.grid(True, alpha=0.3)
        ax.axis('equal') # Wichtig für korrekte Geometrie/Winkel
        ax.legend(loc='upper right')

        st.pyplot(fig)

        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button("💾 Plot speichern", buf.getvalue(), "isohypsen_flow.png", "image/png")

st.caption("FG1–FG8 | Update automatisch via GitHub")
