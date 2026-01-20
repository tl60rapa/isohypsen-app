import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("🚀 Grundwassergleichenplan (FG1–FG8)")

# DB (deine Werte)
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

# Einstellungen
Gx, Gy = 382673.258185, 5642838.300039
terrain_elevation = 396.64

st.caption("Eingabe: BOK-Abstich [m]. Wenn 0 → Brunnen wird ignoriert.")

# Inputs in Reihenfolge FG1..FG8
inputs = {}
for fg in ['FG1','FG2','FG3','FG4','FG5','FG6','FG7','FG8']:
    inputs[fg] = st.number_input(f"{fg} BOK-Abstich [m]", min_value=0.0, value=0.0, step=0.01, key=fg)

# Kontur-Intervall (optional, aber praktisch)
contour_step = st.number_input("Konturintervall [m]", min_value=0.01, value=0.05, step=0.01)

if st.button("🚀 Gleichenplan erstellen", use_container_width=True):
    # Nur Brunnen mit Wert > 0 verwenden
    data = {k: v for k, v in inputs.items() if v and v > 0}

    if len(data) < 3:
        st.error("Für einen Gleichenplan mit Triangulation brauchst du in der Praxis mind. 3 Brunnen (nicht auf einer Linie).")
    else:
        df = pd.DataFrame(list(data.items()), columns=['name', 'bok_abstich'])

        # DB join
        df['x'] = df['name'].map(lambda n: brunnen_db[n]['x'])
        df['y'] = df['name'].map(lambda n: brunnen_db[n]['y'])
        df['rok_elev'] = df['name'].map(lambda n: brunnen_db[n]['rok_elev'])

        # Head-Berechnung
        df['head'] = df['rok_elev'] - df['bok_abstich']         # absolute GW-Höhe [m]
        df['head_rel'] = df['head'] - terrain_elevation          # relativ zu Gelände [m]
        from matplotlib.tri import Triangulation, CubicTriInterpolator

# Triangulation erstellen
triang = Triangulation(df['x_local'], df['y_local'])

# Interpolator für Gradient
tci = CubicTriInterpolator(triang, df['head_rel'])

# Gradient an Brunnenpunkten (dx, dy)
dx, dy = tci.gradient(df['x_local'], df['y_local'])

# Plot
fig, ax = plt.subplots(figsize=(12,10))

# Konturen
levels = np.arange(df['head_rel'].min()-0.1, df['head_rel'].max()+0.1, 0.05)
cs = ax.tricontour(triang, df['head_rel'], levels=levels, colors='black', linewidths=1.5)
ax.clabel(cs, inline=True, fontsize=11, fmt='%.2f')

# Brunnen + Pfeile (Fließrichtung = -Gradient)
norm = np.hypot(dx, dy) + 1e

        # lokale Koordinaten
        df['x_local'] = df['x'] - Gx
        df['y_local'] = df['y'] - Gy

        st.subheader("Daten (verwendet)")
        st.dataframe(df[['name','bok_abstich','rok_elev','head','head_rel']].round(4), use_container_width=True)

        # Levels definieren (auf sauberen Raster von contour_step "snappen")
        zmin = float(df['head_rel'].min())
        zmax = float(df['head_rel'].max())
        start = np.floor(zmin / contour_step) * contour_step
        stop  = np.ceil(zmax / contour_step) * contour_step
        levels = np.arange(start, stop + contour_step/2, contour_step)

        # Plot: tricontour = TIN-Konturen direkt aus Punkten
        fig, ax = plt.subplots(figsize=(12, 10))
        cs = ax.tricontour(df['x_local'], df['y_local'], df['head_rel'],
                           levels=levels, colors='black', linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=11, fmt='%.2f')

        ax.plot(df['x_local'], df['y_local'], 'ro', markersize=10, zorder=5)

        for _, row in df.iterrows():
            ax.annotate(f"{row['name']}\n{row['head_rel']:.3f}",
                        (row['x_local'], row['y_local']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xlabel('x_local [m]')
        ax.set_ylabel('y_local [m]')
        ax.set_title(f'Grundwassergleichenplan (head_rel), n={len(df)}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        st.pyplot(fig)

        # Download PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        st.download_button("💾 PNG herunterladen", buf.getvalue(), "gleichenplan.png", "image/png")

st.caption("FG1–FG8 | 0 = Brunnen ignorieren | Updates nach GitHub-Commit automatisch")

