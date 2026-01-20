import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(layout="wide")
st.title("🚀 Isohypsen Generator FG1-8")

# DEINE DB
brunnen_db = {
    'FG1': {'x': 382673.38885, 'y': 5642833.20171, 'rok_elev': 397.692},
    'FG2': {'x': 382680.94665, 'y': 5642838.43778, 'rok_elev': 397.122},
    'FG3': {'x': 382692.407626, 'y': 5642844.986956, 'rok_elev': 396.881},
    'FG4': {'x': 382675.633003, 'y': 5642838.224092, 'rok_elev': 397.382},
    'FG5': {'x': 382674.85057, 'y': 5642835.96159, 'rok_elev': 397.54},
    'FG6': {'x': 382670.90617, 'y': 5642838.16415, 'rok_elev': 397.413},
    'FG7': {'x': 382673.258185, 'y': 5642838.300039, 'rok_elev': 396.604},
    'FG8': {'x': 382678.273087, 'y': 5642838.375616, 'rok_elev': 396.524}
}

# FG1-FG8 Inputs (Reihenfolge!)
inputs = {}
for fg in ['FG1','FG2','FG3','FG4','FG5','FG6','FG7','FG8']:
    val = st.number_input(f"{fg} BOK [m]", min_value=0.0, value=7.5, step=0.01, key=fg)
    inputs[fg] = val if val > 0 else None  # Skip wenn 0

if st.button("🚀 Plot (nur gefüllte Brunnen)"):
    # Nur Brunnen mit Werten >0
    data = {k:v for k,v in inputs.items() if v is not None}
    if len(data) < 2:
        st.error("Mind. 2 Brunnen eingeben!")
    else:
        df = pd.DataFrame(list(data.items()), columns=['name','bok_abstich'])
        df['x'] = df['name'].map(lambda n: brunnen_db[n]['x'])
        df['y'] = df['name'].map(lambda n: brunnen_db[n]['y'])
        df['rok_elev'] = df['name'].map(lambda n: brunnen_db[n]['rok_elev'])
        df['head'] = df['rok_elev'] - df['bok_abstich']
        
        Gx, Gy = 382673.258185, 5642838.300039
        terrain_elevation = 396.64
        df['x_local'] = df['x'] - Gx
        df['y_local'] = df['y'] - Gy
        df['head_rel'] = df['head'] - terrain_elevation
        
        X = np.column_stack([df['x_local'], df['y_local'], np.ones(len(df))])
        coeffs_rel = np.linalg.lstsq(X, df['head_rel'], rcond=None)[0]
        a_rel, b_rel, c_rel = coeffs_rel
        grad = np.sqrt(a_rel**2 + b_rel**2)
        
        col1, col2 = st.columns(2)
        col1.metric("Gradient i_gesamt", f"{grad:.6f}")
        col2.metric("Anz. Brunnen", len(df))
        
        st.dataframe(df.round(4))
        
        # Plot nur gefüllte Brunnen
        fig, ax = plt.subplots(figsize=(12,10))
        x_min, x_max = df['x_local'].min()-2, df['x_local'].max()+2
        y_min, y_max = df['y_local'].min()-2, df['y_local'].max()+2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        zz = a_rel * xx + b_rel * yy + c_rel
        contour = ax.contour(xx, yy, zz, levels=12, colors='black', linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=11, fmt='%.3f')
        ax.plot(df['x_local'], df['y_local'], 'ro', markersize=12, zorder=5)
        for _, row in df.iterrows():
            ax.annotate(row['name'], (row['x_local'], row['y_local']), xytext=(5,5),
                       textcoords='offset points', fontsize=14, fontweight='bold')
        ax.quiver(0,0, -a_rel, -b_rel, 20, color='darkgreen', width=0.01, scale=1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Isohypsen head_rel (n={len(df)} Brunnen)')
        st.pyplot(fig)
        
        # Download
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button("💾 PNG", buf.getvalue(), "isohypsen.png", "image/png")

st.caption("FG1-FG8 | Leere = skip | Auto-Update nach GitHub Push")
