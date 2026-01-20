# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:16:29 2026

@author: Tobias Lotter
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(layout="wide", page_title="Isohypsen Feld-App")

st.title("🚀 Brunnen-Isohypsen Generator")

# DEINE GENAU DB (hardcoded für Cloud)
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

# Touch-Inputs (FG2-7 wie Script)
col1, col2 = st.columns(2)
with col1:
    fg2 = st.number_input("FG2 BOK [m]", value=7.63, step=0.01)
    fg4 = st.number_input("FG4 BOK [m]", value=7.15, step=0.01)
    fg6 = st.number_input("FG6 BOK [m]", value=7.96, step=0.01)
with col2:
    fg3 = st.number_input("FG3 BOK [m]", value=7.31, step=0.01)
    fg5 = st.number_input("FG5 BOK [m]", value=8.05, step=0.01)
    fg7 = st.number_input("FG7 BOK [m]", value=7.15, step=0.01)

if st.button("🚀 Plot generieren", use_container_width=True):
    bok_data = {'FG2':fg2,'FG3':fg3,'FG4':fg4,'FG5':fg5,'FG6':fg6,'FG7':fg7}
    df = pd.DataFrame(list(bok_data.items()), columns=['name', 'bok_abstich'])
    df['x'] = df['name'].map(lambda n: brunnen_db[n]['x'])
    df['y'] = df['name'].map(lambda n: brunnen_db[n]['y'])
    df['rok_elev'] = df['name'].map(lambda n: brunnen_db[n]['rok_elev'])
    df['head'] = df['rok_elev'] - df['bok_abstich']  # Deine Formel!
    
    Gx, Gy = 382673.258185, 5642838.300039
    terrain_elevation = 396.64
    df['x_local'] = df['x'] - Gx
    df['y_local'] = df['y'] - Gy
    df['head_rel'] = df['head'] - terrain_elevation
    
    X = np.column_stack([df['x_local'], df['y_local'], np.ones(df.shape[0])])
    coeffs_rel = np.linalg.lstsq(X, df['head_rel'], rcond=None)[0]
    a_rel, b_rel, c_rel = coeffs_rel
    
    col1, col2 = st.columns(2)
    col1.metric("Gradient i_gesamt", f"{np.sqrt(a_rel**2 + b_rel**2):.6f}")
    col2.metric("FEFLOW head_rel", f"{a_rel:.6f}x + {b_rel:.6f}y + {c_rel:.3f}")
    
    st.subheader("Datenübersicht")
    st.dataframe(df[['name','bok_abstich','rok_elev','head','head_rel','x_local','y_local']].round(4))
    
    # Plot (exakt dein Script)
    x_min, x_max = df['x_local'].min() - 2, df['x_local'].max() + 2
    y_min, y_max = df['y_local'].min() - 2, df['y_local'].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    zz_rel = a_rel * xx + b_rel * yy + c_rel
    
    fig, ax = plt.subplots(figsize=(12, 10))
    contour_rel = ax.contour(xx, yy, zz_rel, levels=12, colors='black', linewidths=1.5)
    ax.clabel(contour_rel, inline=True, fontsize=11, fmt='%.3f')
    ax.plot(df['x_local'], df['y_local'], 'ro', markersize=10, zorder=5)
    for i, row in df.iterrows():
        ax.annotate(row['name'], (row['x_local'], row['y_local']), xytext=(4, 4),
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    flow_dir_x, flow_dir_y = -a_rel, -b_rel
    ax.quiver([0, 8], [0, 2], [flow_dir_x]*2, [flow_dir_y]*2, 15, color='darkgreen',
              width=0.006, scale=1, label=f'Fließrichtung i={np.sqrt(a_rel**2 + b_rel**2):.4f}')
    ax.set_xlabel('x_local [m]'); ax.set_ylabel('y_local [m]')
    ax.set_title('Isohypsen head_rel (schwarz)')
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)
    
    # Download
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    st.download_button("💾 PNG speichern", buf.getvalue(), "isohypsen.png", "image/png")

# Footer
st.markdown("---")
st.caption("Android PWA ready – Erstellt mit Streamlit")
