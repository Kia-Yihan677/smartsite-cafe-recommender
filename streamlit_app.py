import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import shap
from shapely.geometry import Point
import streamlit as st

CRS_METRIC = 32748
CRS_LATLON = 4326

FEATURES = [
    "competitor_500m", "restaurant_500m", "fastfood_500m",
    "distance_to_university", "distance_to_mall", "distance_to_primary_road",
]

LABEL_ID = {
    "competitor_500m": "Coffee shop lain (500 m)",
    "restaurant_500m": "Tempat makan di sekitar (500 m)",
    "fastfood_500m": "Gerai makan cepat saji (500 m)",
    "distance_to_university": "Jarak ke kampus",
    "distance_to_mall": "Jarak ke mall",
    "distance_to_primary_road": "Jarak ke jalan utama",
}

UNIT = {
    "competitor_500m": "tempat",
    "restaurant_500m": "tempat",
    "fastfood_500m": "tempat",
    "distance_to_university": "m",
    "distance_to_mall": "m",
    "distance_to_primary_road": "m",
}

# Interpretasi kualitatif per fitur, tergantung arah pengaruhnya (naik/turun skor).
INTERPRETASI = {
    "competitor_500m": {
        "up": "Jumlah pesaing sehat, pasar sudah terbukti.",
        "down": "Minim pesaing di sekitar, pasar belum teruji.",
    },
    "restaurant_500m": {
        "up": "Area sudah jadi tujuan makan, trafik pengunjung tinggi.",
        "down": "Belum jadi kawasan kuliner, trafik pengunjung rendah.",
    },
    "fastfood_500m": {
        "up": "Ekosistem makan cepat saji ikut menambah keramaian.",
        "down": "Minim gerai makan cepat saji di sekitar.",
    },
    "distance_to_university": {
        "up": "Dekat sumber pelanggan mahasiswa.",
        "down": "Jauh dari sumber pelanggan mahasiswa.",
    },
    "distance_to_mall": {
        "up": "Kebagian arus pengunjung mall.",
        "down": "Jauh dari arus pengunjung mall.",
    },
    "distance_to_primary_road": {
        "up": "Mudah terlihat dan dilewati kendaraan.",
        "down": "Kurang terlihat dari jalur utama.",
    },
}


@st.cache_resource
def load_model():
    model = joblib.load("smartsite_model.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer


def _to_points(gdf):
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{CRS_METRIC}")
    g = gdf.copy()
    g["geometry"] = g.geometry.centroid
    return g.set_crs(CRS_LATLON, allow_override=True).to_crs(epsg=CRS_METRIC)


def _count_within(point, poi_gdf, radius):
    if len(poi_gdf) == 0:
        return 0
    return int(poi_gdf.geometry.distance(point).le(radius).sum())


def _dist_to_nearest(point, poi_gdf, default=5000.0):
    if len(poi_gdf) == 0:
        return default
    return float(poi_gdf.geometry.distance(point).min())


def _fetch(tags, lat, lon, dist):
    try:
        return ox.features_from_point((lat, lon), tags=tags, dist=dist)
    except Exception:
        return None


def predict_location(lat, lon, model, explainer):
    if not (-7.1 < lat < -6.7 and 107.4 < lon < 107.9):
        return None, ("Koordinat di luar jangkauan Bandung. Gunakan lat sekitar "
                       "-6.7..-7.1 dan lon sekitar 107.4..107.9.")

    R = 800
    cafes = _to_points(_fetch({"amenity": "cafe", "shop": "coffee"}, lat, lon, R))
    rests = _to_points(_fetch({"amenity": "restaurant"}, lat, lon, R))
    ffood = _to_points(_fetch({"amenity": "fast_food"}, lat, lon, R))
    unis = _to_points(_fetch({"amenity": "university"}, lat, lon, 5000))
    malls = _to_points(_fetch({"shop": "mall"}, lat, lon, 5000))

    pt = gpd.GeoSeries([Point(lon, lat)], crs=CRS_LATLON).to_crs(epsg=CRS_METRIC).iloc[0]

    try:
        roads = ox.graph_to_gdfs(
            ox.graph_from_point((lat, lon), dist=1500, network_type="drive"),
            nodes=False, edges=True,
        )
        primary = roads[roads["highway"].apply(
            lambda h: any(t in (h if isinstance(h, list) else [h]) for t in ["primary", "trunk"])
        )].to_crs(epsg=CRS_METRIC)
        dist_road = float(primary.geometry.distance(pt).min()) if len(primary) else 2000.0
    except Exception:
        dist_road = 2000.0

    feats = {
        "competitor_500m": _count_within(pt, cafes, 500),
        "restaurant_500m": _count_within(pt, rests, 500),
        "fastfood_500m": _count_within(pt, ffood, 500),
        "distance_to_university": _dist_to_nearest(pt, unis),
        "distance_to_mall": _dist_to_nearest(pt, malls),
        "distance_to_primary_road": dist_road,
    }
    X = pd.DataFrame([feats])[FEATURES]

    score = float(model.predict_proba(X)[:, 1][0])

    sv = explainer.shap_values(X)
    sv = sv[:, :, 1] if np.array(sv).ndim == 3 else sv
    kontrib = pd.Series(sv[0], index=FEATURES).sort_values(key=abs, ascending=False)

    return {"score": score, "feats": feats, "kontrib": kontrib}, None


# ---------------- UI ----------------

st.set_page_config(page_title="SmartSite", page_icon="☕")
st.title("☕ SmartSite — Analisis Lokasi Coffee Shop (Bandung)")
st.markdown(
    "Analisis potensi lokasi coffee shop berdasarkan ketersediaan fasilitas, "
    "aksesibilitas, dan kompetisi di sekitarnya — dihitung langsung dari data "
    "OpenStreetMap.\n\n"
    "*Contoh: Dago (-6.8915, 107.6107), Alun-alun (-6.9218, 107.6069).*"
)

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=-6.8915, format="%.4f")
with col2:
    lon = st.number_input("Longitude", value=107.6107, format="%.4f")

if st.button("Cek analisis lokasi", type="primary"):
    with st.spinner("Menarik data OpenStreetMap & menghitung skor (beberapa detik)..."):
        model, explainer = load_model()
        result, error = predict_location(lat, lon, model, explainer)

    if error:
        st.error(error)
    else:
        score = result["score"]
        feats = result["feats"]
        kontrib = result["kontrib"]
        skor_100 = round(score * 100)

        if score > 0.7:
            kategori = "Sangat cocok"
            deskripsi = "Lokasi ini punya trafik dan visibilitas yang kuat."
            warna = "#3ecf8e"
        elif score > 0.5:
            kategori = "Cukup cocok"
            deskripsi = "Lokasi ini punya potensi, tapi belum optimal."
            warna = "#f5a623"
        else:
            kategori = "Kurang cocok"
            deskripsi = "Lokasi ini kurang didukung trafik dan visibilitas sekitar."
            warna = "#e05252"

        st.markdown(
            f"""
            <div style="margin-top:8px;">
              <div style="font-size:13px; color:#888; letter-spacing:0.02em;">Skor kecocokan</div>
              <div style="line-height:1.1; margin-top:2px;">
                <span style="font-size:48px; font-weight:700; color:{warna};">{skor_100}</span>
                <span style="font-size:20px; color:#888; font-weight:400;"> / 100</span>
              </div>
              <div style="font-size:16px; font-weight:600; margin-top:10px;">{kategori}</div>
              <div style="font-size:14px; color:#888;">{deskripsi}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='margin-top:28px; margin-bottom:4px; font-size:12px; "
            "letter-spacing:0.08em; color:#888; font-weight:600;'>YANG MEMENGARUHI SKOR</div>",
            unsafe_allow_html=True,
        )

        for f, v in kontrib.items():
            arah = "up" if v > 0 else "down"
            panah = "&uarr;" if v > 0 else "&darr;"
            warna_panah = "#3ecf8e" if v > 0 else "#e05252"
            nilai = f"{feats[f]:.0f} {UNIT[f]}"
            st.markdown(
                f"""
                <div style="padding:14px 0; border-top:1px solid rgba(128,128,128,0.25);">
                  <div style="display:flex; gap:10px; align-items:flex-start;">
                    <div style="color:{warna_panah}; font-size:16px; line-height:1.4;">{panah}</div>
                    <div>
                      <div style="font-weight:600; font-size:15px;">{LABEL_ID[f]}: {nilai}</div>
                      <div style="font-size:13px; color:#888;">{INTERPRETASI[f][arah]}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Lihat data mentah"):
            tabel = pd.DataFrame({
                "Indikator": [LABEL_ID[f] for f in FEATURES],
                "Nilai": [f"{feats[f]:.1f} {UNIT[f]}" for f in FEATURES],
            })
            st.dataframe(tabel, hide_index=True, use_container_width=True)
