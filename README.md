# SmartSite — Rekomendasi Lokasi Coffee Shop Berbasis Machine Learning Spasial

SmartSite memprediksi **lokasi optimal untuk membuka coffee shop baru di Bandung** menggunakan data spasial publik (OpenStreetMap) dan model Machine Learning (Random Forest), lengkap dengan penjelasan *interpretable* (SHAP) untuk setiap rekomendasi.

> Masalah: keputusan buka cabang baru itu krusial, tapi banyak UMKM & bisnis menengah masih memilih lokasi berdasarkan *feeling* atau keramaian sesaat. SmartSite mengganti tebakan itu dengan analisis data yang cepat, murah, dan bisa dijelaskan.

---

## Demo

- **Live demo (Streamlit Community Cloud):** https://smartsite-cafe-recommender-mwpbsjqsv4jklncn8v9dwa.streamlit.app/
- **Peta interaktif (HTML):** `smartsite_peta.html`

## Hasil Utama

| Metrik | Nilai |
|---|---|
| AUC-ROC (random split, test set) | **0.9675** |
| AUC-ROC (spatial-aware CV, held-out wilayah) | **0.8538** ± 0.0518 |
| Selisih random vs spatial (indikasi spatial leakage) | 0.1137 |
| Best params (GridSearchCV) | `max_depth=12, min_samples_leaf=3, n_estimators=400` |
| CV AUC (GridSearchCV) | 0.9425 |

**Catatan penting:** angka 0.9675 didapat dari random train-test split dan cenderung optimis karena *spatial autocorrelation* (lihat notebook bagian 10d). Angka **0.8538** dari spatial-aware cross-validation lebih merepresentasikan performa realistis model di lokasi yang benar-benar belum pernah dilihat — inilah yang seharusnya dijadikan acuan utama, bukan 0.9675.

**Perbandingan model (5-fold CV AUC):** Random Forest terbukti unggul dibanding XGBoost dan Logistic Regression lewat perbandingan langsung, bukan asumsi (lihat notebook bagian 10b).

Fitur terpenting: **kepadatan kompetitor 500m** dan **kepadatan restoran 500m** — cafe cenderung tumbuh di ekosistem komersial yang sudah ramai, bukan di lokasi terisolasi.

> Catatan reproducibility: karena data ditarik live dari OpenStreetMap, angka-angka di atas bisa sedikit berbeda tiap kali notebook dijalankan ulang (OSM terus diedit komunitas). Angka di tabel ini adalah snapshot dari run terakhir yang sudah diverifikasi bersih (execution count urut, tanpa sesi tercampur).

---

## Cara Kerja (Pipeline)

1. **Ambil data OSM** — coffee shop, restoran, fast food, universitas, mall, jaringan jalan di Bandung (via OSMnx/Overpass).
2. **Standardisasi CRS** — semua layer diproyeksikan ke EPSG:32748 (UTM 48S) agar perhitungan jarak dalam **meter**.
3. **Feature engineering spasial** — 6 fitur: `competitor_500m`, `restaurant_500m`, `fastfood_500m`, `distance_to_university`, `distance_to_mall`, `distance_to_primary_road`.
4. **Sampling** — 478 sampel positif (cafe existing) + 500 sampel negatif (*pseudo-absence*, titik acak ≥150 m dari cafe).
5. **Training & evaluasi** — Random Forest, dibandingkan dengan Logistic Regression & XGBoost; validasi via stratified train/test + 5-fold CV.
6. **Explainability** — SHAP untuk global importance dan alasan per-rekomendasi.
7. **Prediksi wilayah** — grid 300m di seluruh Bandung → skor kecocokan → Top-10 dengan filter jarak antar-rekomendasi ≥1000 m.

## Struktur Repo

```
.
├── SmartSite_Final.ipynb        # notebook utama (end-to-end, Restart & Run All clean)
├── streamlit_app.py             # Streamlit demo (deploy ke Streamlit Community Cloud)
├── smartsite_model.pkl          # model Random Forest terlatih
├── smartsite_dataset.csv        # dataset fitur + label
├── smartsite_top_lokasi.csv     # Top-10 rekomendasi
├── smartsite_peta.html          # peta interaktif
├── requirements.txt
└── README.md
```

## Menjalankan

```bash
pip install -r requirements.txt
# Notebook: buka SmartSite_Final.ipynb → Restart & Run All
# Demo lokal:
streamlit run streamlit_app.py
```

## Limitasi

- *Pseudo-absence* = titik acak, bukan bukti "lokasi buruk" secara ekonomi.
- Hanya menangkap cafe yang terdaftar di OSM (cafe informal bisa terlewat).
- Belum ada fitur ekonomi (harga sewa, daya beli) maupun validasi ke omzet nyata.

## Rencana Pengembangan

- Integrasi data BPS (kepadatan penduduk, pengeluaran per kapita).
- Proxy foot-traffic dari kelas jalan & transit.
- Backtesting rekomendasi ke cafe existing yang ramai.
- Generalisasi ke kota lain (parameter `PLACE_NAME` sudah mendukung).

## Sumber Data & Lisensi

OpenStreetMap © kontributor OSM, lisensi ODbL.

