# Sistem Rekomendasi Varietas Cabai

Sistem rekomendasi untuk menentukan jenis varietas tanaman cabai berdasarkan kondisi lingkungan menggunakan algoritma K-Nearest Neighbors.

## Fitur

- Rekomendasi varietas cabai berdasarkan:
  - Ketinggian tempat
  - Suhu
  - Kebutuhan air
  - Kebutuhan cahaya
  - Kondisi tanah
- Perhitungan similarity score untuk setiap rekomendasi
- Interface web menggunakan Flask

## Instalasi

1. Clone repository

```bash
git clone https://github.com/rezakhaebar/rekomendasi-varietas-cabai.git
cd sistem-rekomendasi-cabai
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi

```bash
python app.py
```

4. Buka browser dan akses `http://localhost:5000`

## Penggunaan

1. Masukkan parameter lingkungan:
   - Ketinggian tempat (mdpl)
   - Suhu (°C)
   - Kebutuhan air
   - Kebutuhan cahaya
   - Kondisi tanah
2. Klik tombol "Submit"
3. Sistem akan menampilkan rekomendasi varietas cabai yang cocok beserta similarity score

## Dataset

Dataset berisi informasi varietas cabai dengan atribut:

- Nama varietas
- Ketinggian tempat
- Suhu
- Kebutuhan air
- Kebutuhan cahaya
- Kondisi tanah

## Teknologi yang Digunakan

- Python
- Flask
- scikit-learn
- pandas
- numpy
- HTML/CSS/JavaScript
