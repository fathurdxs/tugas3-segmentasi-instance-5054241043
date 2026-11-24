# Tugas 3 — Segmentasi Instance dengan Deep Learning (Evaluasi CPMK 3)
Muhammad Fathur Aziz — 5054241043

## Struktur repo

```
Tugas-3/
├── README.md
├── notebook/
│   └── segmentasi_instance_5054241043.ipynb   notebook utama (Colab): training 3 model, evaluasi 2 test set,
│                                              variabel NRP, analisis kegagalan, eksperimen, demo
├── scripts/
│   └── siapkan_foto.py                        konversi foto HP (HEIC) -> JPG, tabel EXIF, saran kondisi sulit
├── data/
│   └── kondisi_foto.csv                       kondisi tiap foto test sendiri (normal/redup/backlight/oklusi/sudut_ekstrem)
├── bukti/
│   ├── metadata_exif.csv                      EXIF tiap foto (tanpa koordinat GPS)
│   └── screenshot_anotasi/                    screenshot proses anotasi di Roboflow
└── docs/
    ├── Logbook.md                             logbook mingguan + hipotesis bertanggal
    └── Penggunaan_AI.md                       deklarasi penggunaan AI + kasus saran AI yang salah
```

Tidak di-commit (lihat `.gitignore`): `foto/` (foto asli HEIC berisi GPS dan hasil konversinya; test set tersimpan di project Roboflow privat), `materi/` (PDF materi + ketentuan tugas, hal. 29–32), serta bobot model dan hasil training (tersimpan di Google Drive `MyDrive/Tugas3_Segmentasi/`).

## Desain eksperimen

| Ketentuan PDF | Pemenuhannya |
|---|---|
| ≥3 model, ≥2 keluarga arsitektur | YOLO11n-seg + YOLOv8n-seg (**one-stage**, mask prototype 160×160) vs Mask R-CNN R50-FPN v2 (**two-stage**, mask RoI 28×28) |
| Data latih Panoptic Indoor Segmentation v5 | diunduh via Roboflow API (isi slug di sel konfigurasi) |
| Variabel NRP, digit terakhir **3** | **jumlah data latih 25% / 50% / 100%** (subset acak bersarang, seed 0), diterapkan pada **ketiga model** (YOLO11n-seg, YOLOv8n-seg, Mask R-CNN) → `bukti/tabel_variabel_nrp.csv` + grafik |
| Hipotesis bertanggal sebelum eksperimen | entri di `docs/Logbook.md` (lihat catatan kronologi di sana) |
| Riwayat commit Git | repositori Git bertahap (bukan satu commit besar di akhir) |
| Demo & tanya jawab | bagian **8. Demo** di notebook: inferensi foto baru, ubah `CONF_DEMO`, `ap_kelas('chair')` |
| Performa di test asli vs test sendiri + domain shift | tabel utama, AP per kelas, AP per kondisi (normal/redup/backlight/oklusi/sudut) |
| 5 foto gagal + mekanisme + eksperimen kecil | diagnosis otomatis per objek (tidak terdeteksi / salah kelas / terpecah / mask bocor) + visualisasi GT vs 3 model + eksperimen H1 (resolusi mask) & H2 (koreksi cahaya) |
| Log training, waktu GPU, logbook | `results.csv` tiap run, `bukti/gpu_time_log.csv` (otomatis), `docs/Logbook.md` |
| Batas 2 jam GPU per model | callback penghenti otomatis |

Semua model dievaluasi dengan **COCOeval segm yang sama** (bukan metrik bawaan masing-masing framework) supaya perbandingannya adil. Metrik bawaan Ultralytics tetap ditampilkan sebagai cek silang.

## Langkah pengerjaan

### 1. Ambil foto (hari 1–2)
- Minimal **40 foto ruangan nyata** (kamar kos, rumah, lab, ruang kelas ITS), sebaiknya ±50 sebagai cadangan.
- Minimal **10 foto kondisi sulit**: cahaya redup (lampu mati, malam), backlight (membelakangi jendela), objek saling menutupi, sudut ekstrem (dari lantai/atas).
- Ambil langsung dengan kamera HP. Jangan pakai screenshot atau foto WhatsApp karena EXIF-nya hilang.
- Sebelum mengambil foto, cek daftar kelas dataset di Roboflow, lalu pastikan ruangan yang difoto berisi objek dari kelas-kelas itu.

### 2. Konversi & EXIF
```powershell
pip install pillow pillow-heif opencv-python numpy
```
```powershell
python scripts/siapkan_foto.py foto/asli_heic
```
Hasilnya: `foto/jpg/ruang_XX.jpg` (siap unggah, GPS dibuang), `bukti/metadata_exif.csv`, dan `data/kondisi_foto.csv`.
**Buka `data/kondisi_foto.csv` lalu koreksi kolom `kondisi`** (`normal` / `redup` / `backlight` / `oklusi` / `sudut_ekstrem`). Skrip hanya bisa menebak redup/backlight dari histogram.

### 3. Anotasi di Roboflow
1. Buat project baru bertipe **Instance Segmentation**, lalu unggah isi `foto/jpg/`.
2. Buat kelas dengan **nama persis sama** dengan dataset Panoptic Indoor v5. Notebook mencocokkan kelas berdasarkan nama (huruf besar/kecil diabaikan).
3. Anotasi dengan polygon (boleh pakai Smart Polygon/SAM, tapi periksa dan rapikan tepinya).
4. Generate version **tanpa augmentasi dan tanpa resize**. Semua gambar boleh di satu split. Pilih export format **YOLOv8**.
5. **Screenshot proses anotasi** (wajib): halaman anotasi dengan polygon, daftar kelas, halaman dataset/version. Simpan di `bukti/screenshot_anotasi/`.

### 4. Jalankan notebook di Colab
1. Unggah `notebook/segmentasi_instance_5054241043.ipynb`, pilih *Runtime → T4 GPU*.
2. Simpan API key Roboflow di **Colab Secrets** dengan nama `ROBOFLOW_API_KEY`.
3. Isi `RF_DATASET` dan `RF_OWN` di sel konfigurasi.
4. Unggah `data/kondisi_foto.csv` ke `MyDrive/Tugas3_Segmentasi/`.
5. Jalankan berurutan. Hasil training tersimpan di Drive. Kalau runtime putus, jalankan ulang dari atas: run yang sudah selesai (ada file `SELESAI`) dilewati otomatis.

Perkiraan waktu GPU T4: YOLO n-seg ±20–45 menit per run (4 run), Mask R-CNN ±60–110 menit. Kuota Colab gratis terbatas, jadi bagi menjadi beberapa sesi/hari dan catat di logbook.

> Laptopmu punya RTX 4060, tapi PyTorch yang terpasang sekarang versi CPU. Kalau ingin training lokal, pasang PyTorch versi CUDA dulu. Notebook ini bisa jalan lokal maupun di Colab.

### 5. Analisis & laporan
Struktur laporan yang disarankan:
1. Pendahuluan & dataset (statistik kelas, `statistik_dataset.csv`)
2. Test set buatan sendiri: cara pengambilan, sebaran kondisi, contoh foto, screenshot anotasi, tabel EXIF
3. Model: dua keluarga dan perbedaan mekanisme mask (prototype vs RoI 28×28)
4. Hasil: tabel utama (test asli vs sendiri), kurva training, waktu GPU, kecepatan inferensi
5. Domain shift: per kelas & per kondisi, beserta penjelasan penyebabnya
6. Variabel NRP: grafik 25/50/100%, termasuk apakah tambahan data memperkecil domain shift
7. Analisis kegagalan: 5 foto × (gambar GT vs prediksi, jenis kegagalan, mekanisme, bukti eksperimen)
8. Penggunaan AI (`docs/Penggunaan_AI.md`) dan logbook (lampiran)

Batas waktu pengerjaan: 2 minggu.
