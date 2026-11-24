# Logbook Tugas 3 — Segmentasi Instance
Muhammad Fathur Aziz — 5054241043

Isi setiap kali bekerja. Catat yang benar-benar terjadi, termasuk yang gagal; dosen menilai proses, bukan hanya hasil akhir.
Waktu GPU tercatat otomatis di `bukti/gpu_time_log.csv`. Di sini tulis konteks dan keputusannya.

| Tanggal | Kegiatan | Kendala | Keputusan & alasan | GPU (menit) |
|---|---|---|---|---|
| 2026-11-20 | Membaca ketentuan tugas; menyiapkan notebook & skrip foto dengan bantuan AI (Claude) | Halaman Roboflow Universe tidak bisa dibaca otomatis (HTTP 403), sehingga slug dataset v5 belum pasti | Dataset diunduh lewat Roboflow API dengan slug yang diisi manual; nama kelas dibaca dari `data.yaml` | 0 |
| 2026-11-20 | Desain eksperimen | Harus ≥2 keluarga arsitektur dan GPU terbatas | YOLO11n-seg & YOLOv8n-seg (one-stage) vs Mask R-CNN (two-stage). Varian *nano* dipilih supaya 4 run YOLO muat di kuota Colab | 0 |
| 2026-11-20 | Variabel NRP (digit 3): jumlah data latih | Pemilihan subset bisa bias | Subset acak bersarang 25 ⊂ 50 ⊂ 100% dengan seed 0; val/test tidak dikurangi | 0 |
| 2026-11-20 | Batas 2 jam GPU | Parameter `time=2` Ultralytics ternyata mengubah jumlah epoch agar training memakai **penuh** 2 jam (dicek di `ultralytics/engine/trainer.py`) | Memakai callback sendiri yang menghentikan training jika epoch berikutnya diperkirakan melewati 2 jam | 0 |
| 2026-11-22 | Memastikan dataset & menyusun rencana foto | Dataset ternyata `panopticindoor/panoptic-indoor-segmentation` v5 (922 citra: train 397 / valid 166 / test 359) dengan 93 kelas, dan banyak yang salah ketik (`ligjt`, `liht`, `palte`, `spon`, `staris`, `towl`, `windo`, …) | Anotasi sendiri hanya memakai nama kelas yang benar; monitor dianotasi sebagai `tv`. Shot list 48 foto di 7 lokasi, 17 di antaranya kondisi sulit | 0 |
| 2026-11-22 | Mengambil 44 foto, lalu menambah 7 foto ruang bersama (total 51); memilih 42 foto (membuang ruang_22–25 yang hampir tanpa kelas dataset dan 5 foto kasur yang nyaris kembar) | Backlight sulit dibuat karena Smart HDR iPhone meratakan cahaya | 42 foto diunggah ke project Roboflow privat `test-ruangan-5054241043` (12 foto kondisi sulit) | 0 |
| 2026-11-22 | Anotasi awal oleh AI (Claude, lewat Claude in Chrome) di editor Roboflow dengan Smart Select (SAM3) | (1) Mask SAM objek besar yang tertutup benda lain (meja, lemari) terpecah jadi beberapa polygon → satu objek terhitung banyak instance; (2) cermin memantulkan isi kamar; (3) shortcut keyboard tak sengaja menandai ruang_40 sebagai "null" (sudah di-undo) | Aturan anotasi: satu objek = satu polygon (potongan kecil dihapus); meja digambar manual/potongan terbesar; `mirror-stuff`, `blanket`, wall/floor/ceiling/rug **tidak** dianotasi; pantulan di cermin tidak dianotasi; meja = `dining table`, monitor = `tv`, semua kotak/kardus = `box`, tas jinjing = `handbag`, lemari = `cabinet-merged`, bantal sandaran sofa = bagian `couch` | 0 |
| 2026-11-22 | Koreksi & cek ulang seluruh anotasi (42 foto) oleh saya sendiri; generate version v1 (Auto-Orient saja) | Anotasi awal AI melewatkan banyak detail | Semua anotasi diperiksa dan diperbaiki manual | 0 |
| 2026-11-22 | Uji cepat notebook di Colab T4 (`UJI_CEPAT=True`: YOLO 2 epoch, Mask R-CNN 1 epoch, folder `Tugas3_Segmentasi_uji`) | (1) Mount Drive gagal karena izin terlambat disetujui → dijalankan ulang; (2) sel evaluasi sangat lambat (>20 menit, GPU 0%), penyebabnya ketemu lewat `py-spy`: thresholding mask pada view numpy berlompatan | `unletterbox` + threshold dipindah ke GPU → evaluasi ±12 menit. Semua sel selesai tanpa error; 42 foto test sendiri terbaca, tidak ada kelas yang tak dikenal; cek silang Ultralytics vs COCOeval konsisten. Angka mAP uji belum bermakna (Mask R-CNN ≈ 0 karena baru 1 epoch warm-up) | 4,4 (training) |
| 2026-11-22 | Run sungguhan di Colab T4 (`UJI_CEPAT=False`, folder `Tugas3_Segmentasi`): Y11 100/50/25% & Y8 100% (60 epoch), Mask R-CNN 100% (24 epoch), evaluasi, analisis, H1/H2 | Tidak ada error. Semua model selesai jauh di bawah batas 2 jam | mAP50-95 test asli vs sendiri: Y11_100 0,094/0,148 · Y11_50 0,074/0,167 · Y11_25 0,066/0,149 · Y8_100 0,078/0,097 · MRCNN 0,095/0,117. Skor test sendiri justru lebih tinggi → perlu dianalisis (komposisi kelas & ukuran objek test sendiri berbeda: 158 objek/42 foto vs 5341/359) | 95,7 |
| 2026-11-24 | Ketentuan tugas diperbarui: variabel NRP untuk **ketiga** model, hipotesis bertanggal sebelum training, riwayat commit Git, demo langsung | Hipotesis seharusnya ditulis sebelum training pertama, padahal 5 run utama sudah selesai 22 Nov | Tidak memundurkan tanggal. Hipotesis ditulis hari ini (lihat bagian *Hipotesis* di bawah), **sebelum** run tambahan Y8_50/Y8_25/MRCNN_50/MRCNN_25 dimulai, dan kronologinya dicatat apa adanya. Notebook: NRP diperluas ke ketiga model, ditambah sel demo, bug legenda grafik H1 diperbaiki | 0 |
|  |  |  |  |  |

## Hipotesis (ditulis 2026-11-24, sebelum run tambahan variabel NRP)
> Catatan kronologi: ketentuan untuk menulis hipotesis sebelum training baru diumumkan pada 24 Nov, setelah run utama (Y11_100, Y11_50, Y11_25, Y8_100, MRCNN_100) selesai pada 22 Nov. Tanggal tidak dimundurkan. Hipotesis ini ditulis sebelum run tambahan Y8_50, Y8_25, MRCNN_50, dan MRCNN_25 dimulai, dan waktunya dapat dicek dari riwayat commit Git.

Model yang unggul kemungkinan adalah YOLO11n-seg karena model tersebut adalah model terbaru dari eksperimen ini. Kelas yang paling bagus kemungkinan adalah dining-table karena bentuk meja dataset dengan data test cukup mirip.


## Catatan keputusan penting
- Kelas yang dipakai di anotasi sendiri: bed, bottle, box, cabinet, chair, couch, dining table, door, handbag, keyboard, laptop, tv, mouse, pillow, toilet.
- Foto kondisi sulit: 12 dari 42 foto (rincian di `kondisi_foto.csv`)
- Model terbaik di test sendiri: YOLO11n-seg karena lebih tahan terhadap domain shift.
