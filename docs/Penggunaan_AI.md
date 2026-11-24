# Deklarasi Penggunaan AI — Tugas 3
Muhammad Fathur Aziz — 5054241043

## Alat yang dipakai
- **Claude (Anthropic) melalui Claude Code**: membaca ketentuan PDF, merancang eksperimen, menulis notebook (`segmentasi_instance_5054241043.ipynb`) dan skrip `siapkan_foto.py`, serta menguji notebook secara lokal dengan dataset sintetis kecil.
- Bagian yang **saya kerjakan sendiri**: pengambilan foto, anotasi di Roboflow, menjalankan training di Colab, memilih 5 foto kegagalan, menulis interpretasi/analisis, dan laporan.
  *(Sesuaikan bagian ini dengan yang benar-benar terjadi.)*

## Anotasi test set (WAJIB dideklarasikan)
Anotasi awal test set buatan sendiri dikerjakan oleh **Claude (AI) yang mengoperasikan editor Roboflow lewat ekstensi Claude in Chrome di akun saya**, memakai Smart Select (SAM3) dan polygon manual. Karena memakai akun saya, riwayat anotasi di Roboflow tercatat atas nama saya. Saya kemudian me-review dan mengoreksi anotasi tersebut *(sesuaikan dengan yang benar-benar kamu lakukan)*.

## Prompt utama
1. "lihat folder tugas dan tugas-3, bisakah kamu bantu aku pengerjaan tugasnya? ketentuannya ada di pdfnya"
2. "bantu aku dalam pengambilan foto"
3. "aku sudah ambil gambar ada di folder datatest, bisakah kamu bantu aku annotate semua ini manual di roboflow?" → "coba laksanakan cara 1"

## Kasus kode/saran AI yang salah dan perbaikannya
Kasus berikut terjadi sungguhan saat notebook diuji lokal (20 November 2026). Pilih minimal satu untuk laporan.

### Kasus 1 — Evaluasi COCO crash saat dipanggil kedua kalinya
- **Kode AI**: fungsi `evaluasi()` meneruskan list prediksi langsung ke `COCO.loadRes()`.
- **Gejala**: evaluasi pertama berhasil, tetapi evaluasi kedua pada prediksi yang sama (analisis per kondisi foto) gagal dengan `ValueError: operands could not be broadcast together with shapes (4,) (0,)`.
- **Penyebab**: `pycocotools.COCO.loadRes` **mengubah dict prediksi secara in-place** (menambahkan `bbox` berupa array numpy, `area`, `id`). Pada pemanggilan berikutnya, pengecekan `anns[0]['bbox'] == []` dilakukan pada array numpy dan error.
- **Perbaikan**: setiap evaluasi membuat salinan baru dict prediksi (`image_id`, `category_id`, `segmentation`, `score`). Diverifikasi dengan prediksi = ground truth: mAP = 1.0 pada pemanggilan pertama maupun kedua.

### Kasus 2 — Label terbaca kosong di Windows tanpa pesan error
- **Kode AI**: `glob` menghasilkan path seperti `.../train/images\foto.jpg` (campuran `/` dan `\`), lalu path label dibuat dengan mengganti `/images/` → `/labels/`.
- **Gejala**: notebook berjalan sampai selesai, tetapi tabel statistik menunjukkan **0 instance** untuk semua kelas dan semua mAP = 0. Tidak ada error sama sekali; kesalahan baru ketahuan setelah output diperiksa.
- **Perbaikan**: semua path dinormalisasi dengan `os.path.abspath`, dan folder label dibentuk dengan `os.path.split` plus assertion bahwa citra berada di folder `images/`. Setelah perbaikan, jumlah instance sesuai (mis. train: 15/15/16).
- **Pelajaran**: hasil yang "tidak error" belum tentu benar; output harus selalu diperiksa kewajarannya.

### Kasus 3 — Urutan import menyebabkan PyTorch gagal dimuat (Windows)
- **Gejala**: `OSError: [WinError 1114] ... c10.dll` saat sel import dijalankan.
- **Penyebab**: `cv2`/`pandas` di-import sebelum `torch`, sehingga DLL bentrok di Windows.
- **Perbaikan**: `import torch` dipindah ke baris pertama. Diuji: urutan `numpy, pandas, cv2, torch` gagal, sedangkan `torch, numpy, pandas, cv2` berhasil.

### Kasus 4 — Evaluasi berjalan >1 jam karena akses memori berlompatan (Colab, 22 November 2026)
- **Kode AI**: `unletterbox()` mengembalikan mask hasil `cv2.resize(...).transpose(2, 0, 1)` (array numpy berukuran n×H×W), lalu `prediksi_yolo()` melakukan `md_[k] > 0.5` per objek.
- **Gejala**: pada uji cepat di Colab, sel evaluasi berjalan >20 menit tanpa output dan baru sampai model ke-2 dari 5. GPU 0%, CPU sibuk. Diagnosis dengan `py-spy dump` di terminal Colab menunjukkan hampir semua waktu habis di baris `m = md_[k] > 0.5`.
- **Penyebab**: setelah `transpose`, `md_[k]` adalah *view* yang elemennya berjarak n×4 byte di memori. Untuk foto 1600 px dengan 100 deteksi, setiap mask (±2 juta piksel) dibaca secara berlompatan sehingga cache CPU tidak efektif.
- **Perbaikan**: resize letterbox → ukuran asli dan thresholding dipindah ke GPU (`torch.nn.functional.interpolate`, lalu `> 0.5`). Hasilnya dikirim ke CPU sebagai array biner contiguous. Evaluasi 5 model × 2 test set turun dari perkiraan >1 jam menjadi ±12 menit. Angka mAP tetap konsisten dengan cek silang metrik bawaan Ultralytics.
- **Pelajaran**: kode yang benar secara logika bisa tetap tidak layak dipakai; alur harus diuji pada data berukuran sebenarnya (foto HP 1600 px), bukan hanya data sintetis kecil.

### Catatan tambahan (bukan kesalahan AI, tapi jebakan yang dicek)
Saran yang umum ditemui untuk membatasi waktu training adalah `model.train(time=2)`. Setelah membaca source code Ultralytics (`engine/trainer.py`), ternyata parameter itu **menghitung ulang jumlah epoch agar training mengisi penuh 2 jam**, bukan sekadar batas atas. Karena itu notebook memakai callback penghenti sendiri.
