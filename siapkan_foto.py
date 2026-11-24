"""Persiapan test set buatan sendiri (Tugas 3 - Segmentasi Instance).

Yang dilakukan skrip ini:
1. Membaca foto asli dari HP (HEIC/JPG/PNG) di folder masukan. File asli TIDAK diubah.
2. Mengonversi ke JPG (rotasi EXIF diterapkan, sisi terpanjang maks 1600 px) dengan nama
   konsisten ruang_01.jpg, ruang_02.jpg, ... -> siap diunggah ke Roboflow.
   Data GPS dibuang dari JPG hasil konversi (privasi: project Roboflow gratis bersifat publik).
3. Menulis bukti/metadata_exif.csv  : EXIF tiap foto (tanggal, perangkat, lensa, ISO, eksposur, flash).
4. Menulis kondisi_foto.csv         : saran kondisi (normal/redup/backlight) dari statistik kecerahan.
   PERIKSA DAN KOREKSI file ini secara manual: oklusi dan sudut ekstrem tidak bisa dideteksi
   otomatis. Notebook memakai file ini untuk analisis performa per kondisi.

Pemakaian:
    pip install pillow pillow-heif opencv-python numpy
    python siapkan_foto.py "folder/foto_asli" --keluar foto_jpg
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Peringatan: pillow-heif belum terpasang, file .HEIC tidak bisa dibaca (pip install pillow-heif)")

EKSTENSI = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
TAG_EXIF = {  # id tag EXIF standar
    "Make": 0x010F, "Model": 0x0110, "DateTime": 0x0132, "Orientation": 0x0112,
    "DateTimeOriginal": 0x9003, "LensModel": 0xA434, "ISO": 0x8827,
    "ExposureTime": 0x829A, "FNumber": 0x829D, "Flash": 0x9209, "FocalLength": 0x920A,
}
IFD_EXIF, IFD_GPS = 0x8769, 0x8825


def baca_exif(img):
    ex = img.getexif()
    sub = ex.get_ifd(IFD_EXIF)
    ambil = lambda k: sub.get(TAG_EXIF[k], ex.get(TAG_EXIF[k], ""))
    eksp = ambil("ExposureTime")
    try:
        eksp = f"1/{round(1 / float(eksp))}" if eksp and float(eksp) < 1 else (f"{float(eksp)}" if eksp else "")
    except (ZeroDivisionError, TypeError, ValueError):
        eksp = str(eksp)
    flash = ambil("Flash")
    return {
        "tanggal_ambil": ambil("DateTimeOriginal") or ambil("DateTime"),
        "perangkat": f"{ambil('Make')} {ambil('Model')}".strip(),
        "lensa": ambil("LensModel"),
        "ISO": ambil("ISO"),
        "eksposur_detik": eksp,
        "f_number": ambil("FNumber"),
        "flash": "" if flash == "" else ("ya" if int(flash) & 1 else "tidak"),
        "ada_gps": "ya" if ex.get_ifd(IFD_GPS) else "tidak",
    }


def statistik_cahaya(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kecil = cv2.resize(g, (400, int(400 * g.shape[0] / g.shape[1])))
    mean = float(kecil.mean())
    terang = float((kecil > 240).mean())
    gelap = float((kecil < 40).mean())
    tajam = float(cv2.Laplacian(kecil, cv2.CV_64F).var())
    if terang > 0.10 and gelap > 0.25:
        saran = "backlight"
    elif mean < 70:
        saran = "redup"
    else:
        saran = "normal"
    return {"kecerahan_rata2": round(mean, 1), "porsi_piksel_terang": round(terang, 3),
            "porsi_piksel_gelap": round(gelap, 3), "ketajaman_laplacian": round(tajam, 1)}, saran


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masukan", help="folder berisi foto asli dari HP")
    ap.add_argument("--keluar", default="foto_jpg", help="folder hasil konversi JPG")
    ap.add_argument("--maks", type=int, default=1600, help="sisi terpanjang (px)")
    ap.add_argument("--awalan", default="ruang", help="awalan nama file hasil")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.masukan) if os.path.splitext(f)[1].lower() in EKSTENSI)
    if not files:
        sys.exit(f"Tidak ada foto di {a.masukan}")
    os.makedirs(a.keluar, exist_ok=True)
    os.makedirs("bukti", exist_ok=True)

    baris_exif, baris_kondisi = [], []
    for i, f in enumerate(files, 1):
        nama = f"{a.awalan}_{i:02d}.jpg"
        img = Image.open(os.path.join(a.masukan, f))
        meta = baca_exif(img)
        exif = img.getexif()
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((a.maks, a.maks), Image.LANCZOS)
        # Salin EXIF tanpa GPS, orientasi di-reset karena piksel sudah diputar.
        exif[TAG_EXIF["Orientation"]] = 1
        if IFD_GPS in exif:
            del exif[IFD_GPS]
        img.save(os.path.join(a.keluar, nama), quality=92, exif=exif.tobytes())

        stat, saran = statistik_cahaya(np.asarray(img))
        baris_exif.append({"file": nama, "file_asli": f, "resolusi_asli": "x".join(map(str, Image.open(os.path.join(a.masukan, f)).size)),
                           **meta, **stat})
        baris_kondisi.append({"file": nama, "kondisi": saran, "saran_otomatis": saran, "catatan": ""})
        print(f"{f:>20} -> {nama}  {meta['tanggal_ambil']}  ISO {meta['ISO']}  kecerahan {stat['kecerahan_rata2']:>5}  saran: {saran}")

    with open("bukti/metadata_exif.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(baris_exif[0]))
        w.writeheader(); w.writerows(baris_exif)
    if os.path.exists("kondisi_foto.csv"):
        print("\nkondisi_foto.csv sudah ada -> tidak ditimpa (supaya koreksi manualmu aman). "
              "Hapus file itu jika ingin membuat ulang.")
    else:
        with open("kondisi_foto.csv", "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=list(baris_kondisi[0]))
            w.writeheader(); w.writerows(baris_kondisi)

    n_sulit = sum(b["saran_otomatis"] != "normal" for b in baris_kondisi)
    print(f"\n{len(files)} foto dikonversi ke '{a.keluar}/'. EXIF: bukti/metadata_exif.csv")
    print(f"Saran otomatis: {n_sulit} foto redup/backlight. Tandai oklusi / sudut_ekstrem secara manual di kondisi_foto.csv.")
    if len(files) < 40:
        print(f"PERHATIAN: baru {len(files)} foto, ketentuan minimal 40.")


if __name__ == "__main__":
    main()
