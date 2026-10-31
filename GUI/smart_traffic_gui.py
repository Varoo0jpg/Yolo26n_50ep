# Coba_kamera_2.py
# GUI CustomTkinter (Coba.py) + deteksi kendaraan 4 kamera (YOLO)
#
# Instalasi:
#   pip install customtkinter pillow opencv-python ultralytics numpy
#
# Penanda perubahan terhadap Coba.py asli:
#   [BARU]       = bagian tambahan (deteksi kamera, tampilan feed)
#   [PERBAIKAN]  = bug pada kode asli yang diperbaiki
import collections
import csv
import os
import random
import statistics
import sys
import threading
import time
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

# ===============================================================
# KONFIGURASI SISTEM
# ===============================================================
DAFTAR_LAJUR = ["Utara", "Timur", "Selatan", "Barat"]

# Bobot heuristik untuk skor kepadatan/prioritas lajur.
# Kendaraan darurat diberi prioritas lebih tinggi; ini bukan standar PCU baku.
BOBOT_KENDARAAN = {
    "Ambulance": 5.0,
    "Bus": 1.5,
    "Car": 1.0,
    "Fire engine": 5.0,
    "Military": 2.0,
    "Motorcycle": 0.5,
    "Police car": 5.0,
    "Truck": 1.5,
}

FASE_HIJAU_MIN = 10     # Batas minimal waktu hijau (detik)
FASE_HIJAU_MAX = 40     # Batas maksimal waktu hijau (detik)
FASE_KUNING = 3         # Durasi lampu kuning (detik)

# Kode Warna UI Hexadecimal
HEX_RED = "#e74c3c"
HEX_YELLOW = "#f1c40f"
HEX_GREEN = "#2ecc71"
HEX_DARK_BG = "#1e2530"

# ---------------- [BARU] Konfigurasi kamera & deteksi ----------------
# Sumber tiap lajur: indeks kamera (0,1,2,..), path file video, atau None (input manual).
# Default: hanya kamera laptop untuk lajur Utara.
SUMBER_KAMERA = {"Utara": "https://www.youtube.com/watch?v=Ck45cxJAh-s", "Timur": "https://www.youtube.com/live/UUj6fNons8E?si=GLkVVlcxD_hAjmLP", "Selatan": "https://www.youtube.com/live/BdPvBNLy5kc?si=tm8jdiBL6TW7jkJy", "Barat": "https://www.youtube.com/watch?v=D_R5EpqQIpo"}
# Contoh 4 kamera USB:
# SUMBER_KAMERA = {"Utara": 0, "Timur": 1, "Selatan": 2, "Barat": 3}
# Contoh 4 file video:
# SUMBER_KAMERA = {"Utara": "utara.mp4", "Timur": "timur.mp4",
#                  "Selatan": "selatan.mp4", "Barat": "barat.mp4"}


def resource_path(nama):
    """Lokasi file bawaan; juga bekerja saat dibungkus PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, nama)


# Checkpoint baseline pada repositori Yolo26n_50ep.
# Struktur yang diharapkan: GUI/smart_traffic_gui.py dan Coba/antytet.pt.
MODEL_PATH = os.environ.get("SMART_TRAFFIC_MODEL", os.path.abspath(resource_path(os.path.join("..", "Coba", "antytet.pt"))))
CONF = 0.20  # Baseline confidence untuk evaluasi awal model V2
IMGSZ = 640                    # kecil = lebih cepat di CPU
UKURAN_FRAME = (640, 480)        # ukuran proses deteksi
UKURAN_TAMPIL = (330, 248)       # ukuran feed di kotak kamera GUI

# Urutan kelas diverifikasi terhadap checkpoint dan data.yaml proyek YOLO26n_50ep.
KELAS_YOLO = {
    0: "Ambulance",
    1: "Bus",
    2: "Car",
    3: "Fire engine",
    4: "Military",
    5: "Motorcycle",
    6: "Police car",
    7: "Truck",
}

NAMA_KELAS_MODEL = {
    0: "ambulance",
    1: "bus",
    2: "car",
    3: "fire engine",
    4: "military",
    5: "motorcycle",
    6: "police car",
    7: "truck",
}


def validasi_model_kendaraan(model):
    """Pastikan bobot yang dimuat memiliki 8 kelas dan urutannya sesuai konfigurasi."""
    nama = getattr(model, "names", None)
    if nama is None:
        raise RuntimeError(
            "Model YOLO tidak menyediakan daftar nama kelas. "
            "Periksa MODEL_PATH dan pastikan antytet.pt merupakan bobot YOLO yang valid."
        )

    try:
        jumlah_kelas = len(nama)
    except TypeError:
        jumlah_kelas = 0

    if jumlah_kelas != len(NAMA_KELAS_MODEL):
        raise RuntimeError(
            f"Model pada {MODEL_PATH!r} harus memiliki tepat 8 kelas, "
            f"tetapi ditemukan {jumlah_kelas}."
        )

    for indeks, nama_yang_diharapkan in NAMA_KELAS_MODEL.items():
        if isinstance(nama, dict):
            aktual = nama.get(indeks, nama.get(str(indeks)))
        elif isinstance(nama, (list, tuple)) and indeks < len(nama):
            aktual = nama[indeks]
        else:
            aktual = None

        aktual = str(aktual).strip().lower().replace("_", " ").replace("-", " ") if aktual is not None else ""
        if aktual != nama_yang_diharapkan:
            raise RuntimeError(
                f"Model pada {MODEL_PATH!r} tidak cocok dengan konfigurasi 8 kelas. "
                f"Kelas indeks {indeks} diharapkan {nama_yang_diharapkan!r}, tetapi ditemukan {aktual!r}. "
                "Pastikan antytet.pt adalah hasil Training V2 dengan urutan kelas yang benar."
            )

# ROI per lajur (pecahan frame): area antrean di depan kamera. SESUAIKAN per kamera.
ROI_DEFAULT = [(0.15, 0.30), (0.85, 0.30), (0.98, 1.00), (0.02, 1.00)]
ROI = {l: list(ROI_DEFAULT) for l in DAFTAR_LAJUR}
# False = hitung kendaraan di SELURUH frame kamera (ROI di atas diabaikan)
# True  = hanya hitung kendaraan di dalam poligon ROI (garis kuning)
GUNAKAN_ROI = False

# Pemilihan lajur hijau berikutnya pada mode OTOMATIS:
#   "PADAT"    = lajur dengan skor tertinggi didahulukan (dengan batas tunggu maksimum)
#   "BERGILIR" = urutan tetap Utara -> Timur -> Selatan -> Barat
PILIH_LAJUR_BERIKUT = "PADAT"
BATAS_TUNGGU_MAKS = 180          # detik; lajur yang menunggu lebih lama dipaksa mendapat giliran


# ===============================================================
# [BARU] SUBSISTEM: DETEKSI KENDARAAN DARI KAMERA (THREAD)
# ===============================================================
def resolve_sumber(sumber):
    """[BARU] Ubah link YouTube (live/video) menjadi URL stream langsung via yt-dlp.
    Sumber lain (indeks kamera, file video, None) dikembalikan apa adanya."""
    if isinstance(sumber, str) and ("youtube.com" in sumber or "youtu.be" in sumber):
        import yt_dlp
        # Audio tidak dipakai, jadi stream video-saja (bv*) juga boleh
        opsi = {"quiet": True, "noplaylist": True,
                "format": "bv*[height<=480]/b[height<=480]/bv*/b"}
        with yt_dlp.YoutubeDL(opsi) as ydl:
            info = ydl.extract_info(sumber, download=False)
        return info["url"]
    return sumber


class DetektorKamera(threading.Thread):
    """Membaca semua kamera, deteksi YOLO (batch), hitung kendaraan per jenis dalam ROI."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.running = True
        self.siap = False          # True setelah model dimuat & kamera dicoba dibuka
        self.error = None
        self.caps = {}
        self.frames = {l: None for l in DAFTAR_LAJUR}
        self.jumlah = {l: {j: 0 for j in BOBOT_KENDARAAN} for l in DAFTAR_LAJUR}
        self._riwayat = {l: collections.deque(maxlen=5) for l in DAFTAR_LAJUR}

    def _buka_sumber(self):
        for lajur, sumber in SUMBER_KAMERA.items():
            if sumber is None:
                continue
            try:
                sumber_nyata = resolve_sumber(sumber)
            except Exception as e:
                print(f"[!] Gagal membaca link YouTube lajur {lajur}: {e}")
                continue
            if isinstance(sumber_nyata, int) and os.name == "nt":
                cap = cv2.VideoCapture(sumber_nyata, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(sumber_nyata)
            if cap.isOpened():
                self.caps[lajur] = cap
            else:
                print(f"[!] Sumber lajur {lajur} ({sumber}) tidak terbuka")

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            validasi_model_kendaraan(model)
            self._buka_sumber()
        except Exception as e:
            self.error = repr(e)
            self.siap = True
            return
        self.siap = True

        w, h = UKURAN_FRAME
        waktu_debug_terakhir = 0.0
        while self.running and self.caps:
            imgs = {}
            for lajur, cap in self.caps.items():
                ok, img = cap.read()
                if not ok:
                    if isinstance(SUMBER_KAMERA[lajur], str):     # file video: ulang
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                imgs[lajur] = cv2.resize(img, UKURAN_FRAME)
            if not imgs:
                time.sleep(0.02)
                continue

            daftar = list(imgs)
            hasil = model(
                [imgs[l] for l in daftar],
                conf=CONF,
                imgsz=IMGSZ,
                classes=list(KELAS_YOLO),
                device="cpu",
                verbose=False,
            )

            # DEBUG: ringkasan deteksi tiap 5 detik untuk membantu mengecek
            # apakah model mendeteksi kelas selain Motorcycle pada feed langsung.
            waktu_sekarang = time.time()
            if waktu_sekarang - waktu_debug_terakhir >= 5:
                for lajur_debug, res_debug in zip(daftar, hasil):
                    ringkasan = {nama: [] for nama in KELAS_YOLO.values()}
                    for box_debug in res_debug.boxes:
                        nama_debug = KELAS_YOLO.get(int(box_debug.cls[0]), "Unknown")
                        ringkasan.setdefault(nama_debug, []).append(float(box_debug.conf[0]))
                    terdeteksi = [
                        f"{nama}={len(nilai)} (conf: {', '.join(f'{c:.2f}' for c in nilai)})"
                        for nama, nilai in ringkasan.items() if nilai
                    ]
                    print(
                        f"[DEBUG DETEKSI {lajur_debug}] "
                        + (" | ".join(terdeteksi) if terdeteksi else "tidak ada deteksi")
                    )
                waktu_debug_terakhir = waktu_sekarang

            for lajur, res in zip(daftar, hasil):
                img = imgs[lajur]
                poly = np.array([(int(x * w), int(y * h)) for x, y in ROI[lajur]], np.int32)
                hitung = {j: 0 for j in BOBOT_KENDARAAN}
                for box in res.boxes:
                    jenis = KELAS_YOLO[int(box.cls[0])]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dalam = (not GUNAKAN_ROI) or cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
                    warna = (0, 255, 0) if dalam else (120, 120, 120)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
                    if dalam:
                        hitung[jenis] += 1
                if GUNAKAN_ROI:
                    cv2.polylines(img, [poly], True, (0, 255, 255), 2)

                # median 5 frame terakhir agar angka tidak berkedip
                rw = self._riwayat[lajur]
                rw.append(hitung)
                median = {j: int(round(float(np.median([r[j] for r in rw]))))
                          for j in BOBOT_KENDARAAN}
                teks = " ".join(f"{j[:2]}:{n}" for j, n in median.items())
                cv2.putText(img, f"{lajur}  {teks}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                with self.lock:
                    self.jumlah[lajur] = median
                    self.frames[lajur] = img

    def ambil(self):
        with self.lock:
            jumlah = {l: dict(v) for l, v in self.jumlah.items()}
            frames = {l: (None if f is None else f.copy()) for l, f in self.frames.items()}
        return jumlah, frames

    def stop(self):
        self.running = False
        time.sleep(0.1)
        for cap in self.caps.values():
            cap.release()

# BANYUUU
# ===============================================================
# SUBSISTEM: INPUT DATA KENDARAAN (LAJUR PANEL CLASS)
# ===============================================================
class PanelInputLajur(ctk.CTkFrame):
    def __init__(self, parent, nama_lajur):
        super().__init__(parent, fg_color="#2d2d2d")
        self.nama_lajur = nama_lajur
        self.dikunci = False      # [BARU] True jika angka diisi otomatis oleh kamera

        # Label Title Lajur
        ctk.CTkLabel(self, text=f"Data Kendaraan - {nama_lajur}", font=("Arial", 11, "bold"), text_color="#FFCC00").pack(pady=2)

        # Frame Kontainer Kotak Input
        grid_input = ctk.CTkFrame(self, fg_color="transparent")
        grid_input.pack(fill="x", padx=10, pady=2)
        # Delapan kelas ditata 4 kolom x 2 baris agar panel tidak melebar.
        grid_input.columnconfigure((0, 1, 2, 3), weight=1)

        self.kolom_input = {}
        for indeks, jenis in enumerate(BOBOT_KENDARAAN.keys()):
            baris_kelas = indeks // 4
            kolom_kelas = indeks % 4
            ctk.CTkLabel(
                grid_input, text=jenis, font=("Arial", 9)
            ).grid(row=baris_kelas * 2, column=kolom_kelas, padx=2, pady=1)

            kotak_teks = ctk.CTkEntry(grid_input, width=45, justify="center")
            kotak_teks.insert(0, "0")
            kotak_teks.grid(row=baris_kelas * 2 + 1, column=kolom_kelas, padx=2, pady=2)
            self.kolom_input[jenis] = kotak_teks

    def ambil_skor_bobot(self):
        """Menghitung skor lajur berdasarkan jumlah kendaraan per kelas. Bobot saat ini setara dan bukan nilai PCU baku."""
        total_skor = 0.0
        for jenis, bobot in BOBOT_KENDARAAN.items():
            try:
                jumlah = int(self.kolom_input[jenis].get())
                total_skor += jumlah * bobot
            except ValueError:
                total_skor += 0.0
        return round(total_skor, 2)

    def acak_jumlah_kendaraan(self):
        """Mengisi kolom input dengan angka tiruan acak baru."""
        if self.dikunci:      # [BARU] lajur berkamera tidak diacak
            return
        for kotak_teks in self.kolom_input.values():
            kotak_teks.delete(0, "end")
            kotak_teks.insert(0, str(random.randint(0, 20)))

    # ---------------- [BARU] ----------------
    def set_jumlah(self, data):
        """Mengisi kolom dari hasil deteksi, mis. {'Motor': 3, 'Mobil': 2, ...}."""
        for jenis, kotak in self.kolom_input.items():
            kotak.configure(state="normal")
            kotak.delete(0, "end")
            kotak.insert(0, str(data.get(jenis, 0)))
            if self.dikunci:
                kotak.configure(state="disabled")

    def kunci(self, status=True):
        self.dikunci = status
        for kotak in self.kolom_input.values():
            kotak.configure(state="disabled" if status else "normal")


# ===============================================================
# APLIKASI UTAMA: ENGINE GUI & CORE LOGIC SIMULATION
# ===============================================================
class TrafficGUI(ctk.CTk):
    def __init__(self):
        global MODEL_PATH
        super().__init__()

        # Setup Window Utama Aplikasi
        self.title("Smart Traffic Control System - Deteksi Kamera")
        self.geometry("1550x1000")
        self.minsize(1450, 900)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # --- STATE ENGINE SIMULASI INTERNAL ---
        self.simulasi_aktif = False
        self.fase_saat_ini = "HIJAU"     # Status: "HIJAU" atau "KUNING"
        self.lajur_aktif = "Utara"       # Menunjukkan lajur mana yang sedang hijau
        self.hitung_mundur = 10          # Durasi detik yang tersisa
        self.lajur_terpaksa = None       # Menyimpan antrean paksa lajur dari mode manual

        # [BARU] State kamera & penjadwalan
        self._id_ticker = None
        self.waktu_hijau_terakhir = {l: time.time() for l in DAFTAR_LAJUR}
        self.detektor = None
        self.kotak_kamera = {}
        self.label_kamera = {}
        self.gambar_kamera = {}
        self.jumlah_terakhir = {}
        self._lajur_gagal = set()

        # Pencatatan snapshot penelitian per detik dan per lajur.
        self.research_records = []
        self.research_recording = False
        self._research_after_id = None
        self.research_sample_interval_ms = 1000

        # Konfigurasi Master Grid Layout
        self.grid_columnconfigure(0, weight=0, minsize=360)  # Kolom Kontrol Kiri
        self.grid_columnconfigure(1, weight=1)               # Kolom Grafik Tengah
        self.grid_columnconfigure(2, weight=1)               # Kolom Kamera & Data
        self.grid_columnconfigure(3, weight=1)

        # Pertahankan proporsi layout asli Lampiu:
        # dua baris CCTV sama tinggi; dua baris input menjadi strip pendek.
        # Dengan lebar jendela 1550 px, panel CCTV mendekati bentuk persegi.
        self.grid_rowconfigure(0, weight=1, minsize=240, uniform="camera_rows")
        self.grid_rowconfigure(1, weight=0, minsize=120, uniform="input_rows")
        self.grid_rowconfigure(2, weight=1, minsize=240, uniform="camera_rows")
        self.grid_rowconfigure(3, weight=0, minsize=120, uniform="input_rows")
        self.grid_columnconfigure(2, weight=1, uniform="camera_columns")
        self.grid_columnconfigure(3, weight=1, uniform="camera_columns")

        # Bangun Komponen Tampilan
        self.bangun_panel_kontrol_kiri()
        self.bangun_grid_simulasi_kanan()
        self.bangun_panel_penelitian()

        # Jika bobot tidak tersedia di lokasi default/env, pilih checkpoint secara manual.
        if not os.path.isfile(MODEL_PATH):
            selected_model = filedialog.askopenfilename(
                title="Pilih checkpoint YOLO26n 8 kelas",
                filetypes=[("YOLO checkpoint", "*.pt"), ("Semua file", "*.*")]
            )
            if selected_model:
                MODEL_PATH = selected_model

        # [BARU] Jalankan deteksi kamera
        self.mulai_deteksi()
        self.protocol("WM_DELETE_WINDOW", self.tutup_aplikasi)
        self._research_after_id = self.after(
            self.research_sample_interval_ms, self.capture_research_sample
        )

        # Visualisasi awal status lampu
        self.sinkronisasi_tampilan_ui()

    # -----------------------------------------------------------
    # TATA LETAK UI & WIDGET GENERATOR
    # -----------------------------------------------------------
    def bangun_panel_kontrol_kiri(self):
        """Membangun bilah kendali simulasi di area sebelah kiri."""
        panel_kiri = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color="#1e1e1e")
        panel_kiri.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(panel_kiri, text="PANEL KENDALI UTAMA", font=("Arial", 16, "bold"), text_color="#FFCC00").pack(pady=15)

        # Wadah Indikator Fisik Lampu
        bingkai_lampu = ctk.CTkFrame(panel_kiri, fg_color="#2b2b2b", border_width=1)
        bingkai_lampu.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(bingkai_lampu, text="Status Lampu Lalu Lintas", font=("Arial", 12, "bold")).pack(pady=5)

        self.visual_lampu = {}
        for lajur in DAFTAR_LAJUR:
            baris = ctk.CTkFrame(bingkai_lampu, fg_color="transparent")
            baris.pack(fill="x", padx=20, pady=4)

            ctk.CTkLabel(baris, text=f"Lajur {lajur}: ", width=100, anchor="w").pack(side="left")

            bulatan = ctk.CTkFrame(baris, width=20, height=20, corner_radius=10, fg_color=HEX_RED)
            bulatan.pack(side="left", padx=10)
            self.visual_lampu[lajur] = bulatan

        # Papan Informasi Countdown Status
        bingkai_info = ctk.CTkFrame(panel_kiri, fg_color="#151515", border_width=1, border_color="#333333")
        bingkai_info.pack(fill="x", padx=15, pady=15)

        self.txt_status = ctk.CTkLabel(bingkai_info, text="Status: Non-Aktif", font=("Arial", 14, "bold"), text_color="yellow")
        self.txt_status.pack(pady=5)

        self.txt_timer = ctk.CTkLabel(bingkai_info, text="-- dtk", font=("Arial", 28, "bold"), text_color="#ffffff")
        self.txt_timer.pack(pady=5)

        # Pilihan Mode Operasional
        self.var_mode = ctk.StringVar(value="OTOMATIS")
        bingkai_mode = ctk.CTkFrame(panel_kiri, fg_color="transparent")
        bingkai_mode.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(bingkai_mode, text="Mode Kendali:", font=("Arial", 11, "bold")).pack(anchor="w", padx=5)
        ctk.CTkRadioButton(bingkai_mode, text="Otomatis (Adaptif berdasarkan jumlah kendaraan)", variable=self.var_mode, value="OTOMATIS").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(bingkai_mode, text="Manual (Paksa Klik)", variable=self.var_mode, value="MANUAL").pack(anchor="w", padx=10, pady=2)

        # Tombol Aksi Kendali
        self.btn_mulai = ctk.CTkButton(panel_kiri, text="Mulai Simulasi", fg_color="#27ae60", hover_color="#1e8449", command=self.action_mulai_simulasi)
        self.btn_mulai.pack(fill="x", padx=15, pady=5)

        self.btn_stop = ctk.CTkButton(panel_kiri, text="Hentikan Simulasi", fg_color="#c0392b", hover_color="#962d22", command=self.action_hentikan_simulasi, state="disabled")
        self.btn_stop.pack(fill="x", padx=15, pady=5)

        self.btn_acak = ctk.CTkButton(panel_kiri, text="Acak Lajur Tanpa Kamera", fg_color="#2980b9", hover_color="#1f618d", command=self.action_acak_input_kendaraan)
        self.btn_acak.pack(fill="x", padx=15, pady=5)

    def bangun_grid_simulasi_kanan(self):
        """Membangun penempatan layout grid 4 kamera dan grafik suara."""
        # --- KOLOM 1: AUDIO CONFIDENCE & AREA GRAFIK ---
        f_suara1 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara1.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        ctk.CTkLabel(
            f_suara1,
            text="AUDIO CONFIDENCE",
            font=("Arial", 12, "bold"),
            text_color="#5dade2",
        ).pack(pady=(8, 1))
        ctk.CTkLabel(
            f_suara1,
            text="Probabilitas sirene per simpang",
            font=("Arial", 9),
            text_color="#aab7c4",
        ).pack(pady=(0, 4))

        # Nilai dimulai sebagai None. Persentase baru muncul setelah modul audio
        # mengirim hasil agar GUI tidak menampilkan angka confidence palsu.
        self.audio_confidence_values = {l: None for l in DAFTAR_LAJUR}
        self.audio_confidence_widgets = {}
        for lajur in DAFTAR_LAJUR:
            baris = ctk.CTkFrame(f_suara1, fg_color="transparent")
            baris.pack(fill="x", padx=10, pady=3)

            ctk.CTkLabel(
                baris,
                text=lajur,
                width=52,
                anchor="w",
                font=("Arial", 10),
            ).pack(side="left")

            progress = ctk.CTkProgressBar(
                baris,
                height=10,
                progress_color="#3498db",
            )
            progress.pack(side="left", fill="x", expand=True, padx=(4, 6))
            progress.set(0)

            persen = ctk.CTkLabel(
                baris,
                text="--%",
                width=38,
                anchor="e",
                font=("Arial", 10, "bold"),
            )
            persen.pack(side="right")

            self.audio_confidence_widgets[lajur] = {
                "progress": progress,
                "persen": persen,
            }

        self.lbl_status_audio = ctk.CTkLabel(
            f_suara1,
            text="Menunggu input audio",
            font=("Arial", 9),
            text_color="#aab7c4",
        )
        self.lbl_status_audio.pack(pady=(2, 7))

        # Baris tepat di bawah Audio Confidence dipakai untuk tabel penelitian.

        # Area ini disiapkan untuk grafik yang dikembangkan terpisah.
        f_suara2 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara2.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(
            f_suara2,
            text="[ Area Grafik Analisis ]\n(Dikembangkan terpisah)",
            text_color="#5dade2",
        ).pack(expand=True)

        f_akustik2 = ctk.CTkFrame(self, fg_color="#151a21")
        f_akustik2.grid(row=3, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_akustik2, text="Data Akustik 2", text_color="gray").pack(expand=True)

        # --- KOLOM 2 & 3: BLOK MONITOR KAMERA & DATA INPUT LAJUR ---
        self.kumpulan_panel = {}

        # Kamera 1 - Utara
        self.buat_box_kamera_simulasi(row=0, col=2, nomor=1, nama="Utara")
        self.kumpulan_panel["Utara"] = PanelInputLajur(self, "Utara")
        self.kumpulan_panel["Utara"].grid(row=1, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 2 - Timur
        self.buat_box_kamera_simulasi(row=0, col=3, nomor=2, nama="Timur")
        self.kumpulan_panel["Timur"] = PanelInputLajur(self, "Timur")
        self.kumpulan_panel["Timur"].grid(row=1, column=3, sticky="nsew", padx=4, pady=4)

        # Kamera 3 - Selatan
        self.buat_box_kamera_simulasi(row=2, col=2, nomor=3, nama="Selatan")
        self.kumpulan_panel["Selatan"] = PanelInputLajur(self, "Selatan")
        self.kumpulan_panel["Selatan"].grid(row=3, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 4 - Barat
        self.buat_box_kamera_simulasi(row=2, col=3, nomor=4, nama="Barat")
        self.kumpulan_panel["Barat"] = PanelInputLajur(self, "Barat")
        self.kumpulan_panel["Barat"].grid(row=3, column=3, sticky="nsew", padx=4, pady=4)

        # Pasang Tombol Paksa Manual di bagian bawah tiap-tiap blok data lajur
        for lajur in DAFTAR_LAJUR:
            btn_paksa = ctk.CTkButton(
                self.kumpulan_panel[lajur],
                text=f"Paksa Hijau {lajur}",
                width=120,
                height=22,
                fg_color="#34495e",
                command=lambda l=lajur: self.action_pemicu_manual_lajur(l)
            )
            btn_paksa.pack(pady=4)

    def buat_box_kamera_simulasi(self, row, col, nomor, nama):
        f = ctk.CTkFrame(self, fg_color="#242424", border_width=2, border_color="#555555")
        f.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        lbl = ctk.CTkLabel(f, text=f"CAMERA {nomor}\n(Lajur {nama})", font=("Arial", 14, "bold"))
        lbl.pack(expand=True)
        self.kotak_kamera[nama] = f      # [BARU]
        self.label_kamera[nama] = lbl    # [BARU]

    # -----------------------------------------------------------
    # PANEL DATA PENELITIAN
    # -----------------------------------------------------------
    def bangun_panel_penelitian(self):
        """Tabel penelitian ringkas tepat di bawah Audio Confidence."""
        panel = ctk.CTkFrame(self, fg_color="#151a21", border_width=1)
        panel.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=6, pady=(3, 0))
        ctk.CTkLabel(
            header, text="DATA PENELITIAN",
            font=("Arial", 11, "bold"), text_color="#FFCC00"
        ).pack(side="left")
        self.lbl_research_status = ctk.CTkLabel(
            header, text="Nonaktif | 0 sampel",
            font=("Arial", 9), text_color="#aab7c4"
        )
        self.lbl_research_status.pack(side="right")

        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self.btn_research_record = ctk.CTkButton(
            controls, text="Rekam", width=62, height=23,
            fg_color="#27864b", command=self.toggle_research_recording
        )
        self.btn_research_record.pack(side="left", padx=2)
        ctk.CTkButton(
            controls, text="CSV Mentah", width=88, height=23,
            command=self.export_research_raw
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            controls, text="CSV Ringkasan", width=100, height=23,
            command=self.export_research_summary
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            controls, text="Reset", width=52, height=23,
            fg_color="#7f3b3b", hover_color="#642d2d",
            command=self.clear_research_records
        ).pack(side="right", padx=2)

        table_frame = ctk.CTkFrame(panel, fg_color="transparent")
        table_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=(1, 3))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Tampilan dibuat ringkas agar muat di kolom Audio Confidence.
        # Delapan kelas tetap direkam lengkap dan tersedia di CSV mentah.
        self.research_columns = (
            "time", "lane", "data_source", "total_vehicles",
            "weighted_score", "signal_phase", "active_lane"
        )
        self.research_tree = ttk.Treeview(
            table_frame, columns=self.research_columns, show="headings", height=2
        )
        headings = {
            "time": ("Waktu", 62),
            "lane": ("Lajur", 48),
            "data_source": ("Sumber", 52),
            "total_vehicles": ("Total", 43),
            "weighted_score": ("Skor", 55),
            "signal_phase": ("Fase", 48),
            "active_lane": ("Hijau", 48),
        }
        for col, (label, width) in headings.items():
            self.research_tree.heading(col, text=label)
            self.research_tree.column(
                col, width=width, minwidth=40, anchor="center", stretch=False
            )

        yscroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.research_tree.yview
        )
        self.research_tree.configure(yscrollcommand=yscroll.set)
        self.research_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")

    def toggle_research_recording(self):
        self.research_recording = not self.research_recording
        if self.research_recording:
            self.btn_research_record.configure(
                text="Jeda Rekam", fg_color="#b7791f", hover_color="#946116"
            )
            status = "Perekaman aktif"
        else:
            self.btn_research_record.configure(
                text="Lanjut Rekam", fg_color="#27864b", hover_color="#1f6b3d"
            )
            status = "Perekaman dijeda"
        self.lbl_research_status.configure(
            text=f"{status} | {len(self.research_records)} sampel"
        )

    def capture_research_sample(self):
        """Ambil snapshot setiap detik untuk setiap lajur saat perekaman aktif."""
        if self.research_recording and hasattr(self, "kumpulan_panel"):
            now = datetime.now().astimezone()
            stamp = now.isoformat(timespec="seconds")
            for lajur in DAFTAR_LAJUR:
                panel = self.kumpulan_panel[lajur]
                counts = {}
                for jenis, entry in panel.kolom_input.items():
                    try:
                        counts[jenis] = max(0, int(entry.get()))
                    except (TypeError, ValueError):
                        counts[jenis] = 0

                total = sum(counts.values())
                weighted_score = round(
                    sum(counts[k] * BOBOT_KENDARAAN[k] for k in BOBOT_KENDARAAN), 3
                )
                phase = (
                    self.fase_saat_ini
                    if self.simulasi_aktif and self.lajur_aktif == lajur
                    else ("MERAH" if self.simulasi_aktif else "NONAKTIF")
                )
                record = {
                    "timestamp": stamp,
                    "time": now.strftime("%H:%M:%S"),
                    "lane": lajur,
                    "data_source": (
                        "kamera" if self.detektor is not None and lajur in self.detektor.caps
                        else "manual"
                    ),
                    **counts,
                    "total_vehicles": total,
                    "weighted_score": weighted_score,
                    "signal_phase": phase,
                    "active_lane": self.lajur_aktif if self.simulasi_aktif else "-",
                    "green_remaining_seconds": max(0, int(self.hitung_mundur)) if phase == "HIJAU" else 0,
                    "mode": self.var_mode.get(),
                }
                self.research_records.append(record)
                self.research_tree.insert(
                    "", "end",
                    values=[record.get(col, "") for col in self.research_columns]
                )

            # Maksimal 120 baris pada tampilan; semua catatan tetap tersimpan di memori.
            children = self.research_tree.get_children()
            if len(children) > 120:
                for item in children[:len(children) - 120]:
                    self.research_tree.delete(item)

            self.lbl_research_status.configure(
                text=f"Perekaman aktif | {len(self.research_records)} sampel"
            )

        self._research_after_id = self.after(
            self.research_sample_interval_ms, self.capture_research_sample
        )

    def clear_research_records(self):
        if not self.research_records:
            return
        if not messagebox.askyesno(
            "Hapus data penelitian",
            "Hapus seluruh sampel yang tersimpan pada sesi GUI ini?"
        ):
            return
        self.research_records.clear()
        for item in self.research_tree.get_children():
            self.research_tree.delete(item)
        self.lbl_research_status.configure(
            text=f"{'Perekaman aktif' if self.research_recording else 'Perekaman nonaktif'} | 0 sampel"
        )

    def export_research_raw(self):
        """Ekspor semua snapshot mentah ke CSV UTF-8-SIG."""
        if not self.research_records:
            messagebox.showwarning("Belum ada data", "Mulai perekaman terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            title="Simpan data observasi mentah",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="smart_traffic_observations.csv",
        )
        if not path:
            return
        fields = [
            "timestamp", "lane", "data_source", *BOBOT_KENDARAAN.keys(),
            "total_vehicles", "weighted_score", "signal_phase",
            "active_lane", "green_remaining_seconds", "mode"
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.research_records)
        messagebox.showinfo("Ekspor selesai", f"Data mentah tersimpan:\n{path}")

    def export_research_summary(self):
        """Buat ringkasan per lajur untuk tabel hasil penelitian."""
        if not self.research_records:
            messagebox.showwarning("Belum ada data", "Mulai perekaman terlebih dahulu.")
            return

        rows = []
        for lajur in DAFTAR_LAJUR:
            samples = [r for r in self.research_records if r["lane"] == lajur]
            if not samples:
                continue
            totals = [int(r["total_vehicles"]) for r in samples]
            scores = [float(r["weighted_score"]) for r in samples]
            try:
                start = datetime.fromisoformat(samples[0]["timestamp"])
                end = datetime.fromisoformat(samples[-1]["timestamp"])
                duration_seconds = max(0, int((end - start).total_seconds()))
            except (ValueError, TypeError):
                duration_seconds = 0

            row = {
                "lane": lajur,
                "data_sources": ";".join(sorted({r["data_source"] for r in samples})),
                "sample_count": len(samples),
                "observation_duration_seconds": duration_seconds,
                "mean_vehicles_per_snapshot": round(statistics.mean(totals), 3),
                "sd_vehicles_per_snapshot": round(statistics.pstdev(totals), 3),
                "max_vehicles_per_snapshot": max(totals),
                "mean_weighted_score": round(statistics.mean(scores), 3),
                "max_weighted_score": round(max(scores), 3),
            }
            for jenis in BOBOT_KENDARAAN:
                values = [int(r[jenis]) for r in samples]
                row[f"mean_{jenis}"] = round(statistics.mean(values), 3)
                row[f"max_{jenis}"] = max(values)
            rows.append(row)

        path = filedialog.asksaveasfilename(
            title="Simpan ringkasan tabel penelitian",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="smart_traffic_paper_summary.csv",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        messagebox.showinfo("Ekspor selesai", f"Ringkasan penelitian tersimpan:\n{path}")

    # -----------------------------------------------------------
    # PEMBARUAN AUDIO CONFIDENCE
    # -----------------------------------------------------------
    def perbarui_confidence_audio(self, hasil):
        """
        Memperbarui panel audio confidence dari hasil modul deteksi sirene.

        Args:
            hasil: dictionary dengan nama lajur dan probabilitas 0.0 sampai 1.0.
                   Gunakan None jika audio untuk lajur tersebut belum tersedia.
        Panggil metode ini pada thread GUI/Tkinter.
        """
        if not isinstance(hasil, dict):
            raise TypeError("hasil audio harus berupa dictionary per lajur")

        tersedia = []
        for lajur in DAFTAR_LAJUR:
            nilai = hasil.get(lajur)
            widget = self.audio_confidence_widgets[lajur]

            if nilai is None:
                self.audio_confidence_values[lajur] = None
                widget["progress"].set(0)
                widget["persen"].configure(text="--%", text_color="#aab7c4")
                continue

            try:
                confidence = float(nilai)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Confidence audio untuk lajur {lajur} harus berupa angka 0.0 sampai 1.0 atau None"
                )

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Confidence audio untuk lajur {lajur} di luar rentang 0.0 sampai 1.0"
                )

            self.audio_confidence_values[lajur] = confidence
            widget["progress"].set(confidence)
            widget["persen"].configure(
                text=f"{confidence * 100:.0f}%",
                text_color="#ffffff",
            )
            tersedia.append((lajur, confidence))

        if not tersedia:
            self.lbl_status_audio.configure(
                text="Menunggu input audio",
                text_color="#aab7c4",
            )
        else:
            lajur_tertinggi, skor_tertinggi = max(tersedia, key=lambda item: item[1])
            self.lbl_status_audio.configure(
                text=f"Sinyal tertinggi: {lajur_tertinggi} ({skor_tertinggi * 100:.0f}%)",
                text_color="#5dade2",
            )

    # -----------------------------------------------------------
    # [BARU] INTEGRASI KAMERA
    # -----------------------------------------------------------
    def _teks_kamera(self, lajur, pesan):
        nomor = DAFTAR_LAJUR.index(lajur) + 1
        return f"CAMERA {nomor}\n(Lajur {lajur})\n{pesan}"

    def mulai_deteksi(self):
        self.lajur_berkamera = [l for l in DAFTAR_LAJUR if SUMBER_KAMERA[l] is not None]
        if not self.lajur_berkamera:
            return
        for l in self.lajur_berkamera:
            panel = self.kumpulan_panel[l]
            panel.set_jumlah({})            # mulai dari nol
            panel.kunci(True)               # isian diatur kamera
            self.label_kamera[l].configure(text=self._teks_kamera(l, "Memuat model YOLO..."))
        self.detektor = DetektorKamera()
        self.detektor.start()
        self.after(200, self.poll_kamera)

    def poll_kamera(self):
        """Dipanggil berkala di thread GUI: ambil hasil deteksi, perbarui feed dan angka."""
        det = self.detektor
        if det is None:
            return
        if det.siap:
            if det.error:
                for l in self.lajur_berkamera:
                    self.label_kamera[l].configure(image=None, text=self._teks_kamera(l, "Gagal memuat deteksi\n(input manual)"))
                    self.kumpulan_panel[l].kunci(False)
                print("[!] Detektor gagal:", det.error)
                return
            jumlah, frames = det.ambil()
            for l in self.lajur_berkamera:
                panel = self.kumpulan_panel[l]
                if l not in det.caps:
                    if l not in self._lajur_gagal:
                        self._lajur_gagal.add(l)
                        self.label_kamera[l].configure(text=self._teks_kamera(l, "Kamera tidak terbuka\n(input manual)"))
                        panel.kunci(False)
                    continue
                frame = frames[l]
                if frame is not None:
                    rgb = cv2.cvtColor(cv2.resize(frame, UKURAN_TAMPIL), cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if l not in self.gambar_kamera:
                        self.gambar_kamera[l] = ctk.CTkImage(light_image=pil, dark_image=pil, size=UKURAN_TAMPIL)
                        self.label_kamera[l].configure(image=self.gambar_kamera[l], text="")
                    else:
                        self.gambar_kamera[l].configure(light_image=pil, dark_image=pil)
                if jumlah[l] != self.jumlah_terakhir.get(l):
                    self.jumlah_terakhir[l] = jumlah[l]
                    panel.set_jumlah(jumlah[l])
        self.after(120, self.poll_kamera)

    def tutup_aplikasi(self):
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
        if self._research_after_id is not None:
            self.after_cancel(self._research_after_id)
        if self.detektor is not None:
            self.detektor.stop()
        self.destroy()

    # -----------------------------------------------------------
    # ENGINE CORE SIMULATION LOGIC
    # -----------------------------------------------------------
    def tentukan_lajur_paling_padat(self, lajur_kecuali):
        """Mencari lajur dengan antrean paling padat untuk giliran berikutnya.
        [PERBAIKAN] Jika skor sama, pilih yang paling lama menunggu (versi asli selalu
        memilih kandidat pertama sehingga Selatan/Barat bisa tidak pernah dapat giliran)."""
        kandidat = [l for l in DAFTAR_LAJUR if l != lajur_kecuali]
        now = time.time()
        return max(kandidat, key=lambda l: (self.kumpulan_panel[l].ambil_skor_bobot(),
                                            now - self.waktu_hijau_terakhir[l]))


       
    def hitung_durasi_fase_hijau(self, lajur_target):
        """
        Mengalokasikan durasi hijau berdasarkan beban PCU.
        Setiap lajur mendapat waktu antara FASE_HIJAU_MIN dan
        FASE_HIJAU_MAX, dengan total alokasi hijau 120 detik.
        """
        TOTAL_WAKTU_SIKLUS = 120
        batas_min = FASE_HIJAU_MIN
        batas_max = FASE_HIJAU_MAX
        lajur_list = list(DAFTAR_LAJUR)

        if lajur_target not in lajur_list:
            raise ValueError(f"Lajur tidak dikenal: {lajur_target}")

        jumlah_lajur = len(lajur_list)
        if jumlah_lajur == 0:
            raise ValueError("DAFTAR_LAJUR tidak boleh kosong.")

        # Pastikan total waktu bisa dibagi tanpa melanggar batas tiap lajur.
        if not (
            jumlah_lajur * batas_min
            <= TOTAL_WAKTU_SIKLUS
            <= jumlah_lajur * batas_max
        ):
            raise ValueError(
                "Batas waktu hijau tidak memungkinkan "
                "untuk memenuhi total waktu siklus."
            )

        # Ambil skor PCU; nilai negatif atau input tidak valid dianggap nol.
        skor = {}
        for lajur in lajur_list:
            try:
                nilai = float(self.kumpulan_panel[lajur].ambil_skor_bobot())
                skor[lajur] = nilai if np.isfinite(nilai) and nilai > 0 else 0.0
            except (KeyError, TypeError, ValueError):
                skor[lajur] = 0.0

        # Alokasi berbobot dengan batas minimum dan maksimum.
        alokasi = {}
        lajur_aktif = list(lajur_list)
        sisa_waktu = float(TOTAL_WAKTU_SIKLUS)

        while lajur_aktif:
            total_skor = sum(skor[lajur] for lajur in lajur_aktif)

            if total_skor > 0:
                usulan = {
                    lajur: sisa_waktu * skor[lajur] / total_skor
                    for lajur in lajur_aktif
                }
            else:
                # Jika semua skor tersisa nol, bagi rata.
                usulan = {
                    lajur: sisa_waktu / len(lajur_aktif)
                    for lajur in lajur_aktif
                }

            # Kunci semua alokasi yang melampaui batas, lalu hitung ulang
            # pembagian untuk lajur yang belum ditetapkan.
            di_atas = [
                lajur for lajur in lajur_aktif
                if usulan[lajur] > batas_max
            ]
            di_bawah = [
                lajur for lajur in lajur_aktif
                if usulan[lajur] < batas_min
            ]

            if not di_atas and not di_bawah:
                alokasi.update(usulan)
                break

            for lajur in di_atas:
                alokasi[lajur] = float(batas_max)
                sisa_waktu -= batas_max

            for lajur in di_bawah:
                alokasi[lajur] = float(batas_min)
                sisa_waktu -= batas_min

            lajur_dikunci = set(di_atas + di_bawah)
            lajur_aktif = [
                lajur for lajur in lajur_aktif
                if lajur not in lajur_dikunci
            ]

            # Perlindungan jika perubahan konfigurasi membuat alokasi mustahil.
            if sisa_waktu < -1e-9:
                raise ValueError("Alokasi waktu hijau melebihi total waktu.")
            if lajur_aktif and sisa_waktu < len(lajur_aktif) * batas_min - 1e-9:
                raise ValueError("Sisa waktu tidak cukup untuk batas minimum.")
            if lajur_aktif and sisa_waktu > len(lajur_aktif) * batas_max + 1e-9:
                raise ValueError("Sisa waktu melebihi batas maksimum yang tersedia.")

        # Konversi ke detik bulat; distribusikan sisa detik berdasarkan
        # pecahan terbesar tanpa melewati batas maksimum.
        hasil = {
            lajur: int(np.floor(alokasi[lajur] + 1e-9))
            for lajur in lajur_list
        }
        sisa_detik = TOTAL_WAKTU_SIKLUS - sum(hasil.values())

        urutan = sorted(
            lajur_list,
            key=lambda lajur: alokasi[lajur] - hasil[lajur],
            reverse=True,
        )

        while sisa_detik > 0:
            ada_perubahan = False
            for lajur in urutan:
                if hasil[lajur] < batas_max:
                    hasil[lajur] += 1
                    sisa_detik -= 1
                    ada_perubahan = True
                    if sisa_detik == 0:
                        break

            if not ada_perubahan:
                raise ValueError("Sisa detik tidak dapat dialokasikan.")

        # Validasi akhir agar hasil selalu memenuhi aturan.
        if sum(hasil.values()) != TOTAL_WAKTU_SIKLUS:
            raise ValueError("Total alokasi hijau tidak sama dengan 120 detik.")
        if any(not batas_min <= hasil[lajur] <= batas_max for lajur in lajur_list):
            raise ValueError("Durasi hijau lajur berada di luar batas yang ditentukan.")

        return hasil[lajur_target]

    def pilih_lajur_berikut(self):
        """[BARU] Menentukan lajur hijau berikutnya setelah fase kuning selesai."""
        if self.var_mode.get() == "MANUAL" or PILIH_LAJUR_BERIKUT == "BERGILIR":
            i = DAFTAR_LAJUR.index(self.lajur_aktif)
            return DAFTAR_LAJUR[(i + 1) % len(DAFTAR_LAJUR)]

        # OTOMATIS: anti-starvation dulu, lalu yang terpadat
        now = time.time()
        kandidat = [l for l in DAFTAR_LAJUR if l != self.lajur_aktif]
        terlama = max(kandidat, key=lambda l: now - self.waktu_hijau_terakhir[l])
        if now - self.waktu_hijau_terakhir[terlama] > BATAS_TUNGGU_MAKS:
            return terlama
        return self.tentukan_lajur_paling_padat(self.lajur_aktif)

    def siklus_transisi_fase(self):
        """[PERBAIKAN] Method ini dipanggil ticker tetapi tidak ada di kode asli
        (akan menyebabkan AttributeError saat hitung mundur habis)."""
        if self.fase_saat_ini == "HIJAU":
            self.fase_saat_ini = "KUNING"
            self.hitung_mundur = FASE_KUNING
        else:
            # kuning selesai -> pindah ke lajur berikutnya
            self.waktu_hijau_terakhir[self.lajur_aktif] = time.time()
            if self.lajur_terpaksa:
                lajur_baru = self.lajur_terpaksa
                self.lajur_terpaksa = None
            else:
                lajur_baru = self.pilih_lajur_berikut()
            self.lajur_aktif = lajur_baru
            self.fase_saat_ini = "HIJAU"
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

    def sinkronisasi_tampilan_ui(self):
        """Memperbarui warna bulatan lampu, countdown, dan bingkai kamera."""
        for lajur, widget in self.visual_lampu.items():
            if lajur == self.lajur_aktif:
                warna = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            else:
                warna = HEX_RED
            widget.configure(fg_color=warna)
            # [BARU] bingkai kamera mengikuti warna lampu
            self.kotak_kamera[lajur].configure(border_color=warna if self.simulasi_aktif else "#555555")

        if self.simulasi_aktif:
            warna_teks = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            self.txt_status.configure(text=f"Status: {self.fase_saat_ini} @ {self.lajur_aktif}", text_color=warna_teks)
            self.txt_timer.configure(text=f"{max(self.hitung_mundur, 0)} dtk", text_color="#ffffff")
        else:
            self.txt_status.configure(text="Simulasi Dihentikan", text_color="orange")
            self.txt_timer.configure(text="-- dtk", text_color="gray")

    # -----------------------------------------------------------
    # INTERACTION & TIMING EVENTS
    # -----------------------------------------------------------
    def engine_simulation_ticker(self):
        """Ticker mandiri yang berjalan mundur 1 detik secara periodik."""
        if not self.simulasi_aktif:
            return

        self.hitung_mundur -= 1

        if self.hitung_mundur <= 0:
            self.siklus_transisi_fase()

        self.sinkronisasi_tampilan_ui()

        # [PERBAIKAN] id disimpan agar bisa dibatalkan (hindari ticker ganda saat stop lalu start cepat)
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_mulai_simulasi(self):
        if self.simulasi_aktif:
            return
        self.simulasi_aktif = True
        self.btn_mulai.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_acak.configure(state="disabled")

        # Reset fase awal dan hitung durasi hijau lajur aktif dari data saat ini
        self.fase_saat_ini = "HIJAU"
        self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

        self.sinkronisasi_tampilan_ui()
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_hentikan_simulasi(self):
        self.simulasi_aktif = False
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
            self._id_ticker = None
        self.btn_mulai.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_acak.configure(state="normal")
        self.sinkronisasi_tampilan_ui()

    def action_acak_input_kendaraan(self):
        if self.research_recording:
            messagebox.showwarning(
                "Perekaman penelitian aktif",
                "Input acak dinonaktifkan selama perekaman agar data penelitian tidak tercampur data simulasi."
            )
            return
        for panel in self.kumpulan_panel.values():
            panel.acak_jumlah_kendaraan()      # lajur berkamera dilewati
        if self.fase_saat_ini == "HIJAU" and self.simulasi_aktif:
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)
            self.txt_timer.configure(text=f"{self.hitung_mundur} dtk")

    def action_pemicu_manual_lajur(self, lajur_tujuan):
        """Memaksa sistem memindahkan lampu hijau ke lajur pilihan (hanya Mode Manual)."""
        if self.var_mode.get() != "MANUAL" or not self.simulasi_aktif:
            return

        # Jika lajur yang diklik sudah sedang hijau, abaikan
        if lajur_tujuan == self.lajur_aktif and self.fase_saat_ini == "HIJAU":
            return

        # Pemicu transisi kuning terlebih dahulu sebelum pindah
        self.fase_saat_ini = "KUNING"
        self.hitung_mundur = FASE_KUNING
        self.lajur_terpaksa = lajur_tujuan
        self.sinkronisasi_tampilan_ui()


# ===============================================================
# APLIKASI RUNNER (MAIN LAUNCHER)
# ===============================================================
if __name__ == "__main__":
    app = TrafficGUI()
    app.mainloop()
