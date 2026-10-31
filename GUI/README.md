# Smart Traffic Light GUI

GUI ini diadaptasi dari aplikasi kamera pada repositori `Lampiu` dan diselaraskan dengan checkpoint YOLO26n 8 kelas pada repositori ini.

## Fitur

- Tampilan empat lajur: Utara, Timur, Selatan, dan Barat.
- Deteksi YOLO pada feed kamera/video dan input manual jika kamera tidak tersedia.
- Validasi jumlah serta urutan kelas model sebelum inferensi.
- Simulasi fase lampu hijau/kuning dan mode otomatis/manual.
- Panel data penelitian dengan pencatatan snapshot per detik untuk setiap lajur.
- Ekspor data observasi mentah dan ringkasan statistik per lajur ke CSV.

## Struktur file yang diharapkan

```text
repository/
├── GUI/
│   ├── smart_traffic_gui.py
│   ├── requirements.txt
│   └── README.md
└── weights/
    └── best.pt
```

Jika checkpoint tidak tersedia pada `weights/best.pt`, GUI akan meminta pengguna memilih file `.pt` saat dijalankan. Lokasi khusus juga dapat ditentukan melalui variabel lingkungan `SMART_TRAFFIC_MODEL`.

## Instalasi dan menjalankan

Dari direktori utama repository:

```bash
python -m pip install -r GUI/requirements.txt
python GUI/smart_traffic_gui.py
```

## Konfigurasi kamera

Ubah dictionary `SUMBER_KAMERA` di bagian konfigurasi. Nilai yang didukung:
- indeks kamera lokal, misalnya `0` atau `1`;
- path video lokal, misalnya `videos/utara.mp4`;
- URL video/live yang dapat dibaca OpenCV melalui resolver `yt-dlp`;
- `None` untuk memakai input manual pada lajur tersebut.

## Data penelitian

Klik **Mulai Rekam** untuk mengambil snapshot setiap 1 detik pada masing-masing lajur. Setiap baris adalah satu snapshot, bukan identitas kendaraan unik. Karena itu, kolom total menunjukkan jumlah objek yang terlihat pada snapshot tersebut, bukan jumlah kendaraan unik selama keseluruhan sesi.

- **Ekspor Data Mentah** menghasilkan seluruh snapshot beserta waktu, lajur, jumlah tiap kelas, total objek, skor berbobot, dan status lampu.
- **Ekspor Ringkasan Paper** menghasilkan statistik per lajur: jumlah sampel, durasi observasi, rerata, simpangan baku, nilai maksimum, dan rerata/maksimum tiap kelas.

Skor berbobot menggunakan bobot heuristik yang didefinisikan di GUI. Skor tersebut bukan nilai PCU terstandar dan harus dijelaskan sebagai indeks internal jika digunakan dalam paper. Data hasil ekspor tetap perlu diperiksa dan disesuaikan dengan protokol penelitian sebelum dipublikasikan.
