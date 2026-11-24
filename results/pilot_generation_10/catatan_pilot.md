# Pilot Evaluasi Generation (10 pertanyaan)

Status ini adalah pilot, belum hasil evaluasi penuh 40 pertanyaan. Set pengujian mengambil 10 pertanyaan pertama dari set uji 40 (seed 42) yang sudah disiapkan. Semua angka BLEU/ROUGE/BERTScore membandingkan teks keluaran dengan `summarize` dan `answer` Klinik Hukumonline; metrik ini bukan penilaian kebenaran hukum.

## Hasil generasi

- Nemotron 3.5 Lightning gratis: 10/10 jawaban berhasil.
- ERNIE 5.5 Flash Preview gratis: 4/10 jawaban berhasil, semuanya terpotong; 6/10 gagal karena batas `429`.
- Gemma 4 26B gratis: 0/10, seluruhnya `429`.
- Gemma 4 31B gratis: 0/10, seluruhnya `429`.
- Qwen 3.8 27B gratis: 0/10, varian gratis tidak tersedia (`404`).
- Biaya model pada hasil tercatat $0.

## Keterbatasan

Hanya dua model menghasilkan teks yang dapat dinilai; jumlah sampel efektif berbeda (Nemotron 10, ERNIE 4). Karena itu angka model tidak sebanding kuat, dan pilot ini tidak memenuhi rencana awal Gemma + Qwen/Gemini + model pembanding. Tidak ada hasil manual 1–5 yang dimasukkan. `penilaian_manual_pilot.csv` berisi 4 pertanyaan yang tersedia dari kedua model, untuk penilaian manual eksploratif saja.

Rincian otomatis: `ringkasan_generation.csv`, `metrics_per_jawaban.csv`; keluaran per model: `generation_*.csv`.
