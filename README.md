# Chatbot Konsultasi Hukum Indonesia
> UAS Natural Language Processing — Sistem RAG berbasis Hybrid Retrieval untuk tanya-jawab hukum dari Klinik Hukumonline.

---

## Dataset

**Sumber:** [Hugging Face](https://huggingface.co/datasets/biznetgio/indonesia-law-qa-embeddings) (`embedings.json`)

| Atribut | Detail |
| :--- | :--- |
| Total dokumen | **7.172** entri konsultasi hukum |
| Ukuran file | ~327 MB |
| Bahasa | 97.2% Bahasa Indonesia, 2.8% Bahasa Inggris (rubrik WNA/ekspatriat) |
| Code-mixing | Banyak istilah hukum serapan (*Force Majeure, Legal Opinion, NDA, Due Diligence*) |

### Struktur Kolom

| Kolom | Deskripsi | Median Panjang |
| :--- | :--- | :--- |
| `title` | Judul artikel konsultasi | — |
| `question` | Pertanyaan asli dari masyarakat | 38 kata |
| `summarize` | Ringkasan jawaban pakar hukum | 129 kata |
| `answer` | Ulasan hukum lengkap (dasar UU, PP, Putusan MA) | 829 kata |
| `source` | URL artikel di hukumonline.com | — |
| `embedding` | Vektor bawaan OpenAI 1536-dim (tidak digunakan) | — |

---

## Arsitektur Pipeline

```
User Query
    │
    ├──► [1. Dense Retrieval]
    │        Encode query dengan multilingual-e5-large
    │        Cari Top-K via FAISS (Cosine Similarity)
    │
    ├──► [2. Sparse Retrieval]
    │        Tokenisasi query
    │        Cari Top-K via BM25 (Keyword Matching)
    │
    ├──► [3. Hybrid Fusion]
    │        Gabungkan hasil FAISS + BM25
    │        menggunakan Reciprocal Rank Fusion (RRF)
    │        → Pilih Top-N dokumen final
    │
    └──► [4. Response Generation]
             Kirim field `answer` dokumen terpilih sebagai konteks
             ke LLM (Google Gemini Free Tier)
             → Jawaban konsultasi hukum final
```

---

## Keputusan Teknis & Justifikasi

### 1. Model Embedding: `intfloat/multilingual-e5-large`

Dipilih berdasarkan benchmark empiris [Lazarus NLP Indonesian Sentence Embeddings](https://lazarusnlp.github.io/indonesian-sentence-embeddings/#results):

| Model | MIRACL (nDCG@10) | TyDiQA (nDCG@10) |
| :--- | :--- | :--- |
| **`multilingual-e5-large`** 🥇 | **76.16%** | **97.44%** |
| `multilingual-e5-base` | 75.16% | 96.69% |
| `multilingual-e5-small` | 74.80% | 95.39% |
| `all-Indo-e5 Small-v4` | 73.23% | 96.43% |

Keunggulan tambahan:
- Native multilingual (100+ bahasa) berbasis XLM-RoBERTa Large — aman untuk dokumen Indonesia dan Inggris sekaligus.
- Arsitektur asymmetric E5: prefix `query:` untuk pertanyaan, `passage:` untuk dokumen.

### 2. Field yang Di-embed: `title + question + summarize`

| Field | Alasan |
| :--- | :--- |
| `title` | Topik utama artikel, sangat diskriminatif untuk retrieval |
| `question` | Paling mirip secara semantik dengan pertanyaan user (*symmetric retrieval*) |
| `summarize` | Inti jawaban hukum tanpa noise; median 129 kata, aman di batas 512 token |

> **Mengapa bukan `answer`?** Meski lebih lengkap, `answer` mengandung banyak noise (boilerplate, navigasi artikel, disclaimer) dan efek *embedding dilution* pada teks panjang menurunkan kualitas representasi vektor.

### 3. Retrieval: Hybrid FAISS + BM25

- **FAISS** (Dense): Menangkap kemiripan semantik — cocok untuk pertanyaan dengan bahasa berbeda dari dokumen.
- **BM25** (Sparse): Menangkap kecocokan keyword spesifik — krusial untuk istilah hukum presisi seperti *"Pasal 1365 KUHPerdata"* atau *"PP 24/1997"* yang bisa luput di dense retrieval.
- **Reciprocal Rank Fusion (RRF)**: Menggabungkan kedua ranking list tanpa perlu training tambahan.

### 4. LLM Generator: Google Gemini Free Tier

- Konteks yang dikirim ke LLM adalah field **`answer`** (ulasan hukum lengkap dari pakar), bukan `summarize`.
- Tidak ada biaya (Free Tier Google AI Studio), tanpa perlu kartu kredit.

---

## Struktur Proyek

```
UAS/
├── embedings.json        # Dataset sumber asli
├── requirements.txt      # Dependensi Python
├── reembed.py            # Re-embedding dengan multilingual-e5-large
├── search.py             # Hybrid search engine (FAISS + BM25 + RRF)
├── rag.py                # Pipeline RAG lengkap dengan LLM
├── documents.json        # (Generated) Metadata dokumen bersih
└── vectors.npy           # (Generated) Matriks vektor 7172 x 1024
```

---

## Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Re-Embedding (Jalankan 1 Kali)
```bash
python reembed.py
```
> Estimasi: ~3–5 menit di Google Colab GPU T4 / ~25–40 menit di CPU laptop.

### 3. Jalankan Search Engine
```bash
python search.py
```

### 4. Jalankan Chatbot RAG
Dapatkan API Key gratis di [Google AI Studio](https://aistudio.google.com/), lalu:
```bash
set GEMINI_API_KEY=your_api_key_here
python rag.py
```

