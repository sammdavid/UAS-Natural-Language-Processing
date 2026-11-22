"""
rag_core.py — pipeline RAG yang diekstrak dari rag.ipynb (Sam David).

Dipakai bersama oleh app.py (Gradio) dan eval_generation.py.

Perilaku bawaan sengaja SAMA dengan rag.ipynb:
  - teks BM25 = title + question + summarize, lowercase, split spasi
  - dense = e5-large, "query: {q}", FAISS IndexFlatIP, kandidat = top_k*10
  - RRF k=60, top-3
  - konteks = judul, ringkasan, answer[:1500], URL
  - system prompt "asli" (Konsultan Hukum AI), temperature 0.2, max_tokens 1200
  - rewriter: temperature 0.1, max_tokens 200, fallback ke query asli

Tambahan (opsional, tidak aktif kecuali diminta):
  - metode retrieval "dense" / "bm25" selain "hybrid"
  - prompt "ketat" (penyedia informasi hukum umum; jawab hanya dari dokumen)
  - pilihan model per panggilan, info token/latensi/biaya di call_llm_detail()

Konfigurasi lewat environment variable (atau file .env, TIDAK di-commit):
  OPENROUTER_API_KEY  key OpenRouter milik sendiri (wajib untuk LLM)
  LLM_MODEL           ID model OpenRouter; kosong = daftar fallback Gemma asli
  RAG_DATA_DIR        folder vectors.npy & documents.json (default: folder file ini)
  EMBED_DEVICE        cpu / mps / cuda (default: otomatis)
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

try:  # .env opsional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass

# ---------------------------------------------------------------- konfigurasi
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", Path(__file__).parent))
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# Daftar fallback asli dari rag.ipynb (dipakai jika LLM_MODEL kosong)
DEFAULT_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
]

DISCLAIMER = (
    "Informasi ini bersifat umum, bukan nasihat hukum. Aturan bisa berubah "
    "(mis. KUHP baru berlaku 2 Januari 2026, sedangkan dataset terakhir diperbarui "
    "Agustus 2024). Untuk kasus Anda, konsultasikan dengan advokat atau lembaga bantuan hukum."
)


def get_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_default_model() -> str | None:
    m = os.getenv("LLM_MODEL", "").strip()
    return m or None


# ---------------------------------------------------------------- data & indeks
@dataclass
class Resources:
    docs: list
    vectors: np.ndarray
    faiss_index: object
    bm25_index: object
    embed_model: object = None
    device: str = "cpu"
    load_seconds: dict = field(default_factory=dict)


_RES: Resources | None = None
_LOCK = threading.Lock()


def _pick_device() -> str:
    env = os.getenv("EMBED_DEVICE", "").strip()
    if env:
        return env
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_resources(load_embedder: bool = True) -> Resources:
    """Muat documents.json, vectors.npy, bangun FAISS & BM25, (opsional) model e5.

    Hanya dijalankan sekali per proses (singleton)."""
    global _RES
    with _LOCK:
        if _RES is None:
            import faiss
            from rank_bm25 import BM25Okapi

            t = {}
            t0 = time.time()
            vec_path, doc_path = DATA_DIR / "vectors.npy", DATA_DIR / "documents.json"
            assert vec_path.exists(), f"File {vec_path} tidak ditemukan!"
            assert doc_path.exists(), f"File {doc_path} tidak ditemukan!"
            vectors = np.load(vec_path).astype(np.float32)
            with open(doc_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            assert len(docs) == vectors.shape[0], "Jumlah dokumen != jumlah vektor"
            t["data"] = time.time() - t0

            t0 = time.time()
            faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            faiss_index.add(vectors)
            bm25_corpus = [
                (d["title"] + " " + d["question"] + " " + d["summarize"]).lower().split()
                for d in docs
            ]
            bm25_index = BM25Okapi(bm25_corpus)
            t["index"] = time.time() - t0
            _RES = Resources(docs, vectors, faiss_index, bm25_index, load_seconds=t)

        if load_embedder and _RES.embed_model is None:
            from sentence_transformers import SentenceTransformer

            t0 = time.time()
            _RES.device = _pick_device()
            _RES.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=_RES.device)
            _RES.load_seconds["embedder"] = time.time() - t0
        return _RES


# ---------------------------------------------------------------- retrieval
def _encode_query(query: str) -> np.ndarray:
    res = load_resources()
    return (
        res.embed_model.encode(
            f"query: {query.strip()}", normalize_embeddings=True, convert_to_numpy=True
        )
        .reshape(1, -1)
        .astype(np.float32)
    )


def dense_ranking(query: str, n: int) -> tuple[list[int], list[float]]:
    res = load_resources()
    scores, ids = res.faiss_index.search(_encode_query(query), n)
    return ids[0].tolist(), scores[0].tolist()


def sparse_ranking(query: str, n: int) -> tuple[list[int], list[float]]:
    res = load_resources(load_embedder=False)
    scores = res.bm25_index.get_scores(query.lower().split())
    ids = np.argsort(scores)[::-1][:n].tolist()
    return ids, [float(scores[i]) for i in ids]


def _pack(doc_id: int, **scores) -> dict:
    item = load_resources(load_embedder=False).docs[doc_id].copy()
    item["doc_index"] = doc_id
    item.update(scores)
    return item


def hybrid_search(query: str, top_k: int = 3, rrf_k: int = 60) -> list[dict]:
    """Sama persis dengan hybrid_search() di rag.ipynb (+ field doc_index & peringkat)."""
    n_docs = len(load_resources(load_embedder=False).docs)
    pool = min(top_k * 10, n_docs)
    dense_ids, _ = dense_ranking(query, pool)
    sparse_ids, _ = sparse_ranking(query, pool)

    rrf: dict[int, float] = {}
    for rank, d in enumerate(dense_ids):
        rrf[d] = rrf.get(d, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, d in enumerate(sparse_ids):
        rrf[d] = rrf.get(d, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    d_rank = {d: r + 1 for r, d in enumerate(dense_ids)}
    s_rank = {d: r + 1 for r, d in enumerate(sparse_ids)}
    return [
        _pack(d, rrf_score=round(s, 5), dense_rank=d_rank.get(d), bm25_rank=s_rank.get(d))
        for d, s in ranked
    ]


def search(query: str, method: str = "hybrid", top_k: int = 3) -> list[dict]:
    """method: 'hybrid' (default chatbot), 'dense', atau 'bm25'."""
    method = method.lower()
    if method == "hybrid":
        return hybrid_search(query, top_k=top_k)
    if method == "dense":
        ids, sc = dense_ranking(query, top_k)
        return [_pack(d, dense_score=round(s, 5)) for d, s in zip(ids, sc)]
    if method in ("bm25", "sparse"):
        ids, sc = sparse_ranking(query, top_k)
        return [_pack(d, bm25_score=round(s, 4)) for d, s in zip(ids, sc)]
    raise ValueError(f"Metode retrieval tidak dikenal: {method}")


# ---------------------------------------------------------------- LLM
@dataclass
class LLMResult:
    content: str
    ok: bool
    model: str | None = None  # model yang benar-benar menjawab (dari respons)
    error: str | None = None
    status_code: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    used_reasoning_fallback: bool = False


def _friendly_error(code: int | None, msg: str) -> str:
    if code == 401:
        return "Key OpenRouter ditolak (401). Cek key di openrouter.ai/keys, saldo kredit, dan variabel OPENROUTER_API_KEY."
    if code == 402:
        return "Saldo kredit OpenRouter tidak cukup (402)."
    if code == 429:
        return "Batas laju/kuota model tercapai (429). Tunggu sebentar atau pilih model lain (model :free sering penuh)."
    if code == 404:
        return f"Model tidak ditemukan (404). Cek ID model di openrouter.ai/models. Detail: {msg}"
    return f"API Error {code}: {msg}" if code else msg


def call_llm_detail(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1000,
    model: str | None = None,
    timeout: int = 90,
    disable_reasoning: bool = True,
) -> LLMResult:
    """Panggil OpenRouter dan kembalikan konten + metadata (token, biaya, latensi).

    model=None -> LLM_MODEL dari env -> daftar fallback Gemma asli."""
    key = get_api_key()
    if not key:
        return LLMResult("", False, error="OPENROUTER_API_KEY belum di-set (lihat .env.example).")

    model = model or get_default_model()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/sammdavid/UAS-Natural-Language-Processing",
        "X-Title": "UAS NLP Hukumonline RAG",
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if model:
        payload["model"] = model
    else:
        payload["models"] = DEFAULT_MODELS
    if disable_reasoning:
        payload["reasoning"] = {"enabled": False}

    t0 = time.time()
    used_fallback = False
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        # Beberapa model menolak parameter reasoning; ulangi sekali tanpa parameter itu.
        if r.status_code == 400 and disable_reasoning and "reason" in r.text.lower():
            payload.pop("reasoning", None)
            used_fallback = True
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        latency = time.time() - t0
        try:
            data = r.json()
        except ValueError:
            return LLMResult("", False, model, _friendly_error(r.status_code, r.text[:200]), r.status_code, latency_s=latency)
        if r.status_code != 200 or "error" in data:
            err = data.get("error", {}) if isinstance(data.get("error"), dict) else {}
            code = err.get("code") if isinstance(err.get("code"), int) else r.status_code
            return LLMResult("", False, model, _friendly_error(code, err.get("message", "Unknown error")), code, latency_s=latency)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content")
        if not content:  # perilaku asli: jatuh ke field reasoning
            content = msg.get("reasoning", "")
        usage = data.get("usage") or {}
        return LLMResult(
            content=(content or "").strip(),
            ok=True,
            model=data.get("model", model),
            status_code=200,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
            latency_s=latency,
            used_reasoning_fallback=used_fallback,
        )
    except Exception as e:  # jaringan, timeout, dsb.
        return LLMResult("", False, model, f"Gagal memanggil LLM: {type(e).__name__}: {e}", latency_s=time.time() - t0)


def call_llm(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1000, model: str | None = None) -> str:
    """Kompatibel dengan call_llm() di rag.ipynb: kembalikan string, '[Error ...]' bila gagal."""
    res = call_llm_detail(messages, temperature, max_tokens, model=model)
    if not res.ok:
        return f"[Error saat memanggil LLM: {res.error}]"
    return res.content


# ---------------------------------------------------------------- rewriter
REWRITER_SYSTEM = (
    "Anda adalah asisten pencarian hukum cerdas. Tugas Anda adalah mengubah pertanyaan "
    "informal atau bahasa sehari-hari dari pengguna menjadi kata kunci hukum formal yang tepat "
    "untuk pencarian artikel hukum di Klinik Hukumonline.\n\n"
    "Aturan:\n"
    "1. Ganti kata slang/sehari-hari (contoh: 'bos' -> 'pengusaha pemberi kerja', 'cairin uang lembur' -> 'upah kerja lembur', 'duit kantor kepake' -> 'tindak pidana penggelapan jabatan').\n"
    "2. Output HANYA berupa frasa kata kunci hasil konversi singkat (3-8 kata). Tanpa tanda kutip, tanpa penjelasan tambahan."
)


def rewrite_query(user_query: str, model: str | None = None) -> str:
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM},
        {"role": "user", "content": f"Pertanyaan informal: {user_query}"},
    ]
    rewritten = call_llm(messages, temperature=0.1, max_tokens=200, model=model)
    cleaned = rewritten.replace('"', "").replace("'", "").strip()
    if cleaned and not cleaned.startswith("[Error"):
        return cleaned
    return user_query


# ---------------------------------------------------------------- prompt
SYSTEM_PROMPTS = {
    # Prompt asli rag.ipynb (tidak diubah)
    "asli": (
        "Anda adalah Konsultan Hukum AI profesional dari Klinik Hukumonline.\n"
        "Tugas Anda adalah memberikan jawaban hukum yang akurat, jelas, empatik, dan terstruktur berdasarkan dokumen rujukan yang dilampirkan.\n\n"
        "Panduan Menjawab:\n"
        "1. Berikan kesimpulan/jawaban lugas di awal.\n"
        "2. Sertakan Dasar Hukum (Pasal/UU/Peraturan) yang disebutkan di dokumen rujukan.\n"
        "3. Jelaskan langkah praktis atau jalur hukum yang dapat ditempuh pengguna.\n"
        "4. Cantumkan tautan rujukan artikel Klinik Hukumonline di akhir jawaban.\n"
        "5. Jika dokumen rujukan tidak mencukupi, sampaikan secara jujur tanpa halusinasi."
    ),
    # Prompt ketat (usulan CONTEXT.md bagian 9.1); dipakai default oleh eval_generation.py
    "ketat": (
        "Anda adalah asisten penyedia informasi hukum umum berbahasa Indonesia, bukan pengacara.\n"
        "Jawab HANYA berdasarkan dokumen rujukan yang dilampirkan.\n\n"
        "Aturan:\n"
        "1. Mulai dengan jawaban singkat dan lugas atas pertanyaan.\n"
        "2. Sebutkan dasar hukum (pasal/UU/peraturan) HANYA jika tertulis di dokumen rujukan; jangan menambah pasal dari ingatan.\n"
        "3. Jelaskan langkah praktis yang disebut atau tersirat jelas di dokumen.\n"
        "4. Jika dokumen tidak cukup untuk menjawab, katakan terus terang bahwa informasinya tidak ditemukan di dokumen rujukan.\n"
        "5. Tulis ringkas (maksimal sekitar 300 kata) dan jangan mengulang isi dokumen kata per kata.\n"
        "6. Ingatkan bahwa ini informasi umum, bukan nasihat hukum, dan aturan bisa berubah."
    ),
}

USER_PROMPT_ENDING = {
    "asli": "Berdasarkan dokumen rujukan di atas, mohon berikan nasihat hukum yang komprehensif:",
    "ketat": "Berdasarkan dokumen rujukan di atas saja, jawablah pertanyaan pengguna:",
}


def build_context(retrieved_docs: list[dict], answer_chars: int = 1500) -> str:
    parts = []
    for i, d in enumerate(retrieved_docs, 1):
        snippet = d.get("answer", "")[:answer_chars].strip()
        parts.append(
            f"[DOKUMEN RUJUKAN {i}]\n"
            f"Judul: {d['title']}\n"
            f"Ringkasan Masalah: {d['summarize']}\n"
            f"Penjelasan Hukum: {snippet}\n"
            f"Sumber: {d['source']}\n"
        )
    return "\n---\n".join(parts)


def build_messages(user_question: str, retrieved_docs: list[dict], prompt_style: str = "asli") -> list[dict]:
    user_prompt = (
        f"Pertanyaan Pengguna:\n{user_question}\n\n"
        f"Berikut adalah Dokumen Rujukan Resmi dari Klinik Hukumonline:\n"
        f"{build_context(retrieved_docs)}\n\n"
        f"{USER_PROMPT_ENDING[prompt_style]}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_style]},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------- pipeline
def rag_chat(
    user_question: str,
    use_query_rewriter: bool = True,
    top_k: int = 3,
    model: str | None = None,
    method: str = "hybrid",
    prompt_style: str = "asli",
    temperature: float = 0.2,
    max_tokens: int = 1200,
    rewriter_model: str | None = None,
) -> dict:
    """Alur RAG lengkap. Dengan argumen bawaan, perilakunya sama dengan rag.ipynb.

    Kunci keluaran asli (question, rewritten_query, retrieved_docs, answer) tetap ada;
    ditambah 'llm' (LLMResult) dan 'timing' (detik per tahap)."""
    timing = {}
    search_query, rewritten_text = user_question, None
    if use_query_rewriter:
        t0 = time.time()
        rewritten_text = rewrite_query(user_question, model=rewriter_model or model)
        search_query = rewritten_text
        timing["rewriter"] = time.time() - t0

    t0 = time.time()
    retrieved_docs = search(search_query, method=method, top_k=top_k)
    timing["retrieval"] = time.time() - t0

    messages = build_messages(user_question, retrieved_docs, prompt_style)
    llm = call_llm_detail(messages, temperature=temperature, max_tokens=max_tokens, model=model)
    timing["llm"] = llm.latency_s
    answer = llm.content if llm.ok else f"[Error saat memanggil LLM: {llm.error}]"

    return {
        "question": user_question,
        "rewritten_query": rewritten_text,
        "search_query": search_query,
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "llm": llm,
        "timing": timing,
    }
