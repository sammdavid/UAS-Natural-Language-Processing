"""
app.py — Website chatbot informasi hukum (RAG Klinik Hukumonline), Gradio.

Halaman: Beranda · Tanya hukum · Cara kerja · Tentang & evaluasi
Navigasi lewat sidebar navy; pengaturan eksperimen di accordion tertutup di halaman Tanya.

Desain: cangkang aplikasi selebar layar (sidebar + area kerja), referensi pola dari juro.com
(sapaan, kotak tanya besar, riwayat percakapan, dan sumber di bawah chat). Palet navy, putih,
kuning emas. Font Source Serif 4 (judul) + Public Sans (isi). Pipeline RAG tidak diubah:
semua logika ada di rag_core.py dan fungsi answer() di bawah.

Menjalankan:
  cp .env.example .env      # isi OPENROUTER_API_KEY milik sendiri
  python app.py             # http://127.0.0.1:7860
  python app.py --share     # link publik sementara (mis. dari Colab)
Tanpa key, app tetap jalan sebagai pencarian artikel (tanpa jawaban otomatis).
"""
from __future__ import annotations

import argparse
import html
import os
import time
from urllib.parse import urlparse

import gradio as gr

import rag_core

# ------------------------------------------------------------------ pilihan
MODEL_CHOICES = [
    "(default: Gemma gratis + fallback)",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
    "qwen/qwen3.6-35b-a3b",
    "openai/gpt-6-luna",
]
METHODS = {
    "Gabungan makna + kata kunci (hybrid, default)": "hybrid",
    "Kemiripan makna saja (dense, e5)": "dense",
    "Kecocokan kata kunci saja (BM25)": "bm25",
}
PROMPTS = {
    "Hanya dari artikel rujukan (default)": "ketat",
    "Instruksi asli notebook (rag.ipynb)": "asli",
}
EXAMPLES = [
    "Gaji lembur tidak dibayar, saya harus lapor ke mana?",
    "Bagaimana cara aman membeli tanah girik?",
    "Kena PHK mendadak tanpa pesangon, apa yang bisa dilakukan?",
    "Belanja online, barang tidak dikirim. Bisa lapor?",
]
PAGES = ["beranda", "tanya", "cara", "tentang"]
NAV_LABELS = {"beranda": "Beranda", "tanya": "Tanya hukum", "cara": "Cara kerja", "tentang": "Tentang & evaluasi"}

DISCLAIMER_SHORT = (
    "Jawaban ini informasi hukum umum sebagai panduan awal, bukan patokan utama. Untuk penjelasan "
    "lengkap, baca artikel rujukannya. Aturan bisa berubah (KUHP baru berlaku 2 Januari 2026, "
    "sedangkan dataset terakhir diperbarui Agustus 2024)."
)
DISCLAIMER_SEARCH = (
    "Ini informasi hukum umum, bukan nasihat hukum. Artikel dipilih otomatis dan belum tentu "
    "sesuai keadaan Anda. Aturan bisa berubah (KUHP baru berlaku 2 Januari 2026, sedangkan "
    "dataset terakhir diperbarui Agustus 2024)."
)


# ------------------------------------------------------------------ ikon (SVG garis, satu gaya)
def _icon(paths: str, size: int = 20) -> str:
    return (f'<svg class="ic" aria-hidden="true" focusable="false" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" '
            f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


IC = {
    "scale": '<path d="M12 3v18"/><path d="M5 21h14"/><path d="M4 7h16"/><path d="M7 7l-3 7a3 3 0 0 0 6 0z"/><path d="M17 7l-3 7a3 3 0 0 0 6 0z"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    "link": '<path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    "alert": '<path d="M12 3l9.5 17H2.5z"/><path d="M12 10v4"/><path d="M12 17.5v.01"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
}


# ------------------------------------------------------------------ logika (tidak mengubah pipeline)
SOURCES_EMPTY = (
    '<section class="refs" aria-labelledby="refs-h"><div class="refs-head"><span class="refs-dot"></span>'
    '<h2 id="refs-h">Artikel rujukan</h2></div>'
    '<p class="refs-empty">Setelah Anda bertanya, artikel Klinik Hukumonline yang dipakai untuk menjawab '
    'muncul di sini lengkap dengan tautannya.</p></section>'
)


def _sources_html(docs: list[dict], used_by_model: bool) -> str:
    if not docs:
        return SOURCES_EMPTY
    sub = ("Dipakai model sebagai bahan jawaban." if used_by_model
           else "Paling relevan menurut sistem pencarian.")
    items = []
    for i, d in enumerate(docs, 1):
        url = d.get("source", "")
        if not url.startswith(("https://", "http://")):
            url = ""
        host = urlparse(url).netloc.replace("www.", "") if url else "sumber tidak tersedia"
        items.append(
            f'<li class="ref" style="--i:{i - 1}"><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'<span class="ref-pill">Rujukan {i}</span>'
            f'<span class="ref-title">{html.escape(d["title"])}</span>'
            f'<span class="ref-host">{html.escape(host)} {_icon(IC["link"], 14)}'
            f'<span class="sr-only"> (buka di tab baru)</span></span></a></li>'
        )
    return (f'<section class="refs" aria-labelledby="refs-h"><div class="refs-head"><span class="refs-dot"></span>'
            f'<h2 id="refs-h">Artikel rujukan</h2></div>'
            f'<p class="refs-count" aria-hidden="true">{len(docs):02d}</p>'
            f'<p class="refs-sub">{sub} Buka artikelnya untuk memastikan.</p>'
            f'<ol class="ref-list">{"".join(items)}</ol></section>')


def _detail(out: dict, has_key: bool) -> tuple[str, list[list]]:
    t, llm = out["timing"], out["llm"]
    lines = [f"**Kueri pencarian:** {html.escape(out['search_query'])}"
             + (" _(hasil query rewriter)_" if out["rewritten_query"] is not None else "")]
    parts = [f"rewriter {t['rewriter']:.1f} dtk" if "rewriter" in t else None,
             f"retrieval {t['retrieval'] * 1000:.0f} ms",
             f"LLM {t['llm']:.1f} dtk" if has_key else None]
    lines.append("**Waktu:** " + ", ".join(p for p in parts if p))
    if llm.model and llm.ok:
        lines.append(f"**Model yang menjawab:** `{llm.model}`")
    if llm.completion_tokens:
        tok = f"**Token:** {llm.prompt_tokens} masuk / {llm.completion_tokens} keluar"
        if llm.cost_usd is not None:
            tok += f", biaya ${llm.cost_usd:.5f}"
        lines.append(tok)
    if llm.finish_reason:
        lines.append(f"**finish_reason:** `{llm.finish_reason}`")
    rows = []
    for i, d in enumerate(out["retrieved_docs"], 1):
        score = d.get("rrf_score", d.get("dense_score", d.get("bm25_score")))
        rows.append([i, d["title"], score, d.get("dense_rank"), d.get("bm25_rank"), d["source"]])
    return "  \n".join(lines), rows


def _label(text: str, kind: str = "ok") -> str:
    icon = IC["chat"] if kind == "ok" else IC["alert"]
    return f'<p class="answer-label {kind}">{_icon(icon, 18)}<span>{text}</span></p>'


def answer(question, model_choice, method_label, use_rewriter, top_k, prompt_label):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Tulis pertanyaan dulu, lalu tekan Kirim.")
    has_key = bool(rag_core.get_api_key())
    model = None if model_choice.startswith("(default") else model_choice.strip()

    try:
        if has_key:
            out = rag_core.rag_chat(
                question, use_query_rewriter=bool(use_rewriter), top_k=int(top_k), model=model,
                method=METHODS[method_label], prompt_style=PROMPTS[prompt_label],
            )
        else:  # mode cari artikel: retrieval saja, LLM tidak dipanggil
            t0 = time.time()
            docs = rag_core.search(question, method=METHODS[method_label], top_k=int(top_k))
            out = {"search_query": question, "rewritten_query": None, "retrieved_docs": docs,
                   "llm": rag_core.LLMResult("", False), "timing": {"retrieval": time.time() - t0}}
    except Exception as e:
        raise gr.Error(f"Pertanyaan gagal diproses ({type(e).__name__}). Kirim ulang dalam beberapa detik.")

    llm = out["llm"]
    if not has_key:
        ans = (_label("Jawaban otomatis belum aktif", "warn") +
               "\n\nKunci layanan model bahasa belum diatur, jadi sistem hanya menampilkan "
               "artikel yang paling relevan dengan pertanyaan Anda. Baca artikel di panel rujukan.")
    elif not llm.ok:
        ans = (_label("Jawaban belum bisa dibuat", "warn") +
               f"\n\n{html.escape(llm.error or 'Layanan model bahasa tidak merespons.')}\n\n"
               "Artikel yang relevan tetap ditampilkan di panel rujukan.")
    elif not llm.content:
        ans = (_label("Jawaban kosong", "warn") +
               "\n\nModel tidak mengembalikan jawaban. Kirim ulang pertanyaan, atau baca artikel rujukan.")
    else:
        ans = _label("Jawaban") + "\n\n" + llm.content
        if llm.finish_reason == "length":
            ans += ("\n\n> **Jawaban terpotong** karena batas panjang keluaran. "
                    "Bagian akhirnya mungkin hilang; lihat artikel rujukan.")
    note = DISCLAIMER_SHORT if has_key else DISCLAIMER_SEARCH
    ans += f"\n\n<div class='note' role='note'><strong>Perlu diingat.</strong> {note}</div>"

    info, rows = _detail(out, has_key)
    src = _sources_html(out["retrieved_docs"], used_by_model=has_key and llm.ok)
    return gr.update(value=ans, visible=True), src, info, rows


# ------------------------------------------------------------------ tampilan
FONTS_HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600'
    '&family=Public+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

CSS = """
.gradio-container {
  --navy-950: #0E1F3B; --navy-900: #152A4D; --navy-700: #1F3D6E; --navy-600: #2A4F8A; --navy-50: #EEF2F8; --navy-25: #F6F8FC;
  --gold-400: #E2B33C; --gold-300: #EDCB72; --gold-100: #FBF1D6; --gold-800: #6E520B;
  --white: #FFFFFF; --ink: #14213A; --slate: #5E6B80; --line: #E3E8F0;
  --serif: 'Source Serif 4', Georgia, serif;
  --sans: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --ease: cubic-bezier(.2,.7,.2,1);

  --body-background-fill: var(--white); --background-fill-primary: var(--white); --background-fill-secondary: var(--navy-25);
  --block-background-fill: var(--white); --block-border-color: var(--line); --border-color-primary: var(--line);
  --block-radius: 14px; --block-shadow: none; --block-border-width: 1px;
  --body-text-color: var(--ink); --body-text-color-subdued: var(--slate); --block-label-text-color: var(--slate);
  --block-title-text-color: var(--ink); --block-info-text-color: var(--slate);
  --color-accent: var(--navy-700); --color-accent-soft: var(--navy-50);
  --link-text-color: var(--navy-700); --link-text-color-hover: var(--navy-950); --link-text-color-visited: var(--navy-700);
  --button-primary-background-fill: var(--navy-700); --button-primary-background-fill-hover: var(--navy-950);
  --button-primary-text-color: var(--white); --button-primary-border-color: var(--navy-700);
  --button-secondary-background-fill: var(--white); --button-secondary-background-fill-hover: var(--navy-50);
  --button-secondary-text-color: var(--navy-700); --button-secondary-border-color: #C8D2E1;
  --button-large-radius: 999px; --button-small-radius: 999px;
  --input-background-fill: var(--navy-25); --input-border-color: var(--line); --input-border-color-focus: var(--navy-600);
  --checkbox-background-color-selected: var(--navy-700); --slider-color: var(--navy-700);
  --font: var(--sans); --body-text-size: 16px;
  max-width: none !important; padding: 0 !important; margin: 0 !important; background: var(--white); font-family: var(--sans);
  overflow: clip !important; /* clip (bukan hidden) supaya sidebar sticky tetap menempel */
}
body { background: #FFFFFF; }
.gradio-container, .gradio-container .prose { font-size: 16px; line-height: 1.6; color: var(--ink); }
.gradio-container main, .gradio-container .main, .gradio-container .main.fillable { padding: 0 !important; max-width: none !important; }
.gradio-container > footer, footer.svelte-1rjryqp { display: none !important; }
.ic { flex: none; vertical-align: -0.2em; }
/* tepi sama rata: blok HTML Gradio tidak boleh menjorok 12px dibanding baris tombol */
.workspace .html-container { padding-left: 0 !important; padding-right: 0 !important; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
.gradio-container :focus-visible { outline: 3px solid var(--gold-400) !important; outline-offset: 2px !important; }
.gradio-container button, .gradio-container a { cursor: pointer; }

/* ---------- cangkang: bar atas + area kerja ---------- */
.shell { gap: 0 !important; min-height: 100vh; }
.topbar { position: sticky; top: 0; z-index: 40; background: rgba(255,255,255,.94) !important; backdrop-filter: saturate(1.2) blur(8px);
  border-bottom: 1px solid var(--line) !important; padding: 10px clamp(16px, 3.5vw, 56px) !important; gap: 12px !important;
  align-items: center !important; flex-wrap: nowrap !important; border-radius: 0 !important; }
.topbar > * { flex: 0 0 auto !important; min-width: 0 !important; }
.topbar .brand-cell { flex: 1 1 auto !important; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand-mark { width: 38px; height: 38px; border-radius: 10px; background: var(--gold-400); color: var(--navy-950); display: grid; place-items: center; flex: none;
  font-family: var(--serif); font-weight: 700; font-size: 1.35rem; line-height: 1; }
.brand-name { font-family: var(--serif); font-size: 1.3rem; font-weight: 600; color: var(--navy-950); line-height: 1.05; margin: 0; white-space: nowrap; }
.brand-sub { font-size: .78rem; color: var(--slate); margin: 0; white-space: nowrap; }
.topnav { gap: 2px !important; flex-wrap: nowrap !important; justify-content: flex-end !important; }
.topnav > * { flex: 0 0 auto !important; min-width: 0 !important; }
.topbar button.nav-btn { background: transparent !important; border: 0 !important; box-shadow: none !important; color: var(--slate) !important;
  font-weight: 600 !important; font-size: .95rem !important; min-height: 44px; padding: 8px 14px !important; border-radius: 10px !important; position: relative; }
.topbar button.nav-btn:hover { color: var(--navy-950) !important; background: var(--navy-25) !important; transform: none !important; }
.topbar button.nav-btn.primary { color: var(--navy-950) !important; }
.topbar button.nav-btn::after { content: ""; position: absolute; left: 14px; right: 14px; bottom: 4px; height: 2px; border-radius: 2px; background: var(--gold-400);
  transform: scaleX(0); transition: transform 240ms var(--ease-ui); }
.topbar button.nav-btn.primary::after { transform: scaleX(1); }
.workspace { padding: 28px clamp(16px, 3.5vw, 56px) 40px !important; min-width: 0 !important; background: var(--white); }
.side-note { background: var(--navy-25); border: 1px solid var(--line); border-radius: 12px; padding: 12px; font-size: .84rem; color: var(--slate); line-height: 1.5; }
.side-note strong { color: var(--navy-950); font-weight: 600; }

/* ---------- halaman tanya: sidebar chat (riwayat) yang bisa dibuka-tutup ---------- */
.chat-shell { gap: 24px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }
.chat-side { position: sticky; top: 84px; max-height: calc(100vh - 104px); overflow-y: auto; background: var(--navy-950) !important;
  border-radius: 18px !important; padding: 16px 12px !important; gap: 10px !important; color: #DCE4F2; flex: 0 0 280px !important; width: 280px; }
.chat-shell.side-closed .chat-side { display: none !important; }
.side-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 2px 4px 6px; }
.side-head .brand-name { color: #fff; font-size: 1.2rem; }
.side-head .brand-sub { color: #9FB0CB; }
.chat-side button.side-new { background: var(--gold-400) !important; color: var(--navy-950) !important; border: 0 !important; font-weight: 700 !important;
  min-height: 44px; border-radius: 12px !important; justify-content: flex-start !important; padding-left: 40px !important; position: relative; }
.chat-side button.side-new::before { content: "+"; position: absolute; left: 16px; top: 50%; transform: translateY(-52%); font-size: 1.35rem; font-weight: 400; }
.chat-side button.side-new:hover { background: var(--gold-300) !important; }
.side-search input, .side-search textarea { background: rgba(255,255,255,.07) !important; border: 1px solid rgba(255,255,255,.14) !important; color: #fff !important;
  border-radius: 12px !important; min-height: 44px; padding-left: 38px !important;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239FB0CB' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='M20 20l-3.5-3.5'/%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important; background-position: 12px center !important; background-size: 16px !important; }
.side-search input::placeholder, .side-search textarea::placeholder { color: #8FA0BD !important; }
div.block.side-search, .side-search .block, .side-search label, .side-search .wrap { background: transparent !important; border: 0 !important; padding: 0 !important; box-shadow: none !important; }
.side-label { font-size: .78rem; color: #8FA0BD; margin: 6px 6px 0; }
.side-hist, .side-hist .wrap, .side-hist fieldset { background: transparent !important; border: 0 !important; padding: 0 !important; box-shadow: none !important; }
.side-hist .wrap { display: flex !important; flex-direction: column !important; gap: 2px !important; }
.side-hist label { background: transparent !important; border: 0 !important; color: #C9D3E4 !important; border-radius: 10px !important; padding: 10px 12px !important;
  min-height: 44px; margin: 0 !important; width: 100%; box-shadow: none !important; cursor: pointer; line-height: 1.35; font-size: .9rem !important; }
.side-hist label:hover { background: rgba(255,255,255,.06) !important; color: #fff !important; }
.side-hist label.selected, .side-hist label:has(input:checked) { background: rgba(226,179,60,.16) !important; color: #fff !important; }
.side-hist input[type="radio"] { display: none !important; }
.side-empty { color: #8FA0BD; font-size: .86rem; margin: 2px 6px; }
.chat-side button.side-del { background: transparent !important; border: 1px solid rgba(255,255,255,.14) !important; color: #9FB0CB !important;
  font-weight: 500 !important; min-height: 44px; border-radius: 12px !important; }
.chat-side button.side-del:hover { color: #fff !important; border-color: rgba(255,255,255,.3) !important; }
.chat-side .side-note { background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.1); color: #B9C6DA; }
.chat-side .side-note strong { color: var(--gold-300); }
.chat-main { min-width: 0 !important; }
button.side-toggle, button.side-close { min-height: 44px; min-width: 44px !important; flex: none !important; padding: 0 14px 0 40px !important; position: relative; font-weight: 600 !important; }
button.side-toggle::before { content: ""; position: absolute; left: 14px; top: 50%; width: 18px; height: 18px; margin-top: -9px; background: currentColor;
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='3'/%3E%3Cpath d='M9 4v16'/%3E%3C/svg%3E") center/contain no-repeat;
          mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='3'/%3E%3Cpath d='M9 4v16'/%3E%3C/svg%3E") center/contain no-repeat; }
.chat-side button.side-close { background: transparent !important; border: 0 !important; color: #9FB0CB !important; padding: 0 !important; width: 44px; }
.chat-side button.side-close { font-size: 0 !important; }
.chat-side button.side-close::before { content: "×"; font-size: 1.6rem; line-height: 1; position: static; }
.side-top { flex-wrap: nowrap !important; align-items: center !important; justify-content: space-between !important; gap: 4px !important; }
.side-top > *:first-child { flex: 1 1 auto !important; min-width: 0 !important; }
@media (min-width: 821px) { .chat-side button.side-close { display: none !important; } }
.chat-side button.side-close:hover { color: #fff !important; }
.side-backdrop { display: none; }

/* sembunyikan header tab bawaan: navigasi lewat sidebar */
.main-tabs > .tab-wrapper, .main-tabs [role="tablist"] { display: none !important; }
.main-tabs { border: 0 !important; padding: 0 !important; background: none !important; }
.main-tabs > .tabitem, .main-tabs .tabitem { padding: 0 !important; border: 0 !important; background: none !important; }

/* ---------- gerak (aturan taste-skill: entrance sekali, hover/press, loop di dalam visual) ---------- */
.gradio-container { --m-in: 150ms; --m-out: 120ms; --m-press: 90ms; --m-cycle: 3.2s;
  --ease-ui: cubic-bezier(.2,.8,.2,1); --ease-enter: cubic-bezier(.22,1,.36,1); }
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes lineup { from { transform: translateY(105%); } to { transform: none; } }
@keyframes settle { from { opacity: .001; transform: scale(.98); } to { opacity: 1; transform: none; } }
/* entrance hanya sekali per kunjungan: setelah html.m-entered tidak diputar ulang saat pindah halaman */
html:not(.m-entered) .hero-grid .ln > span { display: inline-block; animation: lineup 700ms var(--ease-enter) both; }
html:not(.m-entered) .hero-grid .ln:nth-child(2) > span { animation-delay: 80ms; }
html:not(.m-entered) .hero-grid .ln:nth-child(3) > span { animation-delay: 160ms; }
html:not(.m-entered) .hero-grid .enter { animation: rise 520ms var(--ease-enter) both; }
html:not(.m-entered) .hero-grid .enter-1 { animation-delay: 420ms; }
html:not(.m-entered) .hero-grid .enter-2 { animation-delay: 480ms; }
html:not(.m-entered) .hero-grid .enter-3 { animation-delay: 540ms; }
html:not(.m-entered) .hero-grid .frame { animation: settle 900ms 120ms var(--ease-enter) both; }

/* hover dan press: tombol naik 1px, tekan turun 1px + skala .985; hanya pointer halus */
.gradio-container button { transition: background-color var(--m-out) var(--ease-ui), border-color var(--m-out) var(--ease-ui),
  color var(--m-out) var(--ease-ui), transform var(--m-out) var(--ease-ui), box-shadow var(--m-out) var(--ease-ui) !important; }
@media (hover: hover) and (pointer: fine) {
  .workspace button:not(:disabled):hover { transform: translateY(-1px); transition-duration: var(--m-in) !important; }
  .workspace button.primary:not(:disabled):hover { background: color-mix(in srgb, var(--navy-700) 86%, var(--navy-950)) !important; }
  .ref a:hover { transform: translateY(-2px); border-color: var(--navy-600) !important; box-shadow: 0 6px 24px rgba(14,31,59,.10) !important; }
}
.workspace button:not(:disabled):active { transform: translateY(1px) scale(.985) !important; transition-duration: var(--m-press) !important; }
.ref a { transition: transform 180ms var(--ease-ui), border-color 180ms var(--ease-ui), box-shadow 180ms var(--ease-ui) !important; }

/* ---------- beranda: layar 1, hero ---------- */
.hero-grid { min-height: calc(100vh - 68px); align-items: center !important; gap: clamp(24px, 4vw, 64px) !important; flex-wrap: nowrap !important; }
.hero-copy { gap: 0 !important; }
.hero h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(2.5rem, 4.6vw, 4.2rem); line-height: 1.06; letter-spacing: -0.015em;
  color: var(--navy-950); margin: 0 0 20px; max-width: 15ch; }
.hero h1 .ln { display: block; overflow: hidden; padding-bottom: .06em; }
.hero p.lead { font-size: 1.12rem; color: var(--slate); max-width: 46ch; margin: 0 0 26px; }
.hero p.lead b { color: var(--navy-950); font-weight: 600; }
.hero-actions { margin-top: 14px !important; gap: 10px !important; flex-wrap: wrap !important; }
.hero-actions > * { flex: 0 0 auto !important; }
.hero-actions button { min-height: 44px; }

.frame { background: var(--white); border: 1px solid var(--line); border-radius: 20px; overflow: hidden; position: relative;
  box-shadow: 0 1px 2px rgba(14,31,59,.06), 0 28px 60px -32px rgba(14,31,59,.45); }
.frame-bar { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--line); background: var(--navy-25); }
.frame-bar .mode-pill { font-size: .8rem; padding: 4px 10px; }
.frame-bar .who { font-size: .8rem; color: var(--slate); }
.frame-body { padding: 18px 18px 64px; display: grid; gap: 12px; }
.fq { justify-self: end; max-width: 86%; background: var(--navy-700); color: #fff; border-radius: 16px 16px 4px 16px; padding: 10px 14px; font-size: .95rem; line-height: 1.5; }
.fstatus { display: flex; align-items: center; gap: 10px; font-size: .9rem; color: var(--navy-700); font-weight: 600; }
.fstatus .spin, .fstatus .ok { width: 16px; height: 16px; flex: none; }
.fstatus .spin { border: 2px solid var(--navy-50); border-top-color: var(--navy-700); border-radius: 50%; }
.fstatus .ok { color: #1E7A4A; }
.fstatus .lbl-done { display: none; }
.frefs { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
.frefs li { display: flex; gap: 10px; align-items: flex-start; border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; background: #fff; font-size: .9rem; line-height: 1.35; color: var(--navy-950); font-weight: 600; }
.frefs li .p { flex: none; background: var(--gold-100); color: var(--gold-800); border-radius: 999px; padding: 1px 8px; font-size: .74rem; font-weight: 700; margin-top: 1px; }
.pause { position: absolute; right: 12px; bottom: 12px; width: 44px; height: 44px; border-radius: 50%; border: 1px solid var(--line); background: #fff;
  display: grid; place-items: center; color: var(--navy-700); cursor: pointer; }
.pause .i-play { display: none; }
.frame.paused .pause .i-play { display: block; } .frame.paused .pause .i-pause { display: none; }

/* loop app-frame: satu siklus 3,2 dtk, mulai dari gambar jadi, 15% terakhir menahan gambar jadi, reset meredup (tidak kosong) */
@keyframes f-spin { to { transform: rotate(360deg); } }
@keyframes f-status { 0%, 22% { opacity: 1; } 26%, 100% { opacity: 0; } }
@keyframes f-done { 0%, 22% { opacity: 0; } 26%, 100% { opacity: 1; } }
@keyframes f-row1 { 0%, 30% { opacity: .35; } 38%, 100% { opacity: 1; } }
@keyframes f-row2 { 0%, 50% { opacity: .35; } 58%, 100% { opacity: 1; } }
@keyframes f-row3 { 0%, 70% { opacity: .35; } 78%, 100% { opacity: 1; } }
.frame.live .fstatus .spin { animation-name: f-spin; animation-duration: 1s; animation-timing-function: linear; animation-iteration-count: infinite; }
.frame.live .st-run { animation-name: f-status; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; }
.frame.live .st-done { animation-name: f-done; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; }
.frame.live .frefs li:nth-child(1) { animation-name: f-row1; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; }
.frame.live .frefs li:nth-child(2) { animation-name: f-row2; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; }
.frame.live .frefs li:nth-child(3) { animation-name: f-row3; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; }
.fstack { display: grid; } .fstack > * { grid-area: 1 / 1; }
.frame:not(.live) .st-run { opacity: 0; }
.frame.paused * { animation-play-state: paused !important; }

/* ---------- beranda: layar 2, pita navy + diagram (momen halaman) ---------- */
.flow-band { background: var(--navy-950); color: #DCE4F2; border-radius: 24px; padding: clamp(28px, 4vw, 56px); margin-top: 24px; }
.flow-band h2 { font-family: var(--serif); font-weight: 400; color: #fff; font-size: clamp(1.8rem, 3vw, 2.6rem); line-height: 1.12; margin: 0 0 12px; max-width: 22ch; }
.flow-band h2 .muted { color: #8FA0BD; }
.flow-band p.sub { color: #B9C6DA; max-width: 60ch; margin: 0 0 28px; font-size: 1.05rem; }
.dg { width: 100%; height: auto; display: block; }
.dg .edge { fill: none; stroke: rgba(255,255,255,.22); stroke-width: 1.5; }
.dg .pulse { fill: none; stroke: url(#comet); stroke-width: 3; stroke-linecap: round; stroke-dasharray: 14 100; stroke-dashoffset: 114; opacity: 0; }
.dg .node rect { fill: #13284A; stroke: rgba(255,255,255,.16); }
.dg .node.core rect { fill: #1F3D6E; stroke: var(--gold-400); }
.dg .node text.t { fill: #fff; font: 600 15px var(--sans); }
.dg .node text.m { fill: #9FB0CB; font: 500 12.5px var(--sans); }
.dg .node.end rect { fill: var(--gold-400); stroke: var(--gold-400); }
.dg .node.end text.t { fill: var(--navy-950); } .dg .node.end text.m { fill: #4D3A08; }
@keyframes comet { 0% { stroke-dashoffset: 114; opacity: 0; } 4% { opacity: 1; } 34% { stroke-dashoffset: 0; opacity: 1; } 40%, 100% { stroke-dashoffset: 0; opacity: 0; } }
.flow-band.live .dg .pulse { animation-name: comet; animation-duration: var(--m-cycle); animation-timing-function: linear; animation-iteration-count: infinite; animation-delay: calc(var(--d) * 1s); }
.flow-band { position: relative; }
.flow-band .pause { background: transparent !important; border-color: rgba(255,255,255,.25) !important; color: #fff !important; right: 18px; bottom: 18px; }
.flow-band.paused * { animation-play-state: paused !important; }
.flow-list { display: none; margin: 0; padding: 0; list-style: none; counter-reset: f; }
.flow-list li { counter-increment: f; display: flex; gap: 12px; align-items: baseline; padding: 10px 0; border-top: 1px solid rgba(255,255,255,.12); color: #fff; font-weight: 600; }
.flow-list li::before { content: counter(f); color: var(--gold-400); font-family: var(--serif); font-size: 1.3rem; min-width: 18px; }
.flow-list li span { display: block; color: #9FB0CB; font-weight: 400; font-size: .92rem; }

/* ---------- beranda: layar 3, penutup ---------- */
.close-sec { padding: clamp(48px, 9vh, 96px) 0 24px; }
.close-sec h2 { font-family: var(--serif); font-weight: 400; color: var(--navy-950); font-size: clamp(2rem, 3.6vw, 3.2rem); line-height: 1.1; margin: 0 0 14px; max-width: 20ch; }
.close-sec h2 .muted { color: #8A95A8; }
.close-sec p.sub { color: var(--slate); max-width: 58ch; margin: 0; font-size: 1.05rem; }
.close-actions { margin-top: 26px !important; }
.close-actions > * { flex: 0 0 auto !important; }
.close-actions button { min-height: 48px; }
.foot { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--slate); font-size: .85rem; }

.askbar { background: var(--navy-25) !important; border: 1px solid var(--line) !important; border-radius: 20px !important;
  padding: 10px 10px 10px 18px !important; gap: 10px !important; align-items: flex-end !important; flex-wrap: nowrap !important;
  box-shadow: 0 1px 0 rgba(14,31,59,.03), 0 12px 32px -18px rgba(14,31,59,.35); transition: border-color .15s var(--ease), box-shadow .15s var(--ease); }
.askbar:focus-within { border-color: var(--navy-600) !important; box-shadow: 0 0 0 4px rgba(42,79,138,.12), 0 12px 32px -18px rgba(14,31,59,.35); }
.askbar textarea { background: transparent !important; border: 0 !important; box-shadow: none !important; font-size: 1.1rem !important; line-height: 1.5 !important; padding: 8px 0 !important; }
.askbar .block, .askbar label, .askbar .wrap { background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; }
.askbar button.send { min-height: 48px; min-width: 104px !important; flex: none !important; font-weight: 600 !important; }
.hero-ask { max-width: 640px; margin-top: 0 !important; }

.chips { flex-wrap: wrap !important; gap: 10px !important; margin-top: 14px !important; }
.chips > * { flex: 0 1 auto !important; min-width: 0 !important; }
.chips button.chip { position: relative; display: block !important;
  min-height: 44px; padding: 8px 36px 8px 16px !important; font-size: .93rem !important; font-weight: 500 !important; white-space: normal; text-align: left;
  background: var(--white) !important; border: 1px solid var(--line) !important; color: var(--ink) !important; border-radius: 999px !important; }
.chips button.chip::after { content: ""; position: absolute; right: 14px; top: 50%; margin-top: -6px; width: 12px; height: 12px; background: var(--navy-600);
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 19V5M5 12l7-7 7 7'/%3E%3C/svg%3E") center/contain no-repeat;
          mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 19V5M5 12l7-7 7 7'/%3E%3C/svg%3E") center/contain no-repeat; }
.chips button.chip:hover { border-color: var(--navy-600) !important; background: var(--navy-25) !important; }

.band { background: var(--navy-950); color: #E6ECF6; border-radius: 20px; padding: 28px clamp(20px, 3vw, 40px); margin-top: 40px;
  display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 28px; align-items: center; }
.band p.big { font-family: var(--serif); font-weight: 300; font-size: clamp(1.35rem, 2.2vw, 1.9rem); line-height: 1.25; margin: 0; color: #fff; }
.band p.big b { font-weight: 600; color: var(--gold-300); }
.band ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
.band li { display: flex; gap: 10px; align-items: flex-start; color: #D2DBEA; }
.band li .ic { color: var(--gold-400); margin-top: 3px; }

.split { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 36px; border-top: 1px solid var(--line); }
.split > div { padding: 22px 24px 4px 0; }
.split > div + div { padding-left: 24px; border-left: 1px solid var(--line); }
.split h3 { font-family: var(--serif); font-weight: 600; font-size: 1.25rem; color: var(--navy-950); margin: 0 0 6px; }
.split p { color: var(--slate); margin: 0; max-width: 42ch; }

/* ---------- tanya ---------- */
.ask-top { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 4px; }
.ask-top { flex-wrap: nowrap !important; }
.ask-top > * { flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; }
.mode-pill { white-space: nowrap; display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; background: var(--navy-50); color: var(--navy-700); font-weight: 600; font-size: .88rem; }
.mode-pill .tag { background: var(--gold-400); color: var(--navy-950); border-radius: 999px; padding: 1px 8px; font-size: .78rem; font-weight: 700; }
.ask-layout { gap: 28px !important; align-items: flex-start !important; flex-wrap: nowrap !important; }
.refs-col { background: var(--navy-25); border: 1px solid var(--line) !important; border-radius: 18px !important; padding: 20px !important; }
.chat-history { min-height: 420px; max-height: min(62vh, 760px); overflow-y: auto; border: 1px solid var(--line) !important; border-radius: 18px !important; background: var(--navy-25) !important; padding: clamp(16px, 3vw, 30px) !important; }
.chat-history .message { font-size: 1rem !important; line-height: 1.7 !important; }
.chat-history .message.user { background: var(--navy-700) !important; color: white !important; border-radius: 18px 18px 4px 18px !important; }
.chat-history .message.user, .chat-history .message.user * { color: #FFFFFF !important; }
.chat-history .message.bot { background: white !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 18px 18px 18px 4px !important; }
.chat-tools { display:flex; justify-content:space-between; align-items:center; gap:12px; margin: 12px 0 2px; }
.ask-main { width:100%; min-width:0; }
.sources-accordion { margin-top: 20px !important; border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--white) !important; }
.sources-accordion .refs-count { font-size: 2rem; }
.refs-head { display: flex; align-items: center; gap: 8px; }
.refs-head h2 { font-family: var(--sans); font-size: 1rem; font-weight: 600; color: var(--navy-950); margin: 0; }
.refs-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--gold-400); box-shadow: 0 0 0 4px rgba(226,179,60,.2); }
.refs-count { font-family: var(--serif); font-weight: 600; font-size: 4rem; line-height: 1; color: var(--navy-950); margin: 10px 0 4px; font-variant-numeric: tabular-nums; }
.refs-sub, .refs-empty { color: var(--slate); font-size: .92rem; margin: 0 0 14px; }
.refs-empty { margin-top: 10px; }
.ref-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.ref { animation: rise .45s calc(var(--i) * 70ms) var(--ease) both; }
.ref a { display: flex; flex-direction: column; align-items: flex-start; text-align: left; gap: 6px; background: var(--white); border: 1px solid var(--line); border-radius: 14px;
  padding: 14px; text-decoration: none !important; color: var(--ink) !important; transition: border-color .15s var(--ease), box-shadow .15s var(--ease); }
.ref a:hover { border-color: var(--navy-600); box-shadow: 0 8px 20px -14px rgba(14,31,59,.5); }
.ref-pill { background: var(--gold-100); color: var(--gold-800); border-radius: 999px; padding: 2px 10px; font-size: .78rem; font-weight: 700; }
.ref-title { font-weight: 600; color: var(--navy-950); overflow-wrap: anywhere; line-height: 1.35; }
.ref-host { color: var(--slate); font-size: .84rem; display: inline-flex; gap: 4px; align-items: center; }

.greet { margin: 2vh 0 18px; }
.greet h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(2.2rem, 4vw, 3.4rem); line-height: 1.05; color: var(--navy-950); margin: 0; }
.greet p { font-family: var(--serif); font-weight: 300; font-size: clamp(1.5rem, 2.6vw, 2.2rem); line-height: 1.15; color: #8A95A8; margin: 4px 0 0; }
.ask-note { color: var(--slate); font-size: .86rem; margin: 14px 2px 0; }
div.block.answer-card { margin-top: 22px; background: var(--white); border: 1px solid var(--line) !important; border-left: 4px solid var(--gold-400) !important;
  border-radius: 16px; padding: 8px 24px 18px !important; animation: rise .4s var(--ease) both; }
.prose.answer-card { border: 0 !important; padding: 0 !important; background: none !important; max-width: 72ch; }
.answer-label { display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--navy-700); margin: 12px 0 0 !important; }
.answer-label.warn { color: var(--gold-800); }
.note { position: relative; background: var(--gold-100); border-radius: 12px; padding: 10px 12px 10px 40px; color: #4D3A08; font-size: .92rem; margin-top: 14px; }
.note::before { content: ""; position: absolute; left: 12px; top: 12px; width: 18px; height: 18px; background: var(--gold-800);
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5M12 8v.01'/%3E%3C/svg%3E") center/contain no-repeat;
          mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5M12 8v.01'/%3E%3C/svg%3E") center/contain no-repeat; }
.hist-dd { min-width: 280px !important; max-width: 360px; background: var(--white) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; padding: 6px 10px !important; }
.hist-dd label > span:first-child, .hist-dd .label-wrap { font-size: .78rem !important; color: var(--slate) !important; }
button.delhist { color: var(--slate) !important; }
button.newchat { min-height: 44px; font-size: .9rem !important; flex: none !important; }
.exp-acc { margin-top: 26px !important; border-style: dashed !important; border-radius: 14px !important; }

/* ---------- cara kerja & tentang ---------- */
.page-head h1 { font-family: var(--serif); font-weight: 300; font-size: clamp(2rem, 3.6vw, 3rem); color: var(--navy-950); margin: 8px 0 6px; line-height: 1.1; }
.page-head p { color: var(--slate); max-width: 62ch; margin: 0 0 22px; font-size: 1.05rem; }
.steps { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; counter-reset: s; border-top: 2px solid var(--navy-950); }
.steps li { padding: 16px 18px 8px 0; position: relative; }
.steps li + li { padding-left: 18px; border-left: 1px solid var(--line); }
.steps .n { font-family: var(--serif); font-size: 2.4rem; font-weight: 300; color: var(--gold-400); line-height: 1; }
.steps h3 { font-family: var(--sans); font-size: 1.02rem; font-weight: 700; color: var(--navy-950); margin: 10px 0 6px; }
.steps p { color: var(--slate); font-size: .95rem; margin: 0; }
.callout { background: var(--gold-100); border-radius: 14px; padding: 14px 16px; color: #4D3A08; margin-top: 26px; max-width: 80ch; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0; border-top: 2px solid var(--navy-950); margin-bottom: 18px; }
.facts > div { padding: 16px 18px 12px 0; }
.facts > div + div { padding-left: 18px; border-left: 1px solid var(--line); }
.facts .v { font-family: var(--serif); font-size: 2rem; font-weight: 400; color: var(--navy-950); line-height: 1.1; font-variant-numeric: tabular-nums; }
.facts .l { color: var(--slate); font-size: .92rem; margin-top: 4px; }
.prose table { display: block; overflow-x: auto; max-width: 100%; }
.prose th { background: var(--navy-50); color: var(--navy-950); }
.status { display: inline-block; padding: 2px 10px; border-radius: 999px; background: var(--gold-100); color: var(--gold-800); font-size: .82rem; font-weight: 700; white-space: nowrap; }

/* ---------- beranda: nama + tagline (momen pembuka) ---------- */
.wordmark { display: flex; align-items: center; gap: clamp(10px, 1.4vw, 18px); margin: 0 0 6px; }
.wm-word { font-family: var(--serif); font-weight: 600; font-size: clamp(3.6rem, 7.4vw, 6.6rem); line-height: .95; letter-spacing: -0.02em; color: var(--navy-950); display: inline-flex; }
.wm-word > span { display: inline-block; }
.wm-seal { width: clamp(52px, 6vw, 76px); height: clamp(52px, 6vw, 76px); border-radius: 50%; background: var(--gold-400); color: var(--navy-950);
  display: grid; place-items: center; font-family: var(--serif); font-weight: 700; font-size: clamp(1.7rem, 3vw, 2.5rem);
  box-shadow: 0 0 0 6px var(--gold-100); flex: none; }
.tagline { font-family: var(--serif); font-weight: 400; font-size: clamp(1.15rem, 1.8vw, 1.5rem); color: var(--gold-800); margin: 4px 0 22px; max-width: 34ch; }
.hero h1 { font-size: clamp(1.7rem, 2.4vw, 2.3rem) !important; max-width: none !important; margin-bottom: 12px !important; }
.hero h1 .ln > span { white-space: nowrap; }
@media (max-width: 640px) { .hero h1 .ln > span { white-space: normal; } }
@keyframes wm-in { from { opacity: 0; transform: translateY(40%); } to { opacity: 1; transform: none; } }
@keyframes seal-in { from { opacity: 0; transform: scale(.6); } to { opacity: 1; transform: none; } }
html:not(.m-entered) .wm-word > span { animation: wm-in 650ms var(--ease-enter) both; }
html:not(.m-entered) .wm-word > span:nth-child(2) { animation-delay: 70ms; }
html:not(.m-entered) .wm-word > span:nth-child(3) { animation-delay: 140ms; }
html:not(.m-entered) .wm-word > span:nth-child(4) { animation-delay: 210ms; }
html:not(.m-entered) .wm-seal { animation: seal-in 600ms 320ms var(--ease-enter) both; }
html:not(.m-entered) .tagline { animation: rise 520ms 480ms var(--ease-enter) both; }
html:not(.m-entered) .hero-grid .ln > span { animation-delay: 560ms !important; }
html:not(.m-entered) .hero-grid .ln:nth-child(2) > span { animation-delay: 640ms !important; }
html:not(.m-entered) .hero-grid .enter-1 { animation-delay: 820ms !important; }
html:not(.m-entered) .hero-grid .enter-2 { animation-delay: 880ms !important; }
html:not(.m-entered) .hero-grid .enter-3 { animation-delay: 940ms !important; }

/* gambar hero duduk di panel kuning emas berpola titik halus */
.hero-visual { background-color: var(--gold-400) !important; border-radius: 28px !important; padding: clamp(20px, 3vw, 40px) !important;
  background-image: radial-gradient(rgba(14,31,59,.13) 1.2px, transparent 1.4px) !important; background-size: 18px 18px !important; }
.hero-visual .frame { box-shadow: 0 30px 60px -30px rgba(14,31,59,.55), 0 1px 2px rgba(14,31,59,.1); }

/* ---------- beranda: topik (kartu yang langsung bertanya) ---------- */
.topics-sec { background: var(--gold-100) !important; border-radius: 28px !important; padding: clamp(24px, 4vw, 52px) !important; margin-top: 24px !important; gap: 0 !important; }
.topics-sec h2 { font-family: var(--serif); font-weight: 400; color: var(--navy-950); font-size: clamp(1.8rem, 3vw, 2.6rem); line-height: 1.12; margin: 0 0 10px; }
.topics-sec h2 .muted { color: var(--gold-800); }
.topics-sec p.sub { color: #5A4A1E; margin: 0 0 24px; max-width: 60ch; }
.topics { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px !important; }
.topics > * { min-width: 0 !important; }
.topics button.topic { display: flex !important; flex-direction: column !important; align-items: flex-start !important; justify-content: flex-start !important;
  text-align: left !important; white-space: normal; background: var(--white) !important;
  border: 1px solid #EAD9A8 !important; border-radius: 18px !important; padding: 18px 18px 44px 18px !important; min-height: 150px !important; height: 100%;
  color: var(--navy-950) !important; font-weight: 600 !important; font-size: 1rem !important; line-height: 1.4; position: relative; }
.topics button.topic::before { display: block; margin-bottom: 10px; font-size: .78rem; font-weight: 700; color: var(--gold-800);
  background: var(--gold-100); border-radius: 999px; padding: 3px 10px; width: max-content; }
.topics button.topic::after { content: "Tanyakan"; position: absolute; right: 18px; bottom: 16px; font-size: .82rem; font-weight: 600; color: var(--navy-600); }
.t-kerja::before { content: "Ketenagakerjaan"; } .t-tanah::before { content: "Pertanahan"; } .t-keluarga::before { content: "Keluarga"; }
.t-konsumen::before { content: "Konsumen"; } .t-pidana::before { content: "Pidana"; } .t-utang::before { content: "Utang piutang"; }
@media (hover: hover) and (pointer: fine) {
  .topics button.topic:hover { transform: translateY(-2px) !important; border-color: var(--gold-400) !important; box-shadow: 0 6px 24px rgba(14,31,59,.10) !important; background: var(--white) !important; }
}

/* ---------- beranda: penutup di pita kuning emas ---------- */
.close-band { background: var(--gold-400) !important; border-radius: 28px !important; padding: clamp(28px, 4vw, 56px) !important; margin-top: 24px !important; gap: 0 !important; }
.close-band .close-sec { padding: 0; }
.close-band .close-sec h2 { color: var(--navy-950); max-width: 26ch; }
.close-band .close-sec h2 .muted { color: rgba(14,31,59,.62); }
.close-band .close-sec p.sub { color: #2B2510; }
.close-band .close-actions { margin-top: 22px !important; }

@media (max-width: 1100px) { .topics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .topics { grid-template-columns: 1fr; } .topics button.topic { min-height: 0 !important; } }

@media (prefers-reduced-motion: reduce) {
  .gradio-container *, .gradio-container *::before, .gradio-container *::after { animation: none !important; transition: none !important; }
  .workspace button:hover, .workspace button:active, .ref a:hover { transform: none !important; }
}
@media (max-width: 1100px) {
  .hero-grid { flex-wrap: wrap !important; min-height: auto; padding: 8px 0 24px; }
  .hero-grid > * { flex: 1 1 100% !important; }
  .hero h1 { max-width: 18ch; }
}
@media (max-width: 640px) {
  /* ponsel: loop dan denyut diam di gambar jadi; diagram diganti daftar */
  .frame .pause, .flow-band .pause { display: none; }
  .frame.live *, .flow-band.live .pulse { animation: none !important; }
  .frame.live .st-run { opacity: 0; } .frame.live .frefs li { opacity: 1; }
  .dg { display: none; } .flow-list { display: block; }
  .chips button.chip, .sidenav button.nav-btn { min-height: 44px !important; }
  /* satu kolom di ponsel: tombol selebar layar */
  .askbar button.send, .hero-actions button, .close-actions button { width: 100% !important; }
  .hero-actions > *, .close-actions > * { flex: 1 1 100% !important; }
}
@media (max-width: 1100px) { .steps { grid-template-columns: 1fr; border-top: 0; } .steps li, .steps li + li { padding: 14px 0; border-left: 0; border-top: 1px solid var(--line); } }
@media (max-width: 960px) {
  .ask-layout { flex-wrap: wrap !important; }
  .ask-layout > * { flex: 1 1 100% !important; }
  .refs-col { position: static; }
  .band { grid-template-columns: 1fr; }
  .split { grid-template-columns: 1fr; }
  .split > div + div { padding-left: 0; border-left: 0; border-top: 1px solid var(--line); }
}
@media (max-width: 820px) {
  .topbar { flex-wrap: wrap !important; padding: 8px 12px !important; row-gap: 4px !important; }
  .topbar .brand-cell { flex: 1 1 100% !important; }
  .topnav { flex: 1 1 100% !important; justify-content: flex-start !important; flex-wrap: wrap !important; }
  .topbar button.nav-btn { padding: 8px 10px !important; font-size: .9rem !important; }
  .topbar button.nav-btn::after { left: 10px; right: 10px; }
  .workspace { padding: 20px 16px 32px !important; }
  .askbar { flex-wrap: wrap !important; padding: 10px !important; }
  .askbar > * { flex: 1 1 100% !important; }
  .askbar button.send { width: 100%; }
  .facts > div + div { padding-left: 0; border-left: 0; }
  .greet { margin-top: 2vh; }
  .ask-top { flex-wrap: wrap !important; }
  /* sidebar chat jadi panel geser di atas konten; tertutup secara bawaan */
  .chat-side { position: fixed !important; top: 0; left: 0; bottom: 0; max-height: none; width: min(86vw, 320px) !important; z-index: 60;
    border-radius: 0 18px 18px 0 !important; transform: translateX(-105%); transition: transform 240ms var(--ease-ui); display: flex !important; }
  .chat-shell.side-open .chat-side { transform: none; box-shadow: 0 0 0 100vmax rgba(14,31,59,.35); }
  .chat-shell.side-closed .chat-side { display: flex !important; }
}
"""

# Paksa tema terang agar palet konsisten (palet ini dirancang untuk latar putih).
FORCE_LIGHT_JS = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); return; }
  const root = document.documentElement;
  // entrance hero hanya sekali per kunjungan
  setTimeout(() => root.classList.add('m-entered'), 1900);
  // loop visual (app-frame, diagram) hanya jalan bila layar >= 640px dan pengguna tidak memilih kurangi gerakan
  const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const wire = () => document.querySelectorAll('[data-loop]').forEach(el => {
    if (el.dataset.wired) return; el.dataset.wired = '1';
    if (!calm && window.innerWidth >= 640) el.classList.add('live');
  });
  wire(); new MutationObserver(wire).observe(document.body, { childList: true, subtree: true });
  // tombol jeda (loop > 5 detik wajib bisa dihentikan)
  document.addEventListener('click', e => {
    const b = e.target.closest('.pause'); if (!b) return;
    const box = b.closest('[data-loop]'); const p = box.classList.toggle('paused');
    b.setAttribute('aria-pressed', p ? 'true' : 'false'); b.setAttribute('aria-label', p ? 'Putar animasi' : 'Jeda animasi');
  });
}
"""

# Nama dan tagline (usulan; ubah di sini bila kelompok memilih nama lain)
BRAND_NAME = "Asas"
TAGLINE = "Tanya hukum sehari-hari, langsung dengan sumbernya."

BRAND_HTML = f"""
<div class="brand"><div class="brand-mark" aria-hidden="true">§</div>
<div><p class="brand-name">{BRAND_NAME}</p><p class="brand-sub">Chatbot informasi hukum</p></div></div>
"""

SIDE_HEAD_HTML = f"""
<div class="side-head"><div class="brand"><div class="brand-mark" aria-hidden="true">§</div>
<div><p class="brand-name">{BRAND_NAME}</p><p class="brand-sub">Riwayat percakapan</p></div></div></div>
"""

SIDE_NOTE = """<div class="side-note"><strong>Panduan awal, bukan patokan utama.</strong> Jawaban disusun dari artikel
Klinik Hukumonline untuk membantu Anda memahami persoalan. Untuk keputusan penting, sebaiknya konsultasikan
dengan advokat atau lembaga bantuan hukum.</div>"""

PAUSE_BTN = ('<button class="pause" type="button" aria-pressed="false" aria-label="Jeda animasi">'
             '<svg class="i-pause" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4h3v16H7zM14 4h3v16h-3z" fill="currentColor"/></svg>'
             '<svg class="i-play" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4l13 8-13 8z" fill="currentColor"/></svg></button>')

# Layar 1: teks hero (judul menyebut kategori + pengguna; satu fakta keras di sub-judul)
HERO_HTML = f"""
<section class="hero" aria-labelledby="hero-h">
  <div class="wordmark" role="img" aria-label="{BRAND_NAME}"><span class="wm-word" aria-hidden="true"><span>A</span><span>s</span><span>a</span><span>s</span></span><span class="wm-seal" aria-hidden="true">§</span></div>
  <p class="tagline">{TAGLINE}</p>
  <h1 id="hero-h"><span class="ln"><span>Chatbot informasi hukum</span></span><span class="ln"><span>untuk pertanyaan sehari-hari</span></span></h1>
  <p class="lead enter enter-1">Tulis masalah Anda dengan bahasa biasa. Chatbot mencari di <b>7.172 artikel</b> tanya-jawab
  Klinik Hukumonline, merangkum tiga yang paling cocok, dan menautkan sumbernya.</p>
</section>
"""

# Gambar hero: tampilan asli halaman Tanya (judul rujukan = hasil retrieval nyata untuk pertanyaan ini)
HERO_FRAME = f"""
<div class="frame" data-loop role="img" aria-label="Contoh halaman Tanya hukum: pertanyaan soal gaji lembur dan tiga artikel rujukan yang ditemukan">
  <div class="frame-bar"><span class="mode-pill">Tanya hukum</span><span class="who">Contoh pencarian</span></div>
  <div class="frame-body">
    <p class="fq">Gaji lembur tidak dibayar, saya harus lapor ke mana?</p>
    <div class="fstack">
      <p class="fstatus st-run"><span class="spin"></span>Mencari di 7.172 artikel</p>
      <p class="fstatus st-done"><svg class="ok" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l4.5 4.5L19 7"/></svg>3 artikel rujukan ditemukan</p>
    </div>
    <ol class="frefs">
      <li><span class="p">Rujukan 1</span>Langkah Hukum Jika Gaji Tak Dibayar Pengusaha</li>
      <li><span class="p">Rujukan 2</span>Gaji Tidak Tepat Waktu</li>
      <li><span class="p">Rujukan 3</span>Karyawan Lupa Isi Form Lembur, Wajibkah Upah Lembur Dibayarkan?</li>
    </ol>
  </div>
  {PAUSE_BTN}
</div>
"""


def _node(x, y, w, title, meta, cls=""):
    return (f'<g class="node {cls}"><rect x="{x}" y="{y}" width="{w}" height="64" rx="14"/>'
            f'<text class="t" x="{x + 16}" y="{y + 27}">{title}</text><text class="m" x="{x + 16}" y="{y + 47}">{meta}</text></g>')


def _edge(d, delay):
    return f'<path class="edge" d="{d}"/><path class="pulse" pathLength="100" style="--d:{delay}" d="{d}"/>'


# Layar 2: diagram alur RAG (satu input, inti, dua jalur paralel), denyut emas satu siklus 3,2 dtk
_DG_NODES = [
    _node(0, 168, 150, "Pertanyaan", "bahasa sehari-hari"),
    _node(196, 168, 170, "Penulis ulang", "3–8 kata kunci"),
    _node(414, 70, 176, "Makna (e5)", "30 kandidat"),
    _node(414, 266, 176, "Kata kunci (BM25)", "30 kandidat"),
    _node(640, 168, 170, "Gabung peringkat", "RRF, 3 teratas", "core"),
    _node(858, 168, 150, "Model bahasa", "via OpenRouter"),
    _node(1050, 168, 150, "Jawaban", "+ 3 tautan sumber", "end"),
]
_DG_EDGES = [
    _edge("M150 200 H196", 0), _edge("M366 200 C390 200 390 102 414 102", .3), _edge("M366 200 C390 200 390 298 414 298", .3),
    _edge("M590 102 C615 102 615 200 640 200", .6), _edge("M590 298 C615 298 615 200 640 200", .6),
    _edge("M810 200 H858", .9), _edge("M1008 200 H1050", 1.2),
]
DIAGRAM_SVG = (
    '<svg class="dg" viewBox="0 0 1200 400" role="img" aria-label="Alur jawaban: pertanyaan ditulis ulang, dicari lewat makna dan kata kunci, '
    'peringkat digabung, tiga artikel teratas diberikan ke model bahasa, lalu jawaban tampil dengan tautan sumber">'
    '<defs><linearGradient id="comet" x1="0" x2="1"><stop offset="0" stop-color="#E2B33C" stop-opacity="0"/>'
    '<stop offset="1" stop-color="#F6D77E"/></linearGradient></defs>'
    + "".join(_DG_EDGES) + "".join(_DG_NODES) + "</svg>"
)

HOME_BELOW = f"""
<section class="flow-band" data-loop aria-labelledby="flow-h">
  <h2 id="flow-h">Jawaban disusun dari tiga artikel rujukan. <span class="muted">Bukan dari ingatan model.</span></h2>
  <p class="sub">Pertanyaan Anda dicari dengan dua cara sekaligus, lewat makna dan lewat kata kunci. Tiga artikel teratas
  lalu diberikan ke model bahasa sebagai satu-satunya bahan jawaban.</p>
  {DIAGRAM_SVG}
  <ol class="flow-list">
    <li><div>Pertanyaan<span>ditulis dengan bahasa sehari-hari</span></div></li>
    <li><div>Penulis ulang<span>diubah jadi 3–8 kata kunci hukum</span></div></li>
    <li><div>Makna (e5) dan kata kunci (BM25)<span>masing-masing 30 kandidat</span></div></li>
    <li><div>Gabung peringkat<span>RRF, ambil 3 teratas</span></div></li>
    <li><div>Model bahasa<span>menjawab hanya dari 3 artikel itu</span></div></li>
    <li><div>Jawaban<span>tampil dengan 3 tautan sumber</span></div></li>
  </ol>
  {PAUSE_BTN}
</section>
"""

# Layar 3: topik yang sering ditanyakan (kartu = tombol yang langsung bertanya)
TOPICS = [
    ("t-kerja", "Kena PHK mendadak tanpa pesangon, apa yang bisa dilakukan?"),
    ("t-tanah", "Bagaimana cara aman membeli tanah girik?"),
    ("t-keluarga", "Suami minta cerai, apa yang perlu saya siapkan?"),
    ("t-konsumen", "Belanja online, barang tidak dikirim. Bisa lapor ke mana?"),
    ("t-pidana", "Dipukul orang di jalan, bisa lapor polisi?"),
    ("t-utang", "Teman pinjam uang lalu tidak mau bayar, bisa digugat?"),
]
TOPICS_HEAD = """<div><h2>Topik yang sering ditanyakan. <span class="muted">Pilih satu untuk mencoba.</span></h2>
<p class="sub">Contoh pertanyaan dari enam bidang hukum yang banyak dibahas di Klinik Hukumonline. Kartu langsung mengirim pertanyaannya.</p></div>"""

CLOSE_HTML = """
<section class="close-sec" aria-labelledby="close-h">
  <h2 id="close-h">Informasi hukum umum, bukan nasihat hukum. <span class="muted">Data sampai Agustus 2024.</span></h2>
  <p class="sub">Aturan sesudahnya, termasuk KUHP baru yang berlaku 2 Januari 2026, belum tentu tercakup. Untuk sengketa,
  ancaman pidana, atau tenggat waktu, hubungi advokat atau lembaga bantuan hukum.</p>
</section>
"""

HOME_FOOT = "<p class='foot'>Isi artikel milik Klinik Hukumonline, dari dataset Hugging Face biznetgio/indonesia-law-qa-embeddings. UAS NLP Kelompok B.</p>"

GREET_HTML = """<div class="greet"><h1>Tanya hukum.</h1><p>Ceritakan keadaannya dengan bahasa sehari-hari.</p></div>"""

ASK_NOTE = ("<p class='ask-note'>Jawaban disusun otomatis sebagai informasi hukum umum dan panduan awal, bukan patokan utama. "
            "Hindari menulis data pribadi seperti nama lengkap atau nomor identitas.</p>")

CARA_KERJA_HTML = """
<div class="page-head"><h1>Cara kerja</h1>
<p>Lima langkah dari pertanyaan Anda sampai jawaban dan sumbernya. Teknik ini disebut Retrieval-Augmented Generation (RAG).</p></div>
<ol class="steps">
  <li><div class="n">1</div><h3>Pertanyaan masuk</h3><p>Anda menulis dengan bahasa sehari-hari. Sistem dapat mengubahnya dulu menjadi istilah hukum yang lebih baku agar pencarian lebih tepat.</p></li>
  <li><div class="n">2</div><h3>Mencari artikel</h3><p>Pertanyaan dicocokkan dengan 7.172 artikel lewat kemiripan makna (multilingual-e5-large) dan kecocokan kata kunci (BM25). Kedua peringkat digabung dengan Reciprocal Rank Fusion.</p></li>
  <li><div class="n">3</div><h3>Memilih rujukan</h3><p>Tiga artikel teratas diambil: judul, ringkasan, dan sebagian ulasannya.</p></li>
  <li><div class="n">4</div><h3>Menyusun jawaban</h3><p>Artikel dan pertanyaan diberikan ke model bahasa (LLM) dengan instruksi menjawab hanya dari artikel dan mengaku bila informasinya tidak ada.</p></li>
  <li><div class="n">5</div><h3>Jawaban dan sumber</h3><p>Jawaban ditampilkan bersama tautan artikel rujukan supaya bisa diperiksa sendiri.</p></li>
</ol>
<p class="callout"><strong>Catatan:</strong> "jawab hanya dari artikel" adalah instruksi kepada model, bukan jaminan.
Seberapa sering model tetap mengarang sedang diuji; hasilnya akan dicantumkan di halaman Tentang &amp; evaluasi.</p>
"""

TENTANG_HTML = """
<div class="page-head"><h1>Tentang &amp; evaluasi</h1>
<p>Dari mana jawaban berasal, apa batasnya, dan bagaimana sistem ini diuji.</p></div>
<div class="facts">
  <div><div class="v">7.172</div><div class="l">artikel tanya-jawab Klinik Hukumonline</div></div>
  <div><div class="v">6 Agustus 2024</div><div class="l">terakhir diperbarui (riwayat dataset di Hugging Face)</div></div>
  <div><div class="v">38 / 129 / 829</div><div class="l">median jumlah kata pertanyaan, ringkasan, dan ulasan lengkap (dihitung tim)</div></div>
</div>
<p>Sumber: dataset Hugging Face <code>biznetgio/indonesia-law-qa-embeddings</code>.
Lisensi apache-2.0 menurut pengunggah; isi artikel milik Klinik Hukumonline.</p>
<p class="callout"><strong>Keterbatasan:</strong> sebagian artikel ditulis sebelum KUHP baru (UU 1/2023) berlaku pada
2 Januari 2026, jadi jawaban pidana bisa usang. Ada artikel berbahasa Inggris. Satu artikel menjawab satu pertanyaan
tertentu, sehingga belum tentu cocok dengan keadaan pribadi Anda.</p>
"""

EVAL_MD = """
## Evaluasi <span class="status">Progress 2, sedang berjalan</span>

| Pengujian | Ukuran | Status |
|:--|:--|:--|
| Ketahanan terhadap parafrase (40 pertanyaan) | Recall@k, Precision@k, MRR | sedang berjalan |
| Pertanyaan pendek (3–5 kata) vs panjang | Hit@k / MRR per kelompok | sedang berjalan |
| Metode pencarian: BM25, FAISS e5, hybrid | Recall, Precision | sedang berjalan |
| Kualitas jawaban beberapa LLM (Gemma, Qwen, pembanding) | BLEU, ROUGE, BERTScore + penilaian manual 1–5 | sedang berjalan |
| Pertanyaan baru di luar dataset | menolak vs mengarang (dibaca manual) | sedang berjalan |

BLEU, ROUGE, dan BERTScore hanya mengukur **kemiripan teks** dengan jawaban Klinik Hukumonline,
bukan kebenaran hukum. Hasil akan ditambahkan di sini setelah pengujian selesai.

### Hasil awal (Progress 1)
| Metode pencarian | Hit@1 | Hit@3 | MRR |
|:--|--:|--:|--:|
| Kemiripan makna (FAISS e5) | 65,0% | 90,0% | 0,772 |
| Kata kunci (BM25) | 50,0% | 80,0% | 0,669 |
| Gabungan (RRF) | 72,5% | 90,0% | 0,815 |

Keterbatasan: hanya 40 pertanyaan (1 pertanyaan = 2,5 poin), dan "benar" dinilai dari kata kunci di judul
artikel, bukan dari ID artikel yang tepat. Angka ini indikasi awal, bukan akurasi final.
"""


PENDING_MSG = "Sedang mencari artikel dan menyusun jawaban…"
MAX_SAVED_CHATS = 30


def _with_sources(answer_md: str, rows) -> str:
    """Sisipkan daftar artikel rujukan (tautan yang bisa diklik) ke dalam jawaban, sebelum catatan 'Perlu diingat'.
    Tautan diambil dari hasil retrieval (rows dari _detail: [no, judul, skor, ..., url]), bukan dari teks model."""
    links = []
    for r in rows or []:
        title, url = str(r[1]), str(r[-1])
        if not url.startswith(("https://", "http://")):
            continue
        safe = title.replace("[", "(").replace("]", ")")
        links.append(f"{r[0]}. [{safe}]({url})")
    if not links:
        return answer_md
    block = "\n\n**Artikel rujukan** (klik untuk membaca artikel aslinya di Klinik Hukumonline):\n\n" + "\n".join(links)
    marker = "\n\n<div class='note'"
    i = answer_md.find(marker)
    return answer_md[:i] + block + answer_md[i:] if i != -1 else answer_md + block


def _msg_text(msg) -> str:
    """Gradio 6 bisa mengembalikan content sebagai string atau daftar potongan {"type","text"}."""
    c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def _msg_role(msg):
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)


def _summary_title(msgs, first_q: str) -> str:
    """Judul riwayat = rangkuman singkat percakapan oleh LLM (3–7 kata); cadangan: potongan pertanyaan pertama."""
    fallback = first_q[:60] + ("…" if len(first_q) > 60 else "")
    if not rag_core.get_api_key():
        return fallback
    answer_snip = next((m["content"] for m in msgs if m["role"] == "assistant"), "")[:600]
    prompt = [
        {"role": "system", "content": "Buat judul singkat 3 sampai 7 kata dalam bahasa Indonesia yang merangkum topik "
                                      "percakapan hukum berikut. Tulis judulnya saja, tanpa tanda kutip, tanpa titik."},
        {"role": "user", "content": f"Pertanyaan: {first_q}\n\nAwal jawaban: {answer_snip}"},
    ]
    t = rag_core.call_llm(prompt, temperature=0.2, max_tokens=30)
    t = t.strip().strip('"\'“”*#').splitlines()[0].strip().rstrip(".") if t else ""
    if not t or t.startswith("[Error") or len(t) > 70:
        return fallback
    return t


def _history_choices(saved):
    """Pilihan menu riwayat: (label tampil, id percakapan), terbaru di atas."""
    return [(f"{c['title']}  ·  {c['time']}", c["id"]) for c in (saved or [])]


def save_chat(history, cur_id, saved):
    """Simpan/perbarui percakapan aktif ke riwayat di browser (localStorage, hanya di perangkat ini)."""
    saved = list(saved or [])
    msgs = [{"role": _msg_role(m), "content": _msg_text(m)} for m in (history or [])
            if _msg_role(m) in ("user", "assistant") and PENDING_MSG not in _msg_text(m)]
    if not any(m["role"] == "user" for m in msgs):
        return saved, cur_id, gr.update()
    first_q = next(m["content"] for m in msgs if m["role"] == "user")
    old = next((c for c in saved if cur_id and c.get("id") == cur_id), None)
    cur_id = cur_id or str(time.time_ns())
    title = old["title"] if old else _summary_title(msgs, first_q)   # judul dibuat sekali per percakapan
    entry = {"id": cur_id, "title": title, "time": time.strftime("%d/%m %H:%M"), "messages": msgs}
    saved = [entry] + [c for c in saved if c.get("id") != cur_id]
    saved = saved[:MAX_SAVED_CHATS]
    return saved, cur_id, gr.update(choices=_history_choices(saved), value=None)


def open_chat(chat_id, saved):
    """Buka percakapan lama dari menu riwayat."""
    for c in saved or []:
        if c.get("id") == chat_id:
            return c["messages"], chat_id, SOURCES_EMPTY
    return gr.update(), gr.update(), gr.update()


def add_user_turn(question, history):
    """Langkah 1: tampilkan pertanyaan langsung di chat (dengan penanda proses) dan kosongkan kotak tanya."""
    history = list(history or [])
    question = (question or "").strip()
    if not question:
        gr.Warning("Tulis pertanyaan dulu, lalu tekan Kirim.")
        return history, "", ""
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": PENDING_MSG}]
    return history, "", question


def answer_in_chat(question, model_choice, method_label, use_rewriter, top_k, prompt_label, history):
    """Langkah 2: jalankan pipeline yang sama (answer()) lalu ganti penanda proses dengan jawaban."""
    history = list(history or [])
    if not question:
        return history, gr.update(), gr.update(), gr.update()
    try:
        result = answer(question, model_choice, method_label, use_rewriter, top_k, prompt_label)
        answer_md = result[0].get("value", "") if isinstance(result[0], dict) else str(result[0])
        extra = result[1:]
        answer_md = _with_sources(answer_md, result[3])
    except Exception as e:  # gr.Error dari answer() tampil di chat, bukan hilang
        answer_md = _label("Jawaban belum bisa dibuat", "warn") + "\n\n" + html.escape(getattr(e, "message", str(e)))
        extra = (gr.update(), gr.update(), gr.update())
    last = history[-1] if history else None
    if last is not None and _msg_role(last) == "assistant" and PENDING_MSG in _msg_text(last):
        history[-1] = {"role": "assistant", "content": answer_md}
    else:
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer_md}]
    return (history, *extra)


def _filter_choices(query, saved):
    """Cari riwayat: cocokkan judul dan isi percakapan (huruf kecil)."""
    q = (query or "").strip().lower()
    items = saved or []
    if q:
        items = [c for c in items if q in c.get("title", "").lower()
                 or any(q in m.get("content", "").lower() for m in c.get("messages", []))]
    return items


def _hist_update(saved, query="", value=None):
    items = _filter_choices(query, saved)
    if not saved:
        info = "<p class='side-empty'>Belum ada riwayat. Percakapan tersimpan otomatis setelah jawaban pertama.</p>"
    elif not items:
        info = "<p class='side-empty'>Tidak ada riwayat yang cocok.</p>"
    else:
        info = ""
    return gr.update(choices=_history_choices(items), value=value), info


# Buka/tutup sidebar riwayat: desktop menyembunyikan kolom, ponsel menggeser panel di atas konten
TOGGLE_SIDE_JS = """
() => {
  const s = document.querySelector('.chat-shell'); if (!s) return;
  if (window.innerWidth <= 820) s.classList.toggle('side-open'); else s.classList.toggle('side-closed');
}
"""
CLOSE_SIDE_MOBILE_JS = """
() => { const s = document.querySelector('.chat-shell'); if (s && window.innerWidth <= 820) s.classList.remove('side-open'); }
"""


def build_ui() -> gr.Blocks:
    default_model = rag_core.get_default_model() or MODEL_CHOICES[0]
    models = MODEL_CHOICES if default_model in MODEL_CHOICES else [default_model] + MODEL_CHOICES
    has_key = bool(rag_core.get_api_key())
    mode_html = ("<span class='mode-pill'>Tanya hukum <span class='tag'>Jawaban otomatis aktif</span></span>" if has_key
                 else "<span class='mode-pill'>Tanya hukum <span class='tag'>Mode cari artikel</span></span>")

    with gr.Blocks(title=f"{BRAND_NAME}: chatbot informasi hukum", fill_width=True) as demo:
        with gr.Column(elem_classes="shell"):
            # ---------------- bar atas (navigasi semua halaman)
            with gr.Row(elem_classes="topbar", equal_height=False):
                with gr.Column(scale=1, elem_classes="brand-cell"):
                    gr.HTML(BRAND_HTML, padding=False)
                with gr.Row(elem_classes="topnav"):
                    nav = {
                        "beranda": gr.Button(NAV_LABELS["beranda"], variant="primary", elem_classes="nav-btn", scale=0),
                        "tanya": gr.Button(NAV_LABELS["tanya"], variant="secondary", elem_classes="nav-btn", scale=0),
                        "cara": gr.Button(NAV_LABELS["cara"], variant="secondary", elem_classes="nav-btn", scale=0),
                        "tentang": gr.Button(NAV_LABELS["tentang"], variant="secondary", elem_classes="nav-btn", scale=0),
                    }

            # ---------------- area kerja
            with gr.Column(elem_classes="workspace"):
                with gr.Tabs(elem_classes="main-tabs") as tabs:
                    # ======== Beranda
                    with gr.Tab("Beranda", id="beranda"):
                        with gr.Row(elem_classes="hero-grid", equal_height=False):
                            with gr.Column(scale=11, elem_classes="hero-copy"):
                                gr.HTML(HERO_HTML, padding=False)
                                with gr.Row(elem_classes=["askbar", "hero-ask", "enter", "enter-2"]):
                                    hq = gr.Textbox(show_label=False, container=False, lines=1, max_lines=6, scale=1,
                                                    placeholder="Contoh: gaji lembur saya tidak dibayar, lapor ke mana?")
                                    hbtn = gr.Button("Ajukan pertanyaan", variant="primary", elem_classes="send", scale=0)
                                with gr.Row(elem_classes=["hero-actions", "enter", "enter-3"]):
                                    hhow = gr.Button("Lihat cara kerja", variant="secondary", scale=0)
                            with gr.Column(scale=9, elem_classes="hero-visual"):
                                gr.HTML(HERO_FRAME, padding=False)
                        gr.HTML(HOME_BELOW, padding=False)
                        with gr.Column(elem_classes="topics-sec"):
                            gr.HTML(TOPICS_HEAD, padding=False)
                            with gr.Row(elem_classes="topics"):
                                topic_btns = [gr.Button(text, elem_classes=["topic", cls]) for cls, text in TOPICS]
                        with gr.Column(elem_classes="close-band"):
                            gr.HTML(CLOSE_HTML, padding=False)
                            with gr.Row(elem_classes="close-actions"):
                                hask2 = gr.Button("Ajukan pertanyaan", variant="primary", scale=0)
                        gr.HTML(HOME_FOOT, padding=False)

                    # ======== Tanya hukum: sidebar riwayat + area chat
                    with gr.Tab("Tanya hukum", id="tanya"):
                        with gr.Row(elem_classes="chat-shell", equal_height=False):
                            with gr.Column(scale=0, min_width=280, elem_classes="chat-side"):
                                with gr.Row(elem_classes="side-top"):
                                    gr.HTML(SIDE_HEAD_HTML, padding=False)
                                    side_close = gr.Button("Tutup panel riwayat", elem_classes="side-close", scale=0)
                                new_chat = gr.Button("Percakapan baru", elem_classes="side-new")
                                search = gr.Textbox(show_label=False, container=False, lines=1, max_lines=1,
                                                    placeholder="Cari riwayat…", elem_classes="side-search")
                                gr.HTML("<p class='side-label'>Riwayat</p>", padding=False)
                                hist = gr.Radio(choices=[], value=None, show_label=False, container=False,
                                                label="Riwayat percakapan", elem_classes="side-hist")
                                hist_info = gr.HTML("", padding=False)
                                del_hist = gr.Button("Hapus semua riwayat", elem_classes="side-del")
                                gr.HTML(SIDE_NOTE, padding=False)

                            with gr.Column(scale=1, elem_classes="chat-main"):
                                with gr.Row(elem_classes="ask-top"):
                                    side_toggle = gr.Button("Riwayat", variant="secondary", size="sm",
                                                            elem_classes="side-toggle", scale=0)
                                    gr.HTML(mode_html, padding=False)
                                gr.HTML(GREET_HTML, padding=False)
                                chat = gr.Chatbot(height=520, layout="bubble", show_label=False,
                                                  elem_classes="chat-history", placeholder="Percakapan Anda akan muncul di sini.")
                                with gr.Row(elem_classes="askbar"):
                                    q = gr.Textbox(show_label=False, container=False, lines=1, max_lines=6, scale=1,
                                                   placeholder="Tulis pertanyaan Anda di sini…")
                                    btn = gr.Button("Kirim pertanyaan", variant="primary", elem_classes="send", scale=0)
                                with gr.Row(elem_classes="chips"):
                                    chips = [gr.Button(e, size="sm", elem_classes="chip") for e in EXAMPLES]
                                gr.HTML(ASK_NOTE, padding=False)
                                with gr.Accordion("Artikel rujukan untuk pertanyaan terakhir", open=False,
                                                  elem_classes="sources-accordion"):
                                    src = gr.HTML(SOURCES_EMPTY, padding=False)

                                with gr.Accordion("Mode eksperimen: untuk pengujian, bukan untuk pengguna umum",
                                                  open=False, elem_classes="exp-acc"):
                                    gr.Markdown("Pengaturan untuk membandingkan komponen sistem saat demo dan evaluasi. "
                                                "Nilai default adalah konfigurasi yang dipakai tim.")
                                    with gr.Row():
                                        model = gr.Dropdown(models, value=default_model, allow_custom_value=True,
                                                            label="Model bahasa (LLM)",
                                                            info="Model yang menyusun jawaban (ID OpenRouter).")
                                        method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="Cara mencari artikel")
                                    with gr.Row():
                                        rewriter = gr.Checkbox(value=True, label="Ubah pertanyaan ke istilah hukum sebelum mencari",
                                                               info="Query rewriter: memakai LLM, menambah waktu ±1–3 detik.")
                                        top_k = gr.Slider(1, 10, value=3, step=1, label="Jumlah artikel rujukan",
                                                          info="Lebih banyak = konteks lebih luas tapi lebih panjang.")
                                    prompt = gr.Radio(list(PROMPTS), value=list(PROMPTS)[0], label="Instruksi ke model")
                                    gr.Markdown("#### Detail hasil terakhir")
                                    info = gr.Markdown()
                                    table = gr.Dataframe(headers=["#", "Judul", "Skor", "Peringkat dense", "Peringkat BM25", "URL"],
                                                         interactive=False, wrap=True)

                    with gr.Tab("Cara kerja", id="cara"):
                        gr.HTML(CARA_KERJA_HTML, padding=False)

                    with gr.Tab("Tentang & evaluasi", id="tentang"):
                        gr.HTML(TENTANG_HTML, padding=False)
                        gr.Markdown(EVAL_MD)

        # ---------------- navigasi
        nav_btns = [nav[p] for p in PAGES]

        def go(target):
            return [gr.Tabs(selected=target)] + [
                gr.update(variant="primary" if p == target else "secondary") for p in PAGES]

        for p in PAGES:
            nav[p].click(lambda t=p: go(t), None, [tabs] + nav_btns)

        # ---------------- tanya (pipeline sama: answer())
        pending = gr.State("")
        cur_id = gr.State("")                                   # id percakapan yang sedang aktif
        saved = gr.BrowserState([], storage_key="chatbot_hukum_riwayat",
                                secret=os.getenv("BROWSER_STATE_SECRET", "chatbot-hukum-uas-nlp-riwayat"))  # kunci tetap: riwayat tetap terbaca setelah app di-restart
        inputs = [pending, model, method, rewriter, top_k, prompt, chat]
        outputs = [chat, src, info, table]
        busy = lambda: gr.update(interactive=False, value="Mencari jawaban…")
        idle = lambda: gr.update(interactive=True, value="Kirim pertanyaan")

        def save_and_list(history, cid, s, query):
            s, cid, _ = save_chat(history, cid, s)
            upd, msg = _hist_update(s, query, value=cid or None)
            return s, cid, upd, msg

        def ask_chain(event):
            return (event.then(busy, None, btn)
                    .then(add_user_turn, [q, chat], [chat, q, pending], show_progress="hidden")
                    .then(answer_in_chat, inputs, outputs, show_progress="hidden")
                    .then(idle, None, btn)
                    .then(save_and_list, [chat, cur_id, saved, search], [saved, cur_id, hist, hist_info], show_progress="hidden"))

        ask_chain(btn.click(lambda: None, None, None))
        ask_chain(q.submit(lambda: None, None, None))
        for b, text in zip(chips, EXAMPLES):
            b.click(lambda t=text: t, None, q)

        # pertanyaan dari beranda (kotak tanya atau kartu topik): pindah ke halaman Tanya lalu jalankan pipeline yang sama
        def route(text):
            return [text] + go("tanya")

        ask_chain(hbtn.click(route, [hq], [q, tabs] + nav_btns))
        ask_chain(hq.submit(route, [hq], [q, tabs] + nav_btns))
        for b, (_, text) in zip(topic_btns, TOPICS):
            ask_chain(b.click(lambda t=text: route(t), None, [q, tabs] + nav_btns))
        hhow.click(lambda: go("cara"), None, [tabs] + nav_btns)
        hask2.click(lambda: go("tanya"), None, [tabs] + nav_btns)

        # sidebar riwayat
        side_toggle.click(None, None, None, js=TOGGLE_SIDE_JS)
        side_close.click(None, None, None, js=TOGGLE_SIDE_JS)

        def new_conversation(history, cid, s, query):
            s, _, _ = save_chat(history, cid, s)                 # simpan yang lama dulu
            upd, msg = _hist_update(s, query, value=None)
            return s, "", upd, msg, "", [], SOURCES_EMPTY, "", None

        (new_chat.click(new_conversation, [chat, cur_id, saved, search],
                        [saved, cur_id, hist, hist_info, q, chat, src, info, table], show_progress="hidden")
                 .then(None, None, None, js=CLOSE_SIDE_MOBILE_JS))
        def _search(qq, s, cid):
            upd, msg = _hist_update(s, qq, value=cid or None)
            return upd, msg

        search.change(_search, [search, saved, cur_id], [hist, hist_info], show_progress="hidden", trigger_mode="always_last")
        (hist.input(open_chat, [hist, saved], [chat, cur_id, src], show_progress="hidden")
             .then(None, None, None, js=CLOSE_SIDE_MOBILE_JS))

        def clear_all():
            upd, msg = _hist_update([], "")
            return [], upd, msg, "", [], "", SOURCES_EMPTY

        del_hist.click(clear_all, None, [saved, hist, hist_info, search, chat, cur_id, src])

        def _load_hist(s):
            print(f"[riwayat] dimuat saat halaman dibuka: {len(s or [])} percakapan", flush=True)
            return _hist_update(s, "")

        demo.load(_load_hist, [saved], [hist, hist_info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="buat link publik sementara (Gradio)")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", 7860)))
    args = ap.parse_args()

    print("Memuat data, indeks, dan model embedding (sekali di awal)…")
    res = rag_core.load_resources()
    print(f"Siap: {len(res.docs)} dokumen, device embedding = {res.device}. "
          + ("Kunci API terdeteksi." if rag_core.get_api_key() else "Kunci API BELUM di-set: mode cari artikel saja."))
    build_ui().queue(default_concurrency_limit=2).launch(
        share=args.share, server_port=args.port, css=CSS, js=FORCE_LIGHT_JS, head=FONTS_HEAD,
        theme=gr.themes.Default(font=["Public Sans", "system-ui", "sans-serif"]),
    )
