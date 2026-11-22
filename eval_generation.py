"""
eval_generation.py — Evaluasi answer generation (tugas 3b, Progress 2).

Alur (jalankan berurutan):
  1. python eval_generation.py prepare            # set uji + retrieval (sama untuk semua LLM)
  2. python eval_generation.py generate --yes     # panggil tiap LLM, simpan CSV per model
  3. python eval_generation.py score              # BLEU, ROUGE, BERTScore + tabel ringkasan
  4. python eval_generation.py manual             # lembar penilaian manual 1-5 (sampel)

Keputusan metodologi (dicatat juga di results/metodologi.json):
  - Set uji: 40 artikel acak, random.seed(42), atau --test-set CSV milik David (kolom 'id',
    opsional kolom pertanyaan parafrase via --query-col).
  - Retrieval: hybrid top-3, TANPA query rewriter (supaya konteks identik untuk semua LLM
    dan tidak bergantung pada satu LLM). Hasil retrieval disimpan sekali lalu dipakai ulang.
  - Prompt: gaya 'ketat' dari rag_core (sama untuk semua LLM), temperature 0.2.
  - Referensi: 'summarize' DAN 'answer' (yang sudah dibersihkan), dilaporkan terpisah.
  - Metrik: BLEU (sacreBLEU, kalimat & korpus), ROUGE-1/2/L F1 (tanpa stemmer, tokenizer
    regex \\w+ agar aman untuk bahasa Indonesia), BERTScore F1 (bert-base-multilingual-cased).
    Semuanya hanya pembanding TEKS dengan jawaban Klinik Hukumonline, BUKAN ukuran
    kebenaran hukum.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

import rag_core

RESULTS = Path(__file__).with_name("results")
SEED = 42
N_TEST = 40
TOP_K = 3

# Usulan model (ID dicek di https://openrouter.ai/api/v1/models pada 22 Nov 2026;
# nama model OpenRouter sering berubah, cek ulang sebelum menjalankan).
DEFAULT_MODELS = [
    "google/gemma-4-26b-a4b-it",  # Gemma (versi berbayar murah; versi :free sering 429)
    "qwen/qwen3.6-35b-a3b",       # Qwen
    "openai/gpt-6-luna",          # pembanding (Luna)
]

GEN_FIELDS = [
    "id", "model", "model_dijawab", "query", "retrieved_ids", "retrieved_titles",
    "gold_in_top3", "gold_rank", "status", "finish_reason", "answer", "latency_s",
    "prompt_tokens", "completion_tokens", "cost_usd", "error",
]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


# ------------------------------------------------------------------ 1. prepare
def cmd_prepare(args):
    RESULTS.mkdir(exist_ok=True)
    res = rag_core.load_resources()
    docs = res.docs
    id_to_idx = {d["id"]: i for i, d in enumerate(docs)}

    if args.test_set:
        ts = pd.read_csv(args.test_set)
        rows = []
        for _, r in ts.iterrows():
            idx = id_to_idx[type(docs[0]["id"])(r["id"])]
            q = str(r[args.query_col]) if args.query_col else docs[idx]["question"]
            rows.append((idx, q))
        source = f"file {args.test_set}" + (f", kolom {args.query_col}" if args.query_col else "")
    else:
        random.seed(SEED)
        idxs = random.sample(range(len(docs)), args.n)
        rows = [(i, docs[i]["question"]) for i in idxs]
        source = f"random.seed({SEED}), random.sample(range({len(docs)}), {args.n})"

    out = []
    for idx, query in rows:
        t0 = time.time()
        hits = rag_core.search(query, method="hybrid", top_k=TOP_K)
        ids = [h["doc_index"] for h in hits]
        out.append({
            "id": docs[idx]["id"],
            "doc_index": idx,
            "title": docs[idx]["title"],
            "query": query,
            "source": docs[idx]["source"],
            "retrieved_doc_index": ids,
            "retrieved_titles": [h["title"] for h in hits],
            "gold_in_top3": idx in ids,
            "gold_rank": ids.index(idx) + 1 if idx in ids else None,
            "retrieval_ms": round((time.time() - t0) * 1000, 1),
        })
    (RESULTS / "test_set_retrieval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(out)[["id", "title", "query", "source", "gold_in_top3", "gold_rank"]].to_csv(
        RESULTS / "test_set.csv", index=False)

    meta = {
        "tanggal": time.strftime("%Y-%m-%d %H:%M"),
        "set_uji": source,
        "n": len(out),
        "query": "kolom question asli dataset" if not args.query_col else args.query_col,
        "retrieval": f"hybrid (FAISS e5 + BM25, RRF k=60), top-{TOP_K}, tanpa query rewriter",
        "catatan_retrieval": "question asli ikut di-embed di indeks, jadi gold hampir pasti terambil (optimistis).",
        "prompt": "rag_core.SYSTEM_PROMPTS['ketat']",
        "system_prompt": rag_core.SYSTEM_PROMPTS["ketat"],
        "konteks": "judul + ringkasan + answer[:1500] + URL untuk tiap dokumen",
    }
    (RESULTS / "metodologi.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    hit = sum(o["gold_in_top3"] for o in out)
    print(f"Set uji: {len(out)} pertanyaan ({source}). Gold di top-{TOP_K}: {hit}/{len(out)}.")
    print(f"Tersimpan: {RESULTS/'test_set.csv'}, {RESULTS/'test_set_retrieval.json'}")


# ------------------------------------------------------------------ 2. generate
def _pricing(models):
    import requests

    try:
        data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
        return {m["id"]: m.get("pricing", {}) for m in data if m["id"] in models}
    except Exception as e:
        print(f"(Gagal mengambil harga model: {e})")
        return {}


def _status(llm) -> str:
    if not llm.ok:
        return "error"
    if not llm.content:
        return "kosong"
    if llm.finish_reason == "length":
        return "terpotong"
    return "ok"


def cmd_generate(args):
    path = RESULTS / "test_set_retrieval.json"
    if not path.exists():
        sys.exit("Jalankan dulu: python eval_generation.py prepare")
    items = json.loads(path.read_text(encoding="utf-8"))
    res = rag_core.load_resources(load_embedder=False)
    if not rag_core.get_api_key():
        sys.exit("OPENROUTER_API_KEY belum di-set. Salin .env.example ke .env lalu isi key milik sendiri.")

    # Perkiraan biaya (kasar: token ~ karakter/3.5; output dianggap max_tokens penuh)
    prices = _pricing(args.models)
    est_in = sum(len(m["content"]) for it in items
                 for m in rag_core.build_messages(it["query"], [res.docs[i] for i in it["retrieved_doc_index"]], "ketat")) / 3.5
    print(f"Perkiraan input ~{est_in:,.0f} token per model ({len(items)} pertanyaan).")
    for m in args.models:
        p = prices.get(m)
        if p is None:
            print(f"  {m}: TIDAK ADA di daftar model OpenRouter (cek ID-nya!)")
            continue
        cost = est_in * float(p.get("prompt", 0)) + len(items) * args.max_tokens * float(p.get("completion", 0))
        print(f"  {m}: maksimal ~${cost:.4f}")
    if not args.yes and input("Lanjut? ketik LANJUT: ").strip() != "LANJUT":
        sys.exit("Dibatalkan.")

    RESULTS.mkdir(exist_ok=True)
    for model in args.models:
        out_path = RESULTS / f"generation_{slug(model)}.csv"
        done = {}
        if out_path.exists():
            prev = pd.read_csv(out_path, dtype=str).fillna("")
            done = {r["id"]: r for _, r in prev.iterrows()}
        rows = []
        print(f"\n== {model} ==")
        for n, it in enumerate(items, 1):
            key = str(it["id"])
            if key in done and done[key]["status"] in ("ok", "terpotong") and not args.redo:
                rows.append(dict(done[key]))
                continue
            retrieved = [res.docs[i] for i in it["retrieved_doc_index"]]
            msgs = rag_core.build_messages(it["query"], retrieved, prompt_style="ketat")
            llm = None
            for attempt in range(args.retries + 1):
                llm = rag_core.call_llm_detail(msgs, temperature=0.2, max_tokens=args.max_tokens, model=model)
                if llm.ok or llm.status_code not in (429, 500, 502, 503, None):
                    break
                time.sleep(args.sleep * (attempt + 2))
            st = _status(llm)
            print(f"  [{n}/{len(items)}] id={key} {st} {llm.latency_s:.1f}s" + (f" - {llm.error}" if llm.error else ""))
            rows.append({
                "id": key, "model": model, "model_dijawab": llm.model or "", "query": it["query"],
                "retrieved_ids": "|".join(str(res.docs[i]["id"]) for i in it["retrieved_doc_index"]),
                "retrieved_titles": " | ".join(it["retrieved_titles"]),
                "gold_in_top3": it["gold_in_top3"], "gold_rank": it["gold_rank"] or "",
                "status": st, "finish_reason": llm.finish_reason or "", "answer": llm.content,
                "latency_s": round(llm.latency_s, 2), "prompt_tokens": llm.prompt_tokens or "",
                "completion_tokens": llm.completion_tokens or "", "cost_usd": llm.cost_usd if llm.cost_usd is not None else "",
                "error": llm.error or "",
            })
            pd.DataFrame(rows, columns=GEN_FIELDS).to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            time.sleep(args.sleep)
        pd.DataFrame(rows, columns=GEN_FIELDS).to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        c = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
        print(f"  Tersimpan {out_path.name}: {c}")


# ------------------------------------------------------------------ 3. score
_URL = re.compile(r"https?://\S+|www\.\S+")
_FOOTNOTE = re.compile(r"\[\d+\]")
_KLINIK_TERKAIT = re.compile(
    r"KLINIK TERKAIT(?:.{0,250}?\d{1,2} (?:Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agt|Agu|Sep|Okt|Nov|Des)\w* \d{4}){1,8}", re.S)
_TAIL_MARKERS = ["Demikian jawaban", "Demikian penjelasan", "Demikian sejauh", "These are the answers",
                 "DASAR HUKUM", "Dasar Hukum:", "REFERENSI", "PUTUSAN", "Tags"]
# Kalimat promosi/penyangkalan situs yang berulang di ribuan artikel (dicek dengan hitung frekuensi kalimat)
_BOILERPLATE = [
    re.compile(r"<!--.*?-->", re.S),  # sisa CSS dari MS Word
    re.compile(r"Belajar Hukum Secara Online dari Pengajar Berkompeten.{0,80}?Lihat Semua Kelas", re.S),
    re.compile(r"(?:Seluruh informasi hukum yang ada di Klinik|All legal informations? (?:which are )?available).{0,450}?"
               r"(?:Mitra Justika|Justika Partner Consultant|Justika)\.", re.S),
    re.compile(r"Perkaya riset hukum Anda.{0,250}?di sini\.", re.S),
    re.compile(r"Dinamisnya perkembangan regulasi.{0,400}?lebih lanjut\.", re.S),
    re.compile(r"Selalu perbarui kewajiban hukum.{0,300}?lebih lanjut\.", re.S),
    re.compile(r"Penjelasan\s+lebih\s+lanjut,?\s+(?:dapat\s+Anda\s+simak\s+dalam|silakan\s+baca)\s+ulasan\s+di\s+bawah\s+ini\.?"),
    # catatan riwayat artikel ("Artikel di bawah ini adalah pemutakhiran ... dipublikasikan pada 5 Mei 2011.")
    re.compile(r"Artikel di bawah ini adalah pemutakhiran.{0,300}?(?:dipublikasikan|diterbitkan) pada \d{1,2} \w+ \d{4}\.", re.S),
    re.compile(r"Artikel ini dibuat berdasarkan KUHP lama.{0,200}?\d{4}\."),
]
_MD = re.compile(r"[*#>`_|]+|^\s*[-•]\s+|^\s*\d+\.\s+", re.M)


def clean_reference_answer(text: str) -> str:
    """Bersihkan field 'answer': buang header, sisa CSS, kalimat promosi/penyangkalan situs,
    blok 'KLINIK TERKAIT', bagian penutup (mulai 'Demikian jawaban'/'DASAR HUKUM'/'REFERENSI'/
    'PUTUSAN'/'Tags', dst.), catatan kaki [n], URL."""
    t = text.replace("\xa0", " ")
    t = re.sub(r"^\s*ULASAN LENGKAP", "", t)
    t = t.replace("Terima kasih atas pertanyaan Anda.", " ")
    for p in _BOILERPLATE:
        t = p.sub(" ", t)
    t = _KLINIK_TERKAIT.sub(" ", t)
    cut = [t.find(m) for m in _TAIL_MARKERS if t.find(m) > 200]
    if cut:
        t = t[: min(cut)]
    t = _FOOTNOTE.sub(" ", t)
    t = _URL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_text(text: str) -> str:
    """Dipakai untuk jawaban LLM dan 'summarize': buang URL & simbol markdown."""
    t = _URL.sub(" ", str(text).replace("\xa0", " "))
    t = _MD.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


class _WordTokenizer:
    """Tokenizer ROUGE: lowercase + \\w+ (huruf/angka Unicode), tanpa stemmer."""

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())


def cmd_score(args):
    import sacrebleu
    from rouge_score import rouge_scorer

    files = sorted(RESULTS.glob("generation_*.csv"))
    if not files:
        sys.exit("Belum ada results/generation_*.csv. Jalankan generate dulu.")
    res = rag_core.load_resources(load_embedder=False)
    by_id = {str(d["id"]): d for d in res.docs}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=_WordTokenizer())

    all_rows, summary = [], []
    for f in files:
        df = pd.read_csv(f, dtype=str).fillna("")
        model = df["model"].iloc[0]
        valid = df[df["status"].isin(["ok", "terpotong"]) & (df["answer"].str.strip() != "")]
        hyps = [clean_text(a) for a in valid["answer"]]
        refs = {
            "summarize": [clean_text(by_id[i]["summarize"]) for i in valid["id"]],
            "answer": [clean_reference_answer(by_id[i]["answer"]) for i in valid["id"]],
        }
        per = pd.DataFrame({"id": valid["id"].values, "model": model, "status": valid["status"].values,
                            "len_jawaban_kata": [len(h.split()) for h in hyps]})
        srow = {"model": model, "n_total": len(df), "n_dinilai": len(valid),
                "n_terpotong": int((df["status"] == "terpotong").sum()),
                "n_kosong": int((df["status"] == "kosong").sum()),
                "n_error": int((df["status"] == "error").sum()),
                "len_jawaban_median": per["len_jawaban_kata"].median() if len(per) else None,
                "latency_mean_s": pd.to_numeric(valid["latency_s"], errors="coerce").mean(),
                "cost_total_usd": pd.to_numeric(df["cost_usd"], errors="coerce").sum()}
        for ref_name, rl in refs.items():
            if not hyps:
                continue
            per[f"len_ref_{ref_name}_kata"] = [len(r.split()) for r in rl]
            per[f"bleu_{ref_name}"] = [sacrebleu.sentence_bleu(h, [r], lowercase=True).score for h, r in zip(hyps, rl)]
            rs = [scorer.score(r, h) for h, r in zip(hyps, rl)]
            for k in ["rouge1", "rouge2", "rougeL"]:
                per[f"{k}_f1_{ref_name}"] = [x[k].fmeasure for x in rs]
            per[f"rouge1_recall_{ref_name}"] = [x["rouge1"].recall for x in rs]
            per[f"rouge1_precision_{ref_name}"] = [x["rouge1"].precision for x in rs]
            srow[f"bleu_korpus_{ref_name}"] = sacrebleu.corpus_bleu(hyps, [rl], lowercase=True).score
            if not args.no_bertscore:
                from bert_score import score as bscore

                _, _, F = bscore(hyps, rl, model_type=args.bertscore_model, batch_size=8, verbose=False)
                per[f"bertscore_f1_{ref_name}"] = F.tolist()
        for c in per.columns:
            if c.startswith(("bleu_", "rouge", "bertscore")):
                srow[f"{c}_mean"] = per[c].mean()
        all_rows.append(per)
        summary.append(srow)
        print(f"{model}: {len(valid)}/{len(df)} jawaban dinilai")

    pd.concat(all_rows).to_csv(RESULTS / "metrics_per_jawaban.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(RESULTS / "ringkasan_generation.csv", index=False)

    # Tabel ringkas untuk slide
    cols = ["model", "n_dinilai", "n_terpotong", "n_error", "len_jawaban_median"]
    for ref in ["summarize", "answer"]:
        cols += [c for c in [f"bleu_{ref}_mean", f"rouge1_f1_{ref}_mean", f"rouge2_f1_{ref}_mean",
                             f"rougeL_f1_{ref}_mean", f"bertscore_f1_{ref}_mean"] if c in sm.columns]
    cols += ["latency_mean_s", "cost_total_usd"]
    md = sm[cols].round(4).to_markdown(index=False)
    note = ("\n\nCatatan: BLEU 0-100; ROUGE & BERTScore 0-1 (F1). Referensi 'answer' sudah dibersihkan "
            "(lihat clean_reference_answer). BERTScore memotong teks di 512 token sehingga referensi "
            "'answer' yang panjang terpotong. Semua metrik hanya mengukur kemiripan teks dengan jawaban "
            "Klinik Hukumonline, bukan kebenaran hukum.\n")
    (RESULTS / "ringkasan_generation.md").write_text(md + note, encoding="utf-8")
    print("\n" + md + note)


# ------------------------------------------------------------------ 4. manual
def cmd_manual(args):
    files = sorted(RESULTS.glob("generation_*.csv"))
    if not files:
        sys.exit("Belum ada results/generation_*.csv.")
    frames = [pd.read_csv(f, dtype=str).fillna("") for f in files]
    common = set.intersection(*[set(f.loc[f["status"].isin(["ok", "terpotong"]), "id"]) for f in frames])
    random.seed(SEED)
    ids = sorted(random.sample(sorted(common), min(args.n, len(common))))
    rows = []
    for df in frames:
        for _, r in df[df["id"].isin(ids)].iterrows():
            rows.append({"model": r["model"], "id": r["id"], "query": r["query"],
                         "retrieved_titles": r["retrieved_titles"], "answer": r["answer"],
                         "skor_kebenaran_dokumen_1_5": "", "skor_relevansi_1_5": "",
                         "skor_kelengkapan_1_5": "", "skor_bahasa_1_5": "",
                         "skor_manual_1_5": "", "catatan": ""})
    out = RESULTS / "penilaian_manual.csv"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} sudah ada (mungkin sudah diisi). Pakai --overwrite untuk menimpa.")
    pd.DataFrame(rows).sort_values(["id", "model"]).to_csv(out, index=False)
    print(f"{len(ids)} pertanyaan x {len(frames)} model -> {out}")
    print("Kriteria: kebenaran thd dokumen sumber (pasal/istilah ada di dokumen?), relevansi, kelengkapan, bahasa.")
    print("Saran: sembunyikan kolom 'model' saat menilai agar tidak bias.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="buat set uji dan retrieval")
    a.add_argument("--n", type=int, default=N_TEST)
    a.add_argument("--test-set", help="CSV dengan kolom 'id' (mis. tabel 40 data David)")
    a.add_argument("--query-col", help="kolom pertanyaan di CSV (mis. question_parafrase); default question asli")
    a.set_defaults(func=cmd_prepare)

    g = sub.add_parser("generate", help="jalankan LLM")
    g.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    g.add_argument("--max-tokens", type=int, default=1200)
    g.add_argument("--retries", type=int, default=2)
    g.add_argument("--sleep", type=float, default=1.0, help="jeda antar panggilan (detik)")
    g.add_argument("--redo", action="store_true", help="ulangi juga yang sudah ok")
    g.add_argument("--yes", action="store_true", help="lewati konfirmasi biaya")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="hitung BLEU/ROUGE/BERTScore")
    s.add_argument("--no-bertscore", action="store_true")
    s.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("manual", help="lembar penilaian manual")
    m.add_argument("--n", type=int, default=15)
    m.add_argument("--overwrite", action="store_true")
    m.set_defaults(func=cmd_manual)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
