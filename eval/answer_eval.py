"""
answer_eval.py — Evaluasi mutu **jawaban**, bukan hanya retrieval.

Kenapa modul ini perlu ada
--------------------------
`run_eval.py` mengukur apakah halaman yang benar berhasil terambil. Ukuran itu
penting, tetapi berhenti di tengah jalan: dua klaim terbesar proyek ini justru
berada di sisi generasi, dan sampai sekarang belum pernah diuji sama sekali.

1. **"Bebas halusinasi"** — apakah setiap pernyataan dalam jawaban benar-benar
   didukung konteks yang diberikan? Retrieval yang sempurna tetap bisa diikuti
   jawaban yang mengarang.
2. **"Sitasi tepat"** — apakah nomor halaman yang *ditulis di dalam jawaban*
   benar-benar berasal dari konteks yang dipakai? Sitasi yang meyakinkan tetapi
   salah lebih berbahaya daripada tidak ada sitasi, sebab pembaca terdorong
   memercayainya.
3. **Penolakan sungguhan** — gate retrieval boleh saja meloloskan konteks yang
   tidak relevan; yang menentukan bagi pengguna adalah apakah jawaban akhirnya
   menolak.

Pembagian kerja: hal yang bisa dipastikan secara deterministik (sitasi,
deteksi penolakan) diperiksa dengan aturan eksplisit dan diuji lewat pytest.
Hanya groundedness yang memerlukan LLM sebagai penilai, sebab menilai
"apakah pernyataan ini didukung teks" memang butuh pemahaman bahasa.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.llm_client import call_utility_llm

# ──────────────────────────────────────────────
# Deteksi penolakan
# ──────────────────────────────────────────────

#: Penanda bahwa jawaban menolak menjawab.
#:
#: Frasa ini bukan karangan sendiri — semuanya diambil langsung dari
#: `app/prompt_builder.py`, yang mewajibkan model memakai kalimat penolakan
#: tertentu. Karena itu deteksinya deterministik dan tidak perlu LLM.
#:
#: Semua penanda terdiri dari beberapa kata. Kata tunggal seperti "maaf"
#: sengaja dihindari: jawaban yang sah pun bisa memuatnya sebagai kesopanan.
REFUSAL_MARKERS: tuple[str, ...] = (
    "tidak ditemukan dalam dokumen",
    "tidak ditemukan di dokumen",
    "informasi mengenai hal tersebut tidak ditemukan",
    "hanya dapat menjawab pertanyaan seputar dokumen",
    "hanya dapat menjawab pertanyaan terkait dokumen",
    "di luar cakupan",
    "tidak tercakup dalam dokumen",
    "hubungi bagian akademik",
    "tidak tersedia dalam dokumen",
    "tidak terdapat dalam dokumen",
)

#: Awal sebuah sitasi halaman di dalam teks jawaban.
_CITATION_START_RE = re.compile(r"halaman\s*:?\s*(?=\d)", re.IGNORECASE)

#: Satu angka, lalu pemisah, lalu angka berikutnya.
_NUMBER_RE = re.compile(r"\d{1,4}")
_SEPARATOR_RE = re.compile(r"\s*(,|dan|s\.?d\.?|[-–—])\s*", re.IGNORECASE)

#: Batas kewajaran agar rentang halaman yang salah baca tidak meledak.
_MAX_RANGE_SPAN = 40


def detect_refusal(answer: str) -> bool:
    """
    Tentukan apakah sebuah jawaban menolak menjawab.

    Args:
        answer: Teks jawaban akhir.

    Returns:
        True bila jawaban memuat salah satu penanda penolakan.
    """
    if not answer:
        return False
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def extract_cited_pages(answer: str) -> list[int]:
    """
    Ambil nomor halaman yang dikutip di dalam teks jawaban.

    Menangani halaman tunggal, rentang ("39-41"), dan daftar ("39, 40 dan 41").

    Diurai maju satu angka sekaligus, bukan dengan satu regex besar. Alasannya
    sebuah kasus nyata: format sitasi yang diminta prompt adalah
    ``Halaman X — [Nama Bagian]``, dan nama bagian sering diawali angka::

        📄 Sumber: Halaman 43 — 5. Beban studi dan beban kredit semester

    Tanda pisah panjang di situ berperan sebagai pemisah judul, bukan penanda
    rentang. Pendekatan regex sebelumnya membaca "43 — 5" sebagai rentang lalu
    tetap memungut angka 5 sebagai halaman, sehingga melaporkan sitasi palsu.
    Parser ini berhenti pada tanda pisah yang tidak diikuti angka lebih besar.

    Args:
        answer: Teks jawaban akhir.

    Returns:
        Daftar nomor halaman terurut tanpa duplikat.
    """
    if not answer:
        return []

    pages: set[int] = set()

    for match in _CITATION_START_RE.finditer(answer):
        position = match.end()

        number_match = _NUMBER_RE.match(answer, position)
        if not number_match:
            continue

        current = int(number_match.group())
        pages.add(current)
        position = number_match.end()

        # Lanjutkan selama pola "pemisah lalu angka" masih membentuk sitasi
        # yang wajar.
        while True:
            separator_match = _SEPARATOR_RE.match(answer, position)
            if not separator_match:
                break

            number_match = _NUMBER_RE.match(answer, separator_match.end())
            if not number_match:
                break

            following = int(number_match.group())
            separator = separator_match.group(1)
            is_dash = separator in {"-", "–", "—"}

            if is_dash:
                # Rentang harus menaik dan masuk akal lebarnya. Bila tidak,
                # tanda pisah itu memisahkan judul bagian — hentikan di sini
                # dan jangan pungut angkanya.
                if following <= current or following - current > _MAX_RANGE_SPAN:
                    break
                pages.update(range(current, following + 1))
            else:
                pages.add(following)

            current = following
            position = number_match.end()

    return sorted(pages)


# ──────────────────────────────────────────────
# Pemeriksaan sitasi
# ──────────────────────────────────────────────

@dataclass
class CitationCheck:
    """Hasil pemeriksaan sitasi sebuah jawaban."""

    cited_pages: list[int] = field(default_factory=list)
    context_pages: list[int] = field(default_factory=list)
    invalid_pages: list[int] = field(default_factory=list)

    @property
    def has_citation(self) -> bool:
        return bool(self.cited_pages)

    @property
    def is_valid(self) -> bool:
        """
        True bila setiap halaman yang dikutip berasal dari konteks.

        Jawaban tanpa sitasi dianggap tidak valid: prompt sistem mewajibkan
        sitasi, dan ketiadaannya membuat jawaban tidak bisa diaudit.
        """
        return self.has_citation and not self.invalid_pages

    @property
    def precision(self) -> float:
        """Porsi halaman terkutip yang benar-benar ada di konteks."""
        if not self.cited_pages:
            return 0.0
        valid = len(self.cited_pages) - len(self.invalid_pages)
        return valid / len(self.cited_pages)


def check_citations(answer: str, contexts) -> CitationCheck:
    """
    Bandingkan halaman yang dikutip jawaban dengan halaman konteks.

    Args:
        answer: Teks jawaban akhir.
        contexts: Kandidat yang dikirim sebagai konteks ke LLM.

    Returns:
        `CitationCheck` berisi halaman terkutip, halaman konteks, dan halaman
        yang dikutip tetapi tidak bersumber dari konteks mana pun.
    """
    cited = extract_cited_pages(answer)
    context_pages = sorted({page for c in contexts for page in c.page_numbers})
    invalid = [page for page in cited if page not in set(context_pages)]

    return CitationCheck(
        cited_pages=cited,
        context_pages=context_pages,
        invalid_pages=invalid,
    )


# ──────────────────────────────────────────────
# Groundedness (LLM sebagai penilai)
# ──────────────────────────────────────────────

GROUNDEDNESS_SYSTEM_PROMPT = """Kamu adalah penilai (evaluator) yang memeriksa apakah sebuah jawaban benar-benar didukung oleh teks sumber yang diberikan.

Tugasmu HANYA menilai dukungan faktual. Bukan menilai gaya bahasa, kelengkapan, atau apakah jawabannya enak dibaca.

Cara menilai:
1. Pecah jawaban menjadi pernyataan-pernyataan faktual.
2. Untuk setiap pernyataan, cari dukungannya di TEKS SUMBER.
3. Pernyataan yang tidak didukung teks sumber adalah pelanggaran, meskipun kebetulan benar menurut pengetahuan umum.

Yang TIDAK dihitung sebagai pelanggaran:
- Parafrasa dan peringkasan yang maknanya setara.
- Kalimat pembuka, penutup, dan ajakan menghubungi bagian akademik.
- Penyebutan nomor halaman atau nama bagian.
- Pernyataan bahwa informasi tidak ditemukan.

Panduan skor (0-10):
- 10 : Seluruh pernyataan faktual didukung teks sumber.
- 7-9 : Didukung, dengan satu detail kecil yang melebihi teks sumber.
- 4-6 : Ada pernyataan faktual penting yang tidak didukung.
- 0-3 : Banyak yang dikarang atau bertentangan dengan teks sumber.

Balas HANYA dengan JSON:
{"score": 8, "unsupported_claims": ["pernyataan yang tidak didukung"]}"""


@dataclass
class GroundednessVerdict:
    """Putusan penilai atas kesetiaan jawaban terhadap konteks."""

    score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    judge_failed: bool = False

    @property
    def is_grounded(self) -> bool:
        """Ambang 7 dipakai: parafrasa wajar diberi ruang, karangan tidak."""
        return self.score is not None and self.score >= GROUNDEDNESS_PASS_SCORE


#: Skor minimum agar jawaban dianggap setia pada konteks.
GROUNDEDNESS_PASS_SCORE = 7.0

#: Batas karakter tiap potongan konteks yang dikirim ke penilai.
_CONTEXT_CHAR_LIMIT = 1400


def build_groundedness_prompt(question: str, contexts, answer: str) -> str:
    """Susun prompt penilaian groundedness."""
    parts = [f"PERTANYAAN:\n{question}\n", "TEKS SUMBER:"]

    for i, candidate in enumerate(contexts, start=1):
        text = candidate.text.strip()
        if len(text) > _CONTEXT_CHAR_LIMIT:
            text = text[:_CONTEXT_CHAR_LIMIT].rsplit(" ", 1)[0] + " …"
        parts.append(f"\n--- Sumber #{i} (Halaman {candidate.page_label}) ---\n{text}")

    parts.append(f"\n\nJAWABAN YANG DINILAI:\n{answer}")
    parts.append("\nNilai dukungan faktual jawaban di atas. Balas hanya JSON.")
    return "\n".join(parts)


def parse_groundedness_response(raw: str) -> GroundednessVerdict:
    """
    Ubah balasan penilai menjadi `GroundednessVerdict`.

    Toleran terhadap code fence dan teks tambahan. Balasan yang tidak bisa
    diurai ditandai sebagai kegagalan penilai — bukan sebagai skor nol, sebab
    keduanya berarti hal yang sangat berbeda.
    """
    if not raw:
        return GroundednessVerdict(judge_failed=True)

    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", payload).strip()

    data = None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return GroundednessVerdict(judge_failed=True)

    if not isinstance(data, dict):
        return GroundednessVerdict(judge_failed=True)

    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError):
        return GroundednessVerdict(judge_failed=True)

    claims_raw = data.get("unsupported_claims") or []
    claims = [str(c) for c in claims_raw if isinstance(claims_raw, list)]

    return GroundednessVerdict(
        score=max(0.0, min(10.0, score)),
        unsupported_claims=claims,
    )


def judge_groundedness(
    question: str,
    contexts,
    answer: str,
    api_key: str | None = None,
) -> GroundednessVerdict:
    """
    Nilai apakah jawaban benar-benar didukung konteks.

    Kegagalan penilai dilaporkan lewat ``judge_failed``, tidak disamarkan
    sebagai skor rendah. Pelajaran dari gate reranker: kegagalan yang tampak
    seperti hasil sah membuat seluruh laporan menyesatkan.
    """
    if not contexts or not answer:
        return GroundednessVerdict(judge_failed=True)

    try:
        raw = call_utility_llm(
            build_groundedness_prompt(question, contexts, answer),
            system_instruction=GROUNDEDNESS_SYSTEM_PROMPT,
            max_tokens=1024,
            api_key=api_key,
            json_mode=True,
        )
    except Exception:
        return GroundednessVerdict(judge_failed=True)

    return parse_groundedness_response(raw)


# ──────────────────────────────────────────────
# Hasil per pertanyaan
# ──────────────────────────────────────────────

@dataclass
class AnswerResult:
    """Hasil evaluasi jawaban untuk satu pertanyaan."""

    id: int
    question: str
    category: str
    answer: str = ""

    # Penolakan.
    refused_by_retrieval: bool = False
    refused_in_answer: bool = False

    # Sitasi.
    cited_pages: list[int] = field(default_factory=list)
    context_pages: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    has_citation: bool = False
    citation_valid: bool = False
    citation_precision: float = 0.0
    expected_pages: list[int] = field(default_factory=list)
    cites_expected_page: bool = False

    # Groundedness.
    groundedness_score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    grounded: bool = False
    judge_failed: bool = False

    #: True bila panggilan pembuat jawaban gagal. Hasilnya harus dikeluarkan
    #: dari seluruh metrik: pesan error tidak memuat klaim faktual, sehingga
    #: penilai groundedness justru memberinya nilai sempurna dan angkanya
    #: menjadi menyesatkan.
    generation_failed: bool = False
    generation_error: str = ""

    latency_ms: float = 0.0

    @property
    def refused(self) -> bool:
        """Menolak, baik oleh gate retrieval maupun oleh jawaban itu sendiri."""
        return self.refused_by_retrieval or self.refused_in_answer

    def to_dict(self) -> dict:
        return dict(self.__dict__, refused=self.refused)


def aggregate_answer_metrics(results: list[AnswerResult]) -> dict:
    """
    Hitung metrik ringkas dari hasil evaluasi jawaban.

    In-scope dan out-of-scope dipisah, sebab yang diharapkan dari keduanya
    berlawanan: yang pertama harus dijawab, yang kedua harus ditolak.
    """
    # Jawaban yang gagal dihasilkan tidak boleh ikut dinilai apa pun.
    usable = [r for r in results if not r.generation_failed]
    generation_failures = sum(1 for r in results if r.generation_failed)

    in_scope = [r for r in usable if r.category == "in_scope"]
    out_of_scope = [r for r in usable if r.category == "out_of_scope"]

    # Groundedness hanya bermakna untuk jawaban yang memang menjawab.
    answered = [r for r in in_scope if not r.refused]
    judged = [r for r in answered if not r.judge_failed and r.groundedness_score is not None]
    cited = [r for r in answered if r.has_citation]

    def ratio(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    return {
        "total": len(results),
        "generation_failures": generation_failures,
        "in_scope_total": len(in_scope),
        "out_of_scope_total": len(out_of_scope),
        "answered_total": len(answered),

        "groundedness_rate": ratio(sum(1 for r in judged if r.grounded), len(judged)),
        "avg_groundedness_score": (
            sum(r.groundedness_score for r in judged) / len(judged) if judged else 0.0
        ),
        "judged_total": len(judged),
        "judge_failures": sum(1 for r in answered if r.judge_failed),

        "citation_presence": ratio(len(cited), len(answered)),
        "citation_validity": ratio(sum(1 for r in answered if r.citation_valid), len(answered)),
        "avg_citation_precision": (
            sum(r.citation_precision for r in cited) / len(cited) if cited else 0.0
        ),
        "cites_expected_page": ratio(
            sum(1 for r in answered if r.cites_expected_page), len(answered)
        ),

        "refusal_accuracy_e2e": ratio(
            sum(1 for r in out_of_scope if r.refused), len(out_of_scope)
        ),
        "false_refusal_e2e": ratio(sum(1 for r in in_scope if r.refused), len(in_scope)),

        "avg_latency_ms": (
            sum(r.latency_ms for r in results) / len(results) if results else 0.0
        ),
    }
