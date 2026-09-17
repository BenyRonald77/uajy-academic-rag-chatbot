"""
test_regressions.py — Penjaga terhadap bug yang pernah benar-benar terjadi.

Setiap test di file ini lahir dari kegagalan nyata yang ditemukan lewat
pengukuran, bukan dari skenario yang dibayang-bayangkan. Semuanya sempat
lolos dari tinjauan kode dan hanya terlihat setelah dijalankan pada dokumen
sungguhan. Karena itu masing-masing mendapat test sendiri: bug seperti ini
mudah kembali saat heuristiknya disetel ulang.
"""

from __future__ import annotations

import pytest

from app.config import RetrievalConfig
from app.reranker import rerank_candidates
from app.text_utils import ROMAN_NUMERALS, tokenize
from ingestion.chunking import MAX_PAGES_PER_CHUNK, _detect_heading, chunk_pages
from ingestion.extract_text import _is_gibberish


class TestRomanNumeralFalsePositive:
    """
    Bug: kata Indonesia diperlakukan sebagai angka romawi.

    Deteksi angka romawi semula memakai regex ``^[ivxlcdm]+$``. Pola itu juga
    cocok dengan kata yang kebetulan hanya tersusun dari huruf tersebut,
    sehingga kata seperti "di" lolos dari penyaring stopword dan masuk ke
    index BM25 sebagai istilah bermakna.
    """

    def test_kata_di_tidak_dianggap_angka_romawi(self):
        # "di" adalah stopword; d dan i keduanya huruf angka romawi.
        assert "di" not in tokenize("Apa syarat di Pasal 12?")

    @pytest.mark.parametrize("word", ["civil", "mil", "dim", "lid"])
    def test_kata_dari_huruf_romawi_tetap_utuh(self, word):
        """Kata bermakna tidak boleh hilang hanya karena hurufnya romawi."""
        assert word in tokenize(f"kata {word} dalam kalimat")

    def test_nomor_bab_tetap_dikenali(self):
        tokens = tokenize("Ketentuan pada BAB XII dan BAB IV")
        assert "xii" in tokens
        assert "iv" in tokens

    def test_daftar_romawi_eksplisit_tidak_memuat_kata_umum(self):
        for word in ("di", "civil", "mil", "dim"):
            assert word not in ROMAN_NUMERALS


class TestChunkPageSpanLeak:
    """
    Bug: overlap membocorkan batas rentang halaman.

    Batas ``MAX_PAGES_PER_CHUNK`` diperiksa sebelum sebuah blok ditambahkan,
    tetapi ketika chunk ditutup, blok overlap yang dibawa ke chunk berikutnya
    langsung memasukkan kembali halaman-halaman lama. Setiap penutupan chunk
    menyeret halaman itu maju terus, sehingga satu chunk bisa mengklaim 13
    halaman meski batasnya tiga — dan sitasi halaman menjadi tidak berguna.
    """

    @staticmethod
    def _pages_with_sparse_text(count: int = 14) -> list[tuple[int, str]]:
        """Banyak halaman yang masing-masing hanya berisi sedikit teks."""
        return [
            (
                page,
                f"Bagian {page} memuat keterangan singkat mengenai prosedur "
                f"administrasi akademik nomor {page} yang berlaku di fakultas.",
            )
            for page in range(1, count + 1)
        ]

    def test_rentang_halaman_tidak_pernah_melewati_batas(self):
        chunks = chunk_pages(
            self._pages_with_sparse_text(),
            chunk_size=1400,
            chunk_overlap=250,
            min_chunk_size=50,
            verbose=False,
        )
        assert chunks, "chunking tidak boleh mengembalikan daftar kosong"

        worst = max(len(c.page_numbers) for c in chunks)
        assert worst <= MAX_PAGES_PER_CHUNK, (
            f"ada chunk yang mengklaim {worst} halaman, "
            f"batasnya {MAX_PAGES_PER_CHUNK}"
        )

    def test_overlap_besar_pun_tetap_menghormati_batas(self):
        """Overlap yang lebih besar dari chunk_size adalah kasus paling rawan."""
        chunks = chunk_pages(
            self._pages_with_sparse_text(20),
            chunk_size=600,
            chunk_overlap=550,
            min_chunk_size=50,
            verbose=False,
        )
        for chunk in chunks:
            assert len(chunk.page_numbers) <= MAX_PAGES_PER_CHUNK

    def test_halaman_chunk_selalu_bersinambung(self):
        """Chunk tidak boleh mengklaim halaman yang melompat-lompat."""
        chunks = chunk_pages(
            self._pages_with_sparse_text(),
            chunk_size=1400,
            chunk_overlap=250,
            min_chunk_size=50,
            verbose=False,
        )
        for chunk in chunks:
            pages = chunk.page_numbers
            assert pages == sorted(pages)
            assert pages[-1] - pages[0] == len(pages) - 1, (
                f"halaman {pages} tidak bersinambung"
            )


class TestGibberishFilterFalsePositive:
    """
    Bug: penyaring teks rusak membuang tabel kurikulum.

    Penyaring menandai baris sebagai rusak bila banyak katanya memuat gugus
    konsonan tak wajar. Kode program studi seperti "TID" dan singkatan seperti
    "SKS" memenuhi kriteria itu, sehingga seluruh tabel mata kuliah terbuang —
    padahal justru bagian yang paling sering ditanyakan mahasiswa. Perbaikan:
    hanya kata panjang dan bukan huruf kapital seluruhnya yang dinilai.
    """

    @pytest.mark.parametrize("line", [
        "TID 1101 Kalkulus I 3 SKS",
        "TIF 2203 Struktur Data dan Algoritma 4 SKS",
        "MKWU 1001 Pendidikan Pancasila 2 SKS",
        "No Nilai Angka Nilai Huruf Nilai Indeks",
        "1 85 - 100 A 4,00",
    ])
    def test_baris_tabel_kurikulum_tidak_dibuang(self, line):
        assert not _is_gibberish(line), f"baris sah ikut terbuang: {line!r}"

    @pytest.mark.parametrize("line", [
        "lratsrutkurts satlukasf aatdluakpa",
        "UUUNNNIIIVVVEEERRRSSSIIITTTAAASSS AAATTTMMMAAA",
    ])
    def test_teks_rusak_tetap_terdeteksi(self, line):
        assert _is_gibberish(line), f"teks rusak lolos: {line!r}"

    def test_prosa_normal_tidak_terpengaruh(self):
        assert not _is_gibberish(
            "Mahasiswa wajib melakukan herregistrasi pada setiap awal semester."
        )


class TestFragmentPromotedToHeading:
    """
    Bug: potongan akhir kalimat terangkat menjadi judul bagian.

    Aturan "baris berhuruf kapital adalah judul" juga menangkap sisa kalimat
    yang terpotong, misalnya "... Universitas Atma Jaya Yogyakarta (UAJY)."
    menyisakan baris "UAJY).". Pengukuran pada dokumen ini menemukan tiga
    pecahan semacam itu mencemari **68 dari 222 chunk (31%)**, sehingga hampir
    sepertiga sitasi menampilkan label bagian tak bermakna seperti
    ``UAJY). › G. Herregistrasi › H. Cuti Studi``.

    Sitasi yang bisa diperiksa adalah nilai utama chatbot ini, jadi label yang
    kacau merusak justru bagian yang paling penting.
    """

    @pytest.mark.parametrize("fragment", [
        "UAJY).",
        "PKKMB).",
        "AB-PUI",
        "FTI).",
        "SKS).",
    ])
    def test_pecahan_kalimat_ditolak(self, fragment):
        assert _detect_heading(fragment) is None, (
            f"pecahan {fragment!r} tidak boleh dianggap judul"
        )

    @pytest.mark.parametrize("heading", [
        "PERPUSTAKAAN",
        "KATA PENGANTAR",
        "DAFTAR ISI",
        "SEJARAH SINGKAT",
        "PENYELENGGARAAN PENDIDIKAN",
        "UNIVERSITAS ATMA JAYA YOGYAKARTA",
        "VISI - MISI",
    ])
    def test_judul_sungguhan_tetap_dikenali(self, heading):
        hasil = _detect_heading(heading)
        assert hasil is not None, f"judul sah {heading!r} ikut tersaring"
        assert hasil[0] == 0

    def test_tanda_kurung_berpasangan_tetap_lolos(self):
        """Judul yang memuat kurung lengkap bukan pecahan kalimat."""
        assert _detect_heading("PROGRAM STUDI (REGULER) TEKNIK INDUSTRI") is not None

    def test_huruf_kapital_berangka_bukan_judul(self):
        """
        Batasan yang diketahui, bukan kelalaian.

        Aturan huruf kapital tidak menerima angka, sehingga "KURIKULUM 2025"
        tidak dikenali sebagai judul. Melonggarkannya berisiko menarik baris
        tabel berhuruf kapital masuk sebagai judul, dan itu perlu diukur lebih
        dulu. Nomor bab dan lampiran sendiri sudah ditangani pola tersendiri.
        """
        assert _detect_heading("KURIKULUM 2025") is None
        assert _detect_heading("BAB IV") is not None
        assert _detect_heading("LAMPIRAN 2") is not None

    @pytest.mark.parametrize("heading", ["BAB I", "BAB XII", "Pasal 1"])
    def test_heading_struktural_tidak_kena_ambang_huruf(self, heading):
        """
        Ambang jumlah huruf hanya berlaku pada aturan huruf kapital.

        "BAB I" hanya punya empat huruf tetapi dikenali lewat pola tersendiri,
        jadi tidak boleh ikut tersaring.
        """
        assert _detect_heading(heading) is not None


class TestPipelineVersionStaleness:
    """
    Bug: index usang karena perubahan kode tidak terdeteksi.

    `index_info.json` mencatat parameter chunking, tetapi parameter saja tidak
    cukup. Ketika daftar gugus konsonan yang sah diperbaiki, jumlah chunk
    berubah dari 219 menjadi 222 tanpa satu pun angka parameter berubah —
    index lama tetap terpakai dan satu-satunya cara mengetahuinya adalah
    mengukur manual.
    """

    def test_versi_lebih_lama_menandai_usang(self, tmp_path):
        from app.config import INGESTION_PIPELINE_VERSION
        from app.index_status import check_index_freshness

        info = {
            "pipeline_version": INGESTION_PIPELINE_VERSION - 1,
            "source_documents": [],
        }
        freshness = check_index_freshness(info, tmp_path)

        assert freshness.pipeline_outdated is True
        assert freshness.is_stale is True
        assert any("pipeline" in m.lower() for m in freshness.messages())

    def test_versi_sama_tidak_menandai_usang(self, tmp_path):
        from app.config import INGESTION_PIPELINE_VERSION
        from app.index_status import check_index_freshness

        info = {
            "pipeline_version": INGESTION_PIPELINE_VERSION,
            "source_documents": [],
        }
        freshness = check_index_freshness(info, tmp_path)

        assert freshness.pipeline_outdated is False
        assert freshness.is_stale is False

    def test_index_tanpa_versi_dianggap_versi_nol(self, tmp_path):
        """Index lama belum menyimpan nomor versi apa pun."""
        from app.index_status import check_index_freshness

        freshness = check_index_freshness({"source_documents": []}, tmp_path)

        assert freshness.indexed_pipeline_version == 0
        assert freshness.pipeline_outdated is True


class TestRetiredModelFallback:
    """
    Bug: model yang dihapus penyedianya memunculkan 404 mentah ke pengguna.

    Dua model Gemini dihapus di tengah masa hidup proyek ini —
    ``gemini-2.5-flash-lite`` lalu ``gemini-2.5-flash`` — dan keduanya
    menolak dengan *"no longer available to new users"*. Yang membuatnya sulit
    diantisipasi: ``models.list()`` tetap melaporkannya tersedia, sehingga
    kegagalannya hanya muncul saat model itu benar-benar dipanggil.

    Akibatnya mahasiswa melihat pesan galat HTTP di jendela chat, bukan
    jawaban. Nama model tunggal karena itu bukan konfigurasi yang aman.
    """

    @pytest.fixture(autouse=True)
    def bersihkan_cache(self):
        from app.llm_client import reset_model_cache

        reset_model_cache()
        yield
        reset_model_cache()

    def test_berpindah_saat_model_dihapus(self, monkeypatch):
        from app import llm_client

        dipanggil: list[str] = []

        def palsu(*, model, **kwargs):
            dipanggil.append(model)
            if model == "auto":
                raise RuntimeError(
                    "404 NOT_FOUND. This model auto is no longer available."
                )
            return "jawaban dari model cadangan"

        monkeypatch.setattr(llm_client, "_generate_once", palsu)

        hasil = llm_client.call_llm("pertanyaan", model="auto", raise_on_error=True)

        assert hasil == "jawaban dari model cadangan"
        assert dipanggil[0] == "auto", "model utama harus dicoba lebih dulu"
        assert len(dipanggil) >= 2, "seharusnya berpindah ke cadangan"

    def test_model_yang_bekerja_diingat(self, monkeypatch):
        """
        Model mati tidak boleh ditabrak ulang setiap permintaan.

        Tanpa ingatan ini, setiap pertanyaan menambah satu perjalanan jaringan
        yang sudah pasti gagal.
        """
        from app import llm_client

        dipanggil: list[str] = []

        def palsu(*, model, **kwargs):
            dipanggil.append(model)
            if model == "auto":
                raise RuntimeError("404 NOT_FOUND. model tidak tersedia")
            return "ok"

        monkeypatch.setattr(llm_client, "_generate_once", palsu)

        llm_client.call_llm("pertanyaan pertama", model="auto", raise_on_error=True)
        jumlah_awal = len(dipanggil)
        llm_client.call_llm("pertanyaan kedua", model="auto", raise_on_error=True)

        assert dipanggil[jumlah_awal] != "auto", (
            "permintaan kedua seharusnya langsung memakai model yang terbukti"
        )
        assert llm_client.effective_model("auto") != "auto"

    def test_kuota_habis_memicu_perpindahan(self, monkeypatch):
        """
        Kuota tingkat gratis berlaku PER MODEL, bukan per akun.

        Ini koreksi atas asumsi awal. Pesan galat API menyatakannya eksplisit::

            limit: 20, model: gemini-3.6-flash
            quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier

        Karena batasnya per model dan hanya 20 permintaan per hari, menyebar
        beban ke model cadangan benar-benar menambah kapasitas.
        """
        from app import llm_client

        dipanggil: list[str] = []

        def palsu(*, model, **kwargs):
            dipanggil.append(model)
            if model == "auto":
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
                    "generate_content_free_tier_requests, limit: 20"
                )
            return "jawaban dari model cadangan"

        monkeypatch.setattr(llm_client, "_generate_once", palsu)

        hasil = llm_client.call_llm("pertanyaan", model="auto", raise_on_error=True)

        assert hasil == "jawaban dari model cadangan"
        assert len(dipanggil) >= 2, "kuota habis per-model harus memicu model lain"

    def test_galat_autentikasi_tidak_memicu_perpindahan(self, monkeypatch):
        """
        Galat yang bukan soal model tidak akan membaik dengan berpindah.

        API key yang salah tetap salah untuk model apa pun, jadi mencoba
        seluruh rantai hanya menunda pesan galat yang benar.
        """
        from app import llm_client

        dipanggil: list[str] = []

        def palsu(*, model, **kwargs):
            dipanggil.append(model)
            raise RuntimeError("400 INVALID_ARGUMENT. API key not valid")

        monkeypatch.setattr(llm_client, "_generate_once", palsu)

        with pytest.raises(RuntimeError, match="400"):
            llm_client.call_llm("pertanyaan", model="auto", raise_on_error=True)

        assert len(dipanggil) == 1

    def test_model_disabled_memicu_fallback(self, monkeypatch):
        from app import llm_client

        dipanggil: list[str] = []

        def palsu(*, model, **kwargs):
            dipanggil.append(model)
            if model == "auto":
                raise RuntimeError(
                    '403 {"error":{"code":"model_disabled"}}'
                )
            return "jawaban dari model cadangan"

        monkeypatch.setattr(llm_client, "_generate_once", palsu)
        hasil = llm_client.call_llm("pertanyaan", model="auto", raise_on_error=True)

        assert hasil == "jawaban dari model cadangan"
        assert dipanggil == ["auto", "deepseek-v4-flash"]

    def test_seluruh_rantai_gagal_melempar_galat_terakhir(self, monkeypatch):
        from app import llm_client

        def palsu(*, model, **kwargs):
            raise RuntimeError(f"404 NOT_FOUND. {model} tidak tersedia")

        monkeypatch.setattr(llm_client, "_generate_once", palsu)

        with pytest.raises(RuntimeError, match="404"):
            llm_client.call_llm("pertanyaan", model="auto", raise_on_error=True)

    def test_setiap_model_utama_punya_cadangan(self):
        """Konfigurasi tanpa cadangan mengembalikan kerapuhan yang sama."""
        from app.config import LLM_MODEL, MODEL_FALLBACKS, UTILITY_MODEL

        for model in (LLM_MODEL, UTILITY_MODEL):
            assert MODEL_FALLBACKS.get(model), f"{model} belum punya model cadangan"

    def test_embedding_tidak_diberi_cadangan(self):
        """
        Model embedding TIDAK boleh punya fallback.

        Query harus di-embed oleh model yang sama dengan yang membangun index.
        Berpindah diam-diam akan membuat seluruh skor kemiripan tidak bermakna
        — kegagalan yang jauh lebih buruk daripada galat yang terlihat.
        """
        from app.config import EMBEDDING_MODEL, MODEL_FALLBACKS

        assert EMBEDDING_MODEL not in MODEL_FALLBACKS


class TestRerankGateFailOpen:
    """
    Bug: gate penolakan gagal-terbuka saat reranker error.

    Ketika ``rerank_candidates`` gagal, seluruh ``rerank_score`` tetap
    ``None``. Filter gate saat itu berbunyi::

        c.rerank_score is None or c.rerank_score >= rerank_min_score

    Klausa pertama meloloskan setiap kandidat tanpa skor, sehingga pertanyaan
    di luar cakupan diterima seolah-olah relevan. Akurasi penolakan turun dari
    100% ke 0% tanpa pesan error apa pun — persis yang terjadi saat API kena
    rate limit di tengah evaluasi.

    Perbaikannya berlapis: status keberhasilan dilaporkan, kandidat yang tidak
    dinilai model diberi skor 0, dan gate hanya diterapkan bila reranking
    benar-benar berjalan.
    """

    def test_kegagalan_dilaporkan_bukan_disembunyikan(self, candidate_factory, monkeypatch):
        monkeypatch.setattr(
            "app.reranker.call_utility_llm",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API mati")),
        )
        candidates = [candidate_factory(chunk_index=i) for i in range(3)]

        result, ok = rerank_candidates("pertanyaan apa pun", candidates)

        assert ok is False, "kegagalan reranker wajib dilaporkan"
        assert len(result) == 3, "kandidat tetap dikembalikan agar chatbot bisa menjawab"
        assert all(c.rerank_score is None for c in result)

    def test_kandidat_tanpa_skor_diberi_nol_bukan_none(self, candidate_factory, monkeypatch):
        """
        Model kadang hanya menilai sebagian kandidat.

        Yang tidak disebut harus dianggap tidak relevan (skor 0), bukan
        dibiarkan ``None`` — sebab ``None`` akan meloloskannya dari gate.
        """
        monkeypatch.setattr(
            "app.reranker.call_utility_llm",
            lambda *a, **k: '[{"id": 1, "score": 9}]',
        )
        candidates = [candidate_factory(chunk_index=i) for i in range(3)]

        result, ok = rerank_candidates("pertanyaan", candidates)

        assert ok is True
        assert all(c.rerank_score is not None for c in result), (
            "kandidat yang tidak dinilai model tidak boleh berskor None"
        )
        assert sorted(c.rerank_score for c in result) == [0.0, 0.0, 9.0]

    def test_semua_skor_rendah_menghasilkan_penolakan(self, candidate_factory, monkeypatch):
        """Skenario out-of-scope: semua kandidat berskor di bawah ambang."""
        monkeypatch.setattr(
            "app.reranker.call_utility_llm",
            lambda *a, **k: '[{"id": 1, "score": 1}, {"id": 2, "score": 0}]',
        )
        config = RetrievalConfig()
        candidates = [candidate_factory(chunk_index=i) for i in range(2)]

        result, ok = rerank_candidates("Apa rumus pythagoras?", candidates)
        passed = [c for c in result if (c.rerank_score or 0.0) >= config.rerank_min_score]

        assert ok is True
        assert passed == [], "kandidat berskor rendah seharusnya tidak lolos gate"

    def test_galat_sesaat_dicoba_ulang_sebelum_menyerah(self, candidate_factory, monkeypatch):
        """
        Galat 503/429 harus dicoba ulang, bukan langsung melumpuhkan gate.

        Reranker adalah satu-satunya tahap yang mampu menolak pertanyaan di
        luar cakupan, jadi menyerah pada percobaan pertama berarti pertahanan
        anti-halusinasi ikut mati setiap kali API sedang sibuk.
        """
        attempts = {"count": 0}

        def flaky(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return '[{"id": 1, "score": 8}, {"id": 2, "score": 2}]'

        monkeypatch.setattr("app.reranker.call_utility_llm", flaky)
        monkeypatch.setattr("app.reranker.time.sleep", lambda _: None)

        candidates = [candidate_factory(chunk_index=i) for i in range(2)]
        result, ok = rerank_candidates("pertanyaan", candidates)

        assert attempts["count"] == 3, "seharusnya mencoba ulang saat galat sesaat"
        assert ok is True
        assert result[0].rerank_score == 8.0

    def test_galat_permanen_tidak_dicoba_ulang(self, candidate_factory, monkeypatch):
        """Galat 404/400 tidak akan membaik, jadi jangan buang waktu."""
        attempts = {"count": 0}

        def broken(*args, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("404 NOT_FOUND: model tidak tersedia")

        monkeypatch.setattr("app.reranker.call_utility_llm", broken)

        _, ok = rerank_candidates(
            "pertanyaan", [candidate_factory(chunk_index=i) for i in range(2)]
        )

        assert attempts["count"] == 1
        assert ok is False
