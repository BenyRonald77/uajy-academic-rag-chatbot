"""test_reranker.py — Parsing balasan reranker dan penyusunan promptnya."""

from __future__ import annotations

import pytest

from app.reranker import (
    SNIPPET_CHAR_LIMIT,
    build_rerank_prompt,
    parse_rerank_response,
    rerank_candidates,
)


class TestParseRerankResponse:
    def test_json_bersih(self):
        raw = '[{"id": 1, "score": 9}, {"id": 2, "score": 3}]'
        assert parse_rerank_response(raw, expected=2) == {1: 9.0, 2: 3.0}

    def test_dibungkus_code_fence(self):
        raw = '```json\n[{"id": 1, "score": 8.5}]\n```'
        assert parse_rerank_response(raw, expected=1) == {1: 8.5}

    def test_code_fence_tanpa_bahasa(self):
        raw = '```\n[{"id": 1, "score": 7}]\n```'
        assert parse_rerank_response(raw, expected=1) == {1: 7.0}

    def test_ada_teks_tambahan(self):
        raw = 'Berikut hasilnya: [{"id": 2, "score": 7}] semoga membantu.'
        assert parse_rerank_response(raw, expected=2) == {2: 7.0}

    def test_array_dibungkus_objek(self):
        raw = '{"results": [{"id": 1, "score": 6}]}'
        assert parse_rerank_response(raw, expected=1) == {1: 6.0}

    def test_id_di_luar_rentang_diabaikan(self):
        raw = '[{"id": 99, "score": 10}, {"id": 1, "score": 5}]'
        assert parse_rerank_response(raw, expected=3) == {1: 5.0}

    def test_skor_dibatasi_nol_sampai_sepuluh(self):
        raw = '[{"id": 1, "score": 15}, {"id": 2, "score": -4}]'
        assert parse_rerank_response(raw, expected=2) == {1: 10.0, 2: 0.0}

    def test_skor_desimal(self):
        assert parse_rerank_response('[{"id": 1, "score": 7.5}]', expected=1) == {1: 7.5}

    @pytest.mark.parametrize("raw", [
        "", "bukan json sama sekali", "[]", "{}", "null",
        '[{"tanpa": "field"}]',
        '[{"id": "abc", "score": "xyz"}]',
    ])
    def test_masukan_rusak_menghasilkan_kosong(self, raw):
        assert parse_rerank_response(raw, expected=3) == {}

    def test_entri_bukan_objek_dilewati(self):
        raw = '[1, 2, {"id": 1, "score": 8}]'
        assert parse_rerank_response(raw, expected=2) == {1: 8.0}


class TestBuildRerankPrompt:
    def test_memuat_pertanyaan_dan_semua_kandidat(self):
        prompt = build_rerank_prompt("Berapa IPK cum laude?", ["potongan A", "potongan B"])
        assert "Berapa IPK cum laude?" in prompt
        assert "potongan A" in prompt
        assert "potongan B" in prompt

    def test_kandidat_diberi_nomor_satu_basis(self):
        prompt = build_rerank_prompt("q", ["a", "b", "c"])
        assert "[1]" in prompt
        assert "[3]" in prompt
        assert "[0]" not in prompt

    def test_menyebut_rentang_yang_harus_dinilai(self):
        assert "[1]-[3]" in build_rerank_prompt("q", ["a", "b", "c"])


class TestRerankCandidates:
    def test_kandidat_tunggal_dilewati_tanpa_panggilan_api(self, candidate_factory, monkeypatch):
        """Tidak ada yang perlu diurutkan, jadi jangan buang kuota."""
        def jangan_dipanggil(*args, **kwargs):
            raise AssertionError("API tidak boleh dipanggil untuk satu kandidat")

        monkeypatch.setattr("app.reranker.call_utility_llm", jangan_dipanggil)
        candidates = [candidate_factory()]
        hasil, ok = rerank_candidates("q", candidates)
        assert hasil == candidates
        assert ok is True

    def test_urutan_mengikuti_skor(self, candidate_factory, monkeypatch):
        monkeypatch.setattr(
            "app.reranker.call_utility_llm",
            lambda *a, **k: '[{"id": 1, "score": 2}, {"id": 2, "score": 9}, {"id": 3, "score": 5}]',
        )
        candidates = [candidate_factory(chunk_index=i) for i in range(3)]
        hasil, ok = rerank_candidates("q", candidates)

        assert ok is True
        assert [c.chunk_index for c in hasil] == [1, 2, 0]
        assert [c.rerank_score for c in hasil] == [9.0, 5.0, 2.0]

    def test_skor_seri_mempertahankan_urutan_fusi(self, candidate_factory, monkeypatch):
        monkeypatch.setattr(
            "app.reranker.call_utility_llm",
            lambda *a, **k: '[{"id": 1, "score": 5}, {"id": 2, "score": 5}, {"id": 3, "score": 5}]',
        )
        candidates = [candidate_factory(chunk_index=i) for i in range(3)]
        hasil, _ = rerank_candidates("q", candidates)
        assert [c.chunk_index for c in hasil] == [0, 1, 2]

    def test_json_rusak_dianggap_gagal(self, candidate_factory, monkeypatch):
        monkeypatch.setattr("app.reranker.call_utility_llm", lambda *a, **k: "ngawur")
        candidates = [candidate_factory(chunk_index=i) for i in range(3)]
        hasil, ok = rerank_candidates("q", candidates)
        assert ok is False
        assert [c.chunk_index for c in hasil] == [0, 1, 2]

    def test_cuplikan_dipotong(self, candidate_factory, monkeypatch):
        """Prompt tidak boleh membengkak karena chunk yang sangat panjang."""
        captured = {}

        def tangkap(prompt, **kwargs):
            captured["prompt"] = prompt
            return '[{"id": 1, "score": 5}, {"id": 2, "score": 5}]'

        monkeypatch.setattr("app.reranker.call_utility_llm", tangkap)
        panjang = "kata " * 2000
        candidates = [
            candidate_factory(chunk_index=0, text=panjang),
            candidate_factory(chunk_index=1, text=panjang),
        ]
        rerank_candidates("q", candidates)

        # Dua cuplikan, masing-masing dibatasi, plus teks prompt itu sendiri.
        assert len(captured["prompt"]) < SNIPPET_CHAR_LIMIT * 2 + 1500
