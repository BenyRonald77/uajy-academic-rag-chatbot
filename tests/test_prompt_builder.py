"""Prompt jawaban harus menyertakan format sitasi yang bisa disalin model."""

from app.prompt_builder import build_prompt
from app.retrieval import RetrievalCandidate


def test_prompt_menyediakan_baris_sumber_persis_dari_konteks():
    context = RetrievalCandidate(
        chunk_index=1,
        text="Mahasiswa mengajukan permohonan cuti studi kepada program studi.",
        page_numbers=[39, 40, 41],
        section_title="H. Cuti Studi",
        heading_path=["LAYANAN AKADEMIK", "H. Cuti Studi"],
    )

    _system_prompt, user_prompt = build_prompt(
        "Bagaimana prosedur pengajuan cuti kuliah?", [context]
    )

    assert "FORMAT SITASI WAJIB" in user_prompt
    assert "📄 Sumber: Halaman 39-41 — H. Cuti Studi" in user_prompt
