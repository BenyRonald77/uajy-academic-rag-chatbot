# PRD — RAG Chatbot Dokumen Kampus

**Status:** Draft v1.0
**Dibuat untuk:** Project portofolio individu, implementasi di VS Code
**Terkait project sebelumnya:** Mirota Campus LLM Chatbot (fine-tuning LoRA) — project ini adalah evolusi dari sisi arsitektur: dari *fine-tuning model kecil* ke *retrieval-augmented generation (RAG) dengan LLM via API*.

---

## 1. Ringkasan Eksekutif

Chatbot tanya-jawab berbasis dokumen (RAG) yang menjawab pertanyaan pengguna **hanya berdasarkan isi dokumen resmi** (contoh: buku pedoman akademik, SOP kemahasiswaan, panduan skripsi) yang diunggah ke sistem. Berbeda dari project LLM sebelumnya yang meng-host model sendiri (berat di RAM, lambat di CPU), project ini memisahkan tanggung jawab:

- **Retrieval** (pencarian potongan dokumen relevan) → dijalankan lokal, ringan (CPU cukup).
- **Generation** (menyusun jawaban natural) → dipanggil lewat API LLM eksternal, sehingga aplikasi Streamlit tidak perlu memuat model besar ke memori.

Hasilnya: app tetap "AI beneran" (bukan cuma FAQ statis), tapi deploy-nya ringan dan stabil di Streamlit Community Cloud (±1 GB RAM, CPU only).

---

## 2. Latar Belakang & Masalah

Mahasiswa/civitas kampus sering kesulitan mencari informasi spesifik di dalam dokumen panjang (buku pedoman akademik, SOP, peraturan) karena:
- Dokumennya panjang (puluhan–ratusan halaman) dan tidak semua orang mau membaca semuanya.
- Informasi tersebar di beberapa bagian/bab berbeda.
- FAQ manual/statis cepat usang saat dokumen direvisi.

Chatbot berbasis LLM generik (tanpa retrieval) berisiko **halusinasi** — menjawab dengan percaya diri padahal informasinya salah/mengarang, apalagi untuk aturan/kebijakan yang butuh akurasi tinggi. RAG mengatasi ini dengan memaksa model menjawab berdasarkan potongan teks asli dokumen yang diambilkan lebih dulu (bukan dari "ingatan" model).

---

## 3. Tujuan (Goals)

1. Pengguna bisa bertanya dalam bahasa natural (Indonesia) dan mendapat jawaban akurat berdasarkan isi dokumen kampus.
2. Setiap jawaban menyertakan **sumber/kutipan** (halaman atau bagian dokumen) supaya bisa diverifikasi.
3. Sistem secara eksplisit bilang "tidak ditemukan di dokumen" ketika informasi memang tidak ada, bukan mengarang.
4. Aplikasi ringan dan bisa diakses publik lewat Streamlit tanpa masalah resource/performa.
5. Codebase cukup bersih untuk didemokan dan dijelaskan alurnya saat interview/portofolio.

### Non-Goals (di luar tujuan utama)
- Bukan chatbot umum yang bisa menjawab topik di luar dokumen yang diunggah.
- Bukan sistem multi-dokumen skala besar (ratusan dokumen, multi-kampus) — fokus 1 korpus dokumen dulu.
- Bukan real-time sync ke sistem akademik kampus (SIAKAD, dsb).

---

## 4. Target Pengguna & Use Case

**Target pengguna:** Mahasiswa, dosen, atau staf yang butuh jawaban cepat dari dokumen resmi kampus.

**Use case utama:**
- "Berapa minimal SKS untuk pengajuan skripsi?"
- "Prosedur cuti kuliah itu bagaimana?"
- "Dokumen apa saja yang dibutuhkan untuk yudisium?"

**Use case yang harus ditolak dengan sopan (guardrail):**
- Pertanyaan di luar topik dokumen ("siapa presiden Indonesia?") → dijawab bahwa itu di luar cakupan chatbot ini.
- Pertanyaan yang jawabannya tidak ada di dokumen → dijawab jujur "tidak ditemukan", bukan menebak.

---

## 5. Ruang Lingkup (Scope)

### MVP (wajib ada)
- Upload/ingest 1 dokumen (PDF) sumber kebenaran.
- Chunking + embedding + penyimpanan vektor lokal.
- Chat UI (Streamlit) single & multi-turn sederhana.
- Retrieval top-k chunk relevan per pertanyaan.
- Generation jawaban via API LLM, dengan instruksi ketat "hanya jawab dari konteks yang diberikan".
- Menampilkan sumber (nomor halaman/section) di setiap jawaban.
- Halaman "tentang" yang menjelaskan dokumen apa yang jadi sumber & batasan chatbot.

### Nice-to-have (kalau ada waktu lebih)
- Highlight teks asli yang jadi sumber jawaban (bukan cuma nomor halaman).
- Upload dokumen sendiri lewat UI (bukan hardcoded).
- Multi-dokumen dengan metadata kategori.
- Riwayat percakapan tersimpan per sesi (export ke PDF/markdown).

### Out of Scope
- Autentikasi user / multi-user roles.
- Fine-tuning model sendiri (sengaja dihindari — pelajaran dari project sebelumnya).
- Voice input/output.

---

## 6. Arsitektur Sistem

```
                ┌─────────────────────┐
                │   Dokumen Sumber     │  (PDF pedoman akademik/SOP)
                └──────────┬──────────┘
                           │  (satu kali / saat update dokumen)
                           ▼
                ┌─────────────────────┐
                │  Ingestion Pipeline  │
                │  - Extract teks      │
                │  - Chunking          │
                │  - Embedding         │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │   Vector Store       │  (FAISS/Chroma, disimpan lokal
                │   (index + metadata) │   sebagai file, ikut di-deploy)
                └──────────┬──────────┘
                           │  (tiap ada pertanyaan)
                           ▼
┌───────────┐   ┌─────────────────────┐   ┌───────────────────┐
│  User      │→→│  Retrieval           │→→│  Prompt Builder     │
│ (Streamlit)│   │  (similarity search, │   │  (system prompt +   │
│            │   │   top-k chunks)      │   │   konteks + tanya)  │
└───────────┘   └─────────────────────┘   └─────────┬──────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │  LLM API (generation)│
                                            │  (Claude/OpenAI/dll) │
                                            └─────────┬──────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Jawaban + Sumber    │
                                            │  ditampilkan ke user │
                                            └─────────────────────┘
```

Prinsip penting: **ingestion pipeline jalan sekali (offline, saat setup/update dokumen)**, hasilnya (index vektor) disimpan sebagai file dan di-deploy bersama app. Saat runtime, app cuma butuh: load index (ringan) + panggil API embedding untuk query + panggil API LLM untuk jawaban. Tidak ada model besar yang dimuat ke RAM.

---

## 7. Tech Stack yang Disarankan

| Komponen | Pilihan | Alasan |
|---|---|---|
| Bahasa | Python | konsisten dengan project sebelumnya |
| Parsing PDF | `pypdf` atau `pdfplumber` | ringan, cukup untuk teks; `pdfplumber` lebih baik untuk tabel |
| Embedding | `sentence-transformers` model multilingual kecil (mis. `paraphrase-multilingual-MiniLM-L12-v2`), **atau** API embedding (mis. dari provider LLM yang dipakai) | model lokal kecil (~120MB) cukup ringan untuk CPU; kalau mau lebih ringan lagi, pakai API embedding supaya app tidak perlu load model sama sekali |
| Vector store | `faiss-cpu` (paling ringan) atau `chromadb` (lebih mudah dipakai, ada metadata filtering) | keduanya jalan tanpa GPU, file index kecil untuk 1 dokumen |
| LLM generation | API pihak ketiga (pilih salah satu sesuai budget/akses): Anthropic Claude API, OpenAI API, atau opsi gratis/murah seperti Groq (Llama), Google Gemini free tier | menghindari masalah resource yang dialami project sebelumnya; kualitas jawaban juga lebih baik dari model 1B yang di-fine-tune sendiri |
| Framework RAG | Ditulis manual (tanpa LangChain/LlamaIndex) untuk MVP | lebih mudah dijelaskan alurnya saat interview; LangChain/LlamaIndex boleh dipakai belakangan sebagai "versi 2" kalau mau efisiensi development |
| Frontend | Streamlit | konsisten dengan project sebelumnya, sudah familiar |
| Deployment | Streamlit Community Cloud | gratis, cukup untuk beban kerja ini karena tidak ada model besar di-load runtime |
| Secret management | `st.secrets` (API key LLM & embedding kalau pakai API) | jangan commit API key ke repo |

> Catatan: pilihan API LLM sengaja tidak dikunci di PRD ini — sesuaikan dengan akses/API key yang kamu punya. Desain sistem dibuat API-agnostic (cukup ganti fungsi `call_llm()`).

---

## 8. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | Sistem dapat mengekstrak teks dari 1 file PDF sumber dan membaginya menjadi chunk (mis. 300–500 token per chunk, overlap 50–100 token) |
| FR-2 | Sistem menghasilkan embedding untuk setiap chunk dan menyimpannya ke vector store lokal beserta metadata (nomor halaman, judul section jika ada) |
| FR-3 | Saat user mengirim pertanyaan, sistem mengambil top-k (default k=4) chunk paling relevan berdasarkan similarity search |
| FR-4 | Sistem menyusun prompt ke LLM yang berisi: system instruction (jawab HANYA dari konteks, jangan mengarang), potongan konteks terpilih, dan pertanyaan user |
| FR-5 | Jika similarity score seluruh hasil retrieval di bawah ambang tertentu, sistem menjawab "informasi tidak ditemukan dalam dokumen" tanpa memanggil LLM (hemat biaya API + mencegah halusinasi) |
| FR-6 | Setiap jawaban menampilkan sumber (nomor halaman/section) dari chunk yang dipakai |
| FR-7 | Chat mendukung multi-turn dengan histori percakapan ditampilkan di UI |
| FR-8 | Tersedia halaman/panel "Tentang" yang menjelaskan dokumen sumber, tanggal versi dokumen, dan batasan chatbot |
| FR-9 | Sidebar pengaturan: jumlah top-k, model LLM yang dipakai (jika lebih dari satu opsi), tombol clear chat |

---

## 9. Non-Functional Requirements

- **Performa:** waktu respons target < 5 detik per pertanyaan (dominan waktu tunggu API LLM, bukan retrieval lokal).
- **Resource:** total memory footprint app (index + dependencies) harus tetap di bawah ±1 GB agar aman di Streamlit Community Cloud.
- **Akurasi/Faithfulness:** jawaban harus bisa ditelusuri balik ke chunk sumber; tidak boleh ada klaim di luar konteks yang diberikan.
- **Bahasa:** mendukung pertanyaan & jawaban Bahasa Indonesia dengan baik (pilih model embedding & LLM yang punya kemampuan multilingual/Indonesia yang memadai).
- **Biaya:** karena generation lewat API berbayar (kecuali pakai tier gratis), desain harus meminimalkan pemanggilan API yang tidak perlu (lihat FR-5).
- **Keamanan:** API key tidak boleh ter-hardcode/ter-commit ke repo publik.
- **Observability minimal:** log sederhana (bisa cukup `print`/file log) untuk mencatat pertanyaan, chunk yang diambil, dan jawaban — berguna untuk debugging & evaluasi kualitas.

---

## 10. Data Requirements

- **Sumber dokumen:** 1 dokumen PDF resmi kampus (disarankan: buku pedoman akademik atau SOP layanan mahasiswa — pilih yang benar-benar kamu punya aksesnya secara legal/publik).
- **Ukuran realistis untuk MVP:** 10–100 halaman. Kalau ingin lebih besar, uji dulu waktu ingestion & ukuran index.
- **Preprocessing:** bersihkan header/footer berulang, nomor halaman yang mengganggu ekstraksi, dan tabel kompleks (butuh penanganan khusus jika ada).
- **Chunking strategy:** mulai dengan chunk berbasis paragraf/section heading (bukan potong sembarang per N karakter) supaya konteks tidak terputus di tengah kalimat penting.

---

## 11. Metrik Evaluasi & Kriteria Sukses

Mengikuti pola evaluasi yang sudah kamu pakai di project sebelumnya (LLM-as-judge), tapi disesuaikan untuk RAG:

1. **Retrieval quality:** buat 15–20 pertanyaan uji dengan jawaban yang sudah diketahui ada di halaman berapa di dokumen. Ukur apakah chunk yang diambil sistem memang mengandung jawaban yang benar (recall@k).
2. **Answer faithfulness:** untuk tiap jawaban, cek manual (atau pakai LLM evaluator seperti project sebelumnya) apakah jawaban benar-benar didukung oleh chunk yang diambil, bukan hasil karangan.
3. **Refusal correctness:** uji dengan pertanyaan yang jawabannya memang tidak ada di dokumen — sistem harus menolak menjawab, bukan mengarang.
4. **Latency:** rata-rata waktu respons dari pertanyaan dikirim sampai jawaban tampil.

**Kriteria sukses MVP:** retrieval recall@4 ≥ 80% pada set uji, faithfulness dinilai baik pada mayoritas sampel uji, dan sistem menolak dengan benar pada seluruh pertanyaan "di luar dokumen" yang diuji.

---

## 12. Rencana Tahapan Development

1. **Setup & data prep** — kumpulkan dokumen sumber, bersihkan teks, tentukan strategi chunking.
2. **Ingestion pipeline** — script standalone: PDF → chunks → embeddings → simpan index lokal.
3. **Retrieval layer** — fungsi query → top-k chunk, uji manual dengan beberapa pertanyaan.
4. **Generation layer** — prompt template + pemanggilan API LLM, tangani kasus "tidak ditemukan".
5. **UI Streamlit** — chat interface, tampilkan sumber, sidebar settings, halaman "Tentang".
6. **Evaluasi** — jalankan set pertanyaan uji, hitung metrik di Bagian 11, catat hasilnya (bisa jadi bagian laporan/portofolio).
7. **Deployment** — Streamlit Community Cloud, setup secrets, uji beban ringan.
8. **Dokumentasi** — README dengan arsitektur, cara jalanin lokal, dan hasil evaluasi (ini yang dipamerkan di portofolio).

---

## 13. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| LLM tetap berhalusinasi walau sudah diberi konteks | Perkuat system prompt (instruksi eksplisit + contoh format penolakan), turunkan temperature, tambahkan FR-5 (ambang similarity) |
| Biaya API membengkak kalau banyak pengguna | Batasi rate/jumlah pertanyaan per sesi, cache jawaban untuk pertanyaan yang identik |
| Kualitas ekstraksi PDF buruk (tabel/format aneh) | Uji ekstraksi di awal sebelum lanjut ke tahap embedding; siapkan fallback pembersihan manual untuk bagian yang parsing-nya buruk |
| Dokumen sumber direvisi kampus | Pisahkan ingestion pipeline dari app utama supaya index tinggal di-generate ulang tanpa mengubah kode app |
| Embedding model lokal ternyata masih berat di Streamlit Cloud | Siapkan opsi fallback: pakai API embedding (bukan model lokal) |

---

## 14. Lampiran: Struktur Folder yang Disarankan

```
rag-chatbot-kampus/
├── data/
│   └── dokumen_sumber.pdf
├── ingestion/
│   ├── extract_text.py
│   ├── chunking.py
│   └── build_index.py        # jalankan sekali, hasilkan index/
├── index/
│   ├── faiss.index           # atau chroma persist dir
│   └── metadata.json
├── app/
│   ├── retrieval.py
│   ├── prompt_builder.py
│   ├── llm_client.py          # fungsi call_llm(), gampang ganti provider
│   └── main_streamlit.py
├── eval/
│   ├── test_questions.json
│   └── run_eval.py
├── requirements.txt
├── .streamlit/secrets.toml   # JANGAN di-commit
└── README.md
```

---

## 15. Pertanyaan Terbuka (isi sebelum mulai coding)

- [ ] Dokumen kampus mana yang akan dipakai sebagai sumber?
- [ ] Provider LLM & embedding mana yang akan dipakai (sesuai akses/API key yang tersedia)?
- [ ] Apakah butuh dukungan upload dokumen dinamis, atau cukup 1 dokumen tetap untuk MVP?
