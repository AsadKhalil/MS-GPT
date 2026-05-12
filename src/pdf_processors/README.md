# `src/pdf_processors/`

Multiple PDF-to-text strategies. Pick one based on the input volume, the structure of the source, and whether an LLM is in the loop.

For high-volume **OCR + text** extraction the canonical entrypoint actually lives in `src/vision_extractors/fast_pdf_extractor.py` (PyMuPDF + OCR fallback). This directory holds the alternatives.

## Which one should I use?

| File | Use when | Engine |
| --- | --- | --- |
| `pymupdf_processor.py` | You want raw PyMuPDF text only, no OCR, no LLM cleaning. Smallest dependency surface, fastest. | PyMuPDF |
| `llm_pdf_processor.py` | A single PDF needs LLM-assisted cleanup of the extracted text (typos, layout artifacts, broken table rows). | LLM (chat completion) |
| `batch_llm_processor.py` | The same as above but for a batch of PDFs. Adds concurrency and progress tracking. | LLM (chat completion) |
| `page_by_page_processor.py` | Long PDFs where LLM cleaning must happen per page (page is the unit of context window). | LLM (chat completion) |
| `grobid_processor.py` | You need structured academic parsing — section headings, references, figure captions. Requires GROBID running on `localhost:8070`. | GROBID |
| `grobid_batch_processor.py` | Same as above for many PDFs. | GROBID |
| `multi_format_converter.py` | Source includes non-PDF formats (DOCX, HTML) and needs unifying. | format-specific |
| `process_corpus_pdfs.py` | Full-corpus run with OCR fallback, resumability, and progress files. Driven by `config/fast_extractor.json`; wraps `vision_extractors/fast_pdf_extractor.FastPDFExtractor`. | PyMuPDF + OCR |

## Default for production

For the 40K-PDF corpus run, use `process_corpus_pdfs.py` — it's the resumable, configurable, OCR-fallback runner. GROBID is the right choice when you need section/reference structure (e.g., for the metadata extractor in `paper_pipeline`).
