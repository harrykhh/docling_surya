# Use SuryaOCR with Docling to OCR a PDF page and print Markdown.
#
# What this example does
# - Configures `SuryaOcrOptions` for OCR processing.
# - Runs the PDF pipeline with SuryaOCR and prints Markdown output.
# - Downloads SuryaOCR models from Hugging Face automatically when needed.
#
# Prerequisites
# - Install Docling with OCR dependencies.
# - Ensure your environment can import `docling`.
#
# How to run
# - From the repo root: `python examples/suryaocr_with_custom_models.py`.
# - The script prints the recognized text as Markdown to stdout.
#
# Notes
# - The default `source` points to an EPA PDF URL; replace with a local path if desired.
# - SuryaOCR models are downloaded automatically on first use.
# - Models are cached (typically under `~/.cache/huggingface`); set HF_HOME
# environment variable or configure a proxy if running in a restricted network environment.

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_surya import SuryaOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    source = "https://19january2021snapshot.epa.gov/sites/static/files/2016-02/documents/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf"

    pipeline = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",
        allow_external_plugins=True,
        ocr_options=SuryaOcrOptions(lang=["en"]),
    )

    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline),
        }
    )

    res = conv.convert(source)
    print(res.document.export_to_markdown())


if __name__ == "__main__":
    main()
