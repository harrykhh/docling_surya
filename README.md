# Docling SuryaOCR Plugin

**Docling plugin** that brings the powerful **Surya OCR** engine into Docling.

> **License:** GPL-3.0-only – **must be used as an external plugin** (`allow_external_plugins=True`).

---

## Installation (uv)

```bash
uv pip install docling-surya
```

> **Note:**  
> - Supported only on **Linux x86_64** (matches `surya-ocr` dependency).  
> - Models (~1–2 GB) are downloaded automatically on first use.  
> - Cached under `~/.cache/huggingface` (or `HF_HOME` if set).

---

## Python Usage

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_surya import SuryaOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_model="suryaocr",           # Plugin engine name
    allow_external_plugins=True,     # Required for third-party plugins
    ocr_options=SuryaOcrOptions(
        lang=["en"],                 # OCR language(s)
        use_gpu=True,                # Optional: force GPU
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

result = converter.convert("path/to/document.pdf")
print(result.document.export_to_markdown())
```

---

## CLI Usage

```bash
# List available external plugins (should show "surya-ocr")
docling --show-external-plugins

# Run conversion with Surya OCR
docling --allow-external-plugins --ocr-engine=suryaocr path/to/document.pdf
```

---

## Example Script

See `examples/docling_with_custom_models.py`:

```bash
uv run examples/docling_with_custom_models.py
```

Processes a sample EPA PDF and prints Markdown output.

---

## Development with `uv`

```bash
# Clone the repo
git clone https://github.com/harrykhh/docling_surya
cd docling_surya

# Create virtual environment + install deps
uv venv
uv sync --all-extras

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Build wheel
uv build

# Install locally
uv pip install dist/docling_surya_ocr-*.whl

# Publish to PyPI (requires token)
uv publish
```

---

## Project Structure

```
docling_surya/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE (GPL-3.0)
├── doclin_surya/
│   ├── __init__.py
│   └── plugin.py           # Full SuryaOcrModel + factory
├── examples/
│   └── docling_with_custom_models.py
└── tests/
    └── test_surya_ocr.py
```

---

## Plugin Registration

The plugin registers via:

```toml
[project.entry-points."docling"]
surya-ocr = "docling_surya.plugin"
```

And exports the OCR engine via:

```python
def ocr_engines():
    return {"ocr_engines": [SuryaOcrModel]}
```

---

## License & Attribution

- **Surya OCR**: [datalab-to/surya](https://github.com/datalab-to/surya) – GPL-3.0

---


**Enjoy high-accuracy OCR on complex PDFs with Docling + Surya!**
