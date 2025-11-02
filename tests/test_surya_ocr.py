"""
Tests for docling-surya OCR plugin.
Uses mocks to avoid real model downloads or GPU requirements.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

if TYPE_CHECKING:
    from docling.datamodel.document import ConvertedDocument

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def mock_surya_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock surya imports to avoid ImportError and model downloads."""

    class MockTextLine:
        """Mock text line with all required attributes."""

        def __init__(self):
            self.bbox = [10, 20, 100, 40]
            self.text = "Hello from Surya"
            self.confidence = 0.95
            self.polygon = [[10, 20], [100, 20], [100, 40], [10, 40]]

    class MockResult:
        """Mock OCR result."""

        def __init__(self):
            self.text_lines = [MockTextLine()]

    class MockPredictor:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def __call__(self, images: list, det_predictor: Any = None) -> list:
            # Return one result per image
            return [MockResult() for _ in images]

    class MockDetectionPredictor:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class MockFoundationPredictor:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    monkeypatch.setattr("surya.recognition.RecognitionPredictor", MockPredictor)
    monkeypatch.setattr("surya.detection.DetectionPredictor", MockDetectionPredictor)
    monkeypatch.setattr("surya.foundation.FoundationPredictor", MockFoundationPredictor)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> str:
    """Create a minimal 1-page PDF with text."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
    c.drawString(50, 150, "Hello from Surya")
    c.save()
    return str(pdf_path)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_plugin_is_discoverable() -> None:
    """Test that the plugin is registered via entry points."""
    import importlib.metadata as im

    entry_points = im.entry_points(group="docling")
    names = [ep.name for ep in entry_points]
    assert "surya-ocr" in names, "Plugin 'surya-ocr' not found in entry points"


def test_surya_model_initializes() -> None:
    """Test that SuryaOcrModel can be imported and initialized."""
    from docling_surya.plugin import SuryaOcrModel, SuryaOcrOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions

    options = SuryaOcrOptions()

    # Check that options instance has the 'kind' field
    assert hasattr(options, "kind"), "SuryaOcrOptions must have a 'kind' field"
    assert options.kind == "suryaocr", f"Expected kind='suryaocr', got '{options.kind}'"

    accel = AcceleratorOptions()

    model = SuryaOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=accel,
    )
    assert model.enabled
    assert model.scale == 3
    assert hasattr(model, "recognition_predictor")


def test_ocr_engines_factory() -> None:
    """Test that the plugin factory returns the model."""
    from docling_surya.plugin import ocr_engines

    engines = ocr_engines()
    assert "ocr_engines" in engines
    assert len(engines["ocr_engines"]) == 1
    from docling_surya.plugin import SuryaOcrModel

    assert engines["ocr_engines"][0] is SuryaOcrModel


def test_surya_options_has_kind_as_class_attr() -> None:
    """Test that SuryaOcrOptions.kind is accessible as a class attribute.

    This is required by docling's factory system which accesses opt_type.kind.

    To fix: Add this line to SuryaOcrOptions after the field definition:
        kind = "suryaocr"  # Make accessible as class attribute
    """
    from docling_surya.plugin import SuryaOcrOptions

    # Verify 'kind' exists as a class attribute (not just instance field)
    assert hasattr(SuryaOcrOptions, "kind"), (
        "SuryaOcrOptions class must have a 'kind' class attribute for docling's factory"
    )

    # Verify the value
    assert SuryaOcrOptions.kind == "suryaocr", (
        f"Expected SuryaOcrOptions.kind='suryaocr', got '{SuryaOcrOptions.kind}'"
    )


def test_pipeline_uses_surya_ocr(sample_pdf: str) -> None:
    """End-to-end test: convert PDF with Surya OCR.

    Note: This test will fail until SuryaOcrOptions is fixed to expose
    'kind' as a class attribute (required by docling's factory system).
    """
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",  # Must match the 'kind' field in SuryaOcrOptions
        allow_external_plugins=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    result = converter.convert(sample_pdf)
    doc: ConvertedDocument = result.document

    # Check that OCR was applied
    text = doc.export_to_markdown()
    assert "Hello from Surya" in text, "OCR text not found in output"


def test_cli_show_external_plugins() -> None:
    """Test CLI lists the plugin.

    Note: This will fail until SuryaOcrOptions is fixed to expose 'kind' as a class attribute.
    The CLI tries to load the plugin and hits the same AttributeError as the pipeline test.
    """
    import subprocess
    import sys

    # Try using 'docling' command directly
    result = subprocess.run(
        ["docling", "--show-external-plugins"],
        capture_output=True,
        text=True,
        check=False,
    )

    # If 'docling' command doesn't exist, skip
    if "not found" in result.stderr.lower() or "cannot find" in result.stderr.lower():
        pytest.skip("Docling CLI command not found in PATH")

    # If there's a module error, docling might not have CLI support
    if result.returncode != 0 and (
        "No module named" in result.stderr or "__main__" in result.stderr
    ):
        pytest.skip("Docling doesn't appear to have CLI support via '__main__'")

    # Check for the known 'kind' AttributeError
    if result.returncode != 0 and "AttributeError: kind" in result.stderr:
        pytest.fail(
            "Plugin failed to load due to missing class attribute.\n"
            "Fix: Add 'kind = \"suryaocr\"' as a class variable in SuryaOcrOptions"
        )

    # If command succeeded, check output
    if result.returncode == 0:
        assert "suryaocr" in result.stdout, (
            f"Plugin 'suryaocr' not found in CLI output.\nOutput was:\n{result.stdout}"
        )
    else:
        # Unknown error - report it
        pytest.fail(
            f"CLI command failed unexpectedly with returncode {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
