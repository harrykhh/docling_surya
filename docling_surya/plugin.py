import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Optional, Type

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
class SuryaOcrOptions(OcrOptions):
    kind: ClassVar[Literal["suryaocr"]] = "suryaocr"
    lang: list[str] = ["en"]
    use_gpu: Optional[bool] = None
    model_storage_directory: Optional[str] = None
    download_enabled: bool = True

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SuryaOcrModel(BaseOcrModel):
    _model_repo_folder = "SuryaOcr"

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: SuryaOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: SuryaOcrOptions
        self.scale = 3  # 72 dpi → 216 dpi

        if self.enabled:
            try:
                local_dir = settings.cache_dir / "models" / self._model_repo_folder
                local_dir.mkdir(parents=True, exist_ok=True)
                os.environ["MODEL_CACHE_DIR"] = str(local_dir)

                from surya.detection import DetectionPredictor
                from surya.foundation import FoundationPredictor
                from surya.recognition import RecognitionPredictor
            except ImportError as exc:
                raise ImportOcrError(
                    "surya-ocr not installed. Install via `uv add surya-ocr` or `pip install surya-ocr`."
                ) from exc

            foundation = FoundationPredictor()
            self.recognition_predictor = RecognitionPredictor(foundation)
            self.detection_predictor = DetectionPredictor()

    # ------------------------------------------------------------------- #
    @staticmethod
    def download_models(local_dir: Optional[Path] = None) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / SuryaOcrModel._model_repo_folder
        local_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODEL_CACHE_DIR"] = str(local_dir)

        import torch

        if torch.cuda.is_available():
            from surya.models import load_predictors

            load_predictors()
        return local_dir

    # ------------------------------------------------------------------- #
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_cells = []

                for rect in ocr_rects:
                    if rect.area() == 0:
                        continue

                    img = page._backend.get_page_image(scale=self.scale, cropbox=rect)
                    preds = self.recognition_predictor(
                        [img], det_predictor=self.detection_predictor
                    )

                    if not preds:
                        _log.warning("SuryaOCR returned empty result!")
                        continue

                    result = [
                        (tl.bbox, tl.text, tl.confidence)
                        for pred in preds
                        for tl in pred.text_lines
                    ]

                    del img

                    cells = [
                        TextCell(
                            index=i,
                            text=txt,
                            orig=txt,
                            confidence=conf,
                            from_ocr=True,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    coord=(
                                        (bbox[0] / self.scale) + rect.l,
                                        (bbox[1] / self.scale) + rect.t,
                                        (bbox[2] / self.scale) + rect.l,
                                        (bbox[3] / self.scale) + rect.t,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                        for i, (bbox, txt, conf) in enumerate(result)
                    ]
                    all_cells.extend(cells)

                self.post_process_cells(all_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return SuryaOcrOptions


# --------------------------------------------------------------------------- #
# Plugin factory (required by Docling)
# --------------------------------------------------------------------------- #
def ocr_engines():
    return {"ocr_engines": [SuryaOcrModel]}
