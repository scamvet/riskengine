"""Model families and calibration."""

from scamvet_riskengine.models.baseline import (
    BaselineConfig,
    Coefficient,
    build_baseline,
    coefficient_table,
    decision_scores,
)
from scamvet_riskengine.models.boosted import (
    BoostedConfig,
    boosted_scores,
    build_lightgbm,
    build_xgboost,
)
from scamvet_riskengine.models.onnx_export import (
    PARITY_TOLERANCE,
    SIZE_CAP_BYTES,
    ExportError,
    ExportResult,
    export_and_verify,
)
from scamvet_riskengine.models.calibration import (
    CALIBRATORS,
    CalibrationError,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
)

__all__ = [
    "CALIBRATORS",
    "PARITY_TOLERANCE",
    "SIZE_CAP_BYTES",
    "ExportError",
    "ExportResult",
    "export_and_verify",
    "BaselineConfig",
    "BoostedConfig",
    "Coefficient",
    "CalibrationError",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "build_baseline",
    "boosted_scores",
    "build_calibrator",
    "build_lightgbm",
    "build_xgboost",
    "coefficient_table",
    "decision_scores",
]
