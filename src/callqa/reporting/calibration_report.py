"""Calibration report (HTML + JSON), via `callqa calibrate`."""

from __future__ import annotations

import json
from pathlib import Path

from callqa.calibration import CalibrationResult
from callqa.reporting.common import jinja_env
from callqa.rubric import Rubric
from callqa.state import atomic_write_text


def write_calibration_reports(
    result: CalibrationResult, rubric: Rubric, output_dir: Path
) -> tuple[Path, Path]:
    dim_names = {d.id: d.name_he for d in rubric.dimensions}
    html = jinja_env().get_template("calibration_report.html.j2").render(
        result=result,
        dim_names=dim_names,
        flagged_names=[dim_names.get(d, d) for d in result.flagged_dimensions],
        footer_meta="",
    )
    html_path = output_dir / "reports" / "calibration.html"
    json_path = output_dir / "reports" / "calibration.json"
    atomic_write_text(html_path, html)
    atomic_write_text(json_path, json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return html_path, json_path
