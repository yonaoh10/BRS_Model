"""The management ("executive") batch report: one self-contained HTML over a batch of calls.

Three depths - executive summary, analysis, per-call detail - built from the
success-only, single-rubric cohort, with verbatim text only from calls whose
redaction is proven on.
"""

from callqa.reporting.executive.dataset import BatchFilters, parse_call_date
from callqa.reporting.executive.render import ExecutiveReport, build_executive_report

__all__ = ["BatchFilters", "ExecutiveReport", "build_executive_report", "parse_call_date"]
