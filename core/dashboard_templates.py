"""
Dashboard HTML templates (extracted from dashboard_server.py).

The HTML/CSS/JS is now in core/templates/dashboard.html.
"""

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_dashboard_html() -> str:
    """Load dashboard HTML from external template file."""
    template_path = _TEMPLATE_DIR / "dashboard.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    # Fallback for backward compatibility
    return """<!DOCTYPE html><html><head><title>Soberana Omega</title></head><body>
<h1>Dashboard not available</h1><p>Template file not found: core/templates/dashboard.html</p>
</body></html>"""


_DASHBOARD_HTML = _load_dashboard_html()
