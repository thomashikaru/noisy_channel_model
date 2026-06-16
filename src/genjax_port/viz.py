"""Open the structured-output JSON (from ``run.py --output_json``) in an interactive browser viewer.

    PYTHONPATH=. python -m src.genjax_port.viz run.json
    PYTHONPATH=. python -m src.genjax_port.viz run.json --out viz.html --no-open

The viewer is a single self-contained HTML file (no dependencies, no build step, no network): it shows
the observed sentence as a surprisal heat-map, a per-word surprisal chart, and -- as you scrub through
the inference steps -- the top-K inferred intended-prefix distribution, the surprisal-gate diagnostics,
and the per-(event, word) rejuvenation accept rates. This launcher just inlines the JSON into the
template (:mod:`viz_template.html`) so the result opens straight from ``file://`` with no CORS dance;
the template alone also works standalone (drag a JSON onto its drop zone).
"""

import argparse
import json
import os
import tempfile
import webbrowser

TEMPLATE = os.path.join(os.path.dirname(__file__), "viz_template.html")
PLACEHOLDER = "__EMBEDDED_JSON__"


def render(data, template_path=TEMPLATE):
    """Inline ``data`` (a parsed JSON dict) into the HTML template; return the HTML string.

    The JSON is escaped so it cannot break out of its ``<script type="application/json">`` host (the
    only sequence that could is a literal ``</script>`` / ``</`` in the data)."""
    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    payload = json.dumps(data).replace("</", "<\\/")
    return html.replace(PLACEHOLDER, payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="path to the structured-output JSON from run.py --output_json")
    ap.add_argument("--out", default=None,
                    help="write the rendered HTML here (default: a temp file)")
    ap.add_argument("--no-open", action="store_true", help="write the HTML but do not open a browser")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    html = render(data)

    out = args.out
    if out is None:
        fd, out = tempfile.mkstemp(prefix="nc_viz_", suffix=".html")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(html)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
    out = os.path.abspath(out)
    print(f"wrote {out}")
    if not args.no_open:
        webbrowser.open("file://" + out)


if __name__ == "__main__":
    main()
