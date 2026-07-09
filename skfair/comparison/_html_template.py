"""HTML template and rendering for ComparisonReport.to_html()."""

from datetime import datetime

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #2c3e50;
    background: #f8f9fa;
    line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
/* std error-bar toggle: show the plain charts by default, swap to the
   error-bar variant when <body> has the .show-std class. */
.chart-std { display: none; }
body.show-std .chart-std { display: block; }
body.show-std .chart-plain { display: none; }
.std-toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
/* sortable tables: clickable headers with a sort-direction cue */
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th::after { content: ""; font-size: 0.75em; color: #95a5a6; }
table.sortable th.sort-asc::after { content: " \\25B2"; }
table.sortable th.sort-desc::after { content: " \\25BC"; }
header {
    background: #fff;
    border-bottom: 3px solid #3498db;
    padding: 24px 0;
}
header h1 { font-size: 1.6rem; font-weight: 600; }
header .meta {
    font-size: 0.85rem;
    color: #7f8c8d;
    margin-top: 6px;
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}
.tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid #dee2e6;
    overflow-x: auto;
    position: sticky;
    top: 0;
    z-index: 100;
    background: #fff;
    padding-top: 8px;
}
.tab-btn {
    padding: 10px 20px;
    cursor: pointer;
    border: none;
    background: none;
    font-size: 0.95rem;
    color: #7f8c8d;
    border-bottom: 3px solid transparent;
    transition: color 0.2s, border-color 0.2s;
    white-space: nowrap;
}
.tab-btn:hover { color: #2c3e50; }
.tab-btn.active {
    color: #3498db;
    border-bottom-color: #3498db;
    font-weight: 600;
}
.controls-bar {
    background: #fff;
    padding: 12px 0 8px;
    font-size: 0.85rem;
    border-bottom: 1px solid #eee;
    position: sticky;
    top: 48px;
    z-index: 99;
}
.cb-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}
.cb-row .cb-label {
    font-weight: 600;
    color: #555;
    min-width: 130px;
    flex-shrink: 0;
}
.cb-row label {
    display: flex;
    align-items: center;
    gap: 3px;
    cursor: pointer;
    white-space: nowrap;
}
.toggle-btn {
    padding: 4px 12px;
    font-size: 0.8rem;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background: #fff;
    color: #555;
    cursor: pointer;
    white-space: nowrap;
}
.toggle-btn:hover { background: #f0f0f0; }
.controls-btns {
    display: flex;
    gap: 8px;
    margin-bottom: 6px;
}
.section-details {
    margin-bottom: 32px;
    border-top: 2px solid #e9ecef;
    padding-top: 8px;
}
.section-details:first-of-type { border-top: none; }
.section-details > summary {
    font-size: 1.3rem;
    font-weight: 600;
    color: #2c3e50;
    cursor: pointer;
    padding: 12px 0;
    list-style: none;
}
.section-details > summary::-webkit-details-marker { display: none; }
.section-details > summary::before {
    content: "\\25B6";
    display: inline-block;
    margin-right: 8px;
    font-size: 0.8rem;
    transition: transform 0.2s;
}
.section-details[open] > summary::before {
    transform: rotate(90deg);
}
.dataset-card {
    background: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    padding: 20px;
    margin-bottom: 20px;
    overflow-x: auto;
}
.dataset-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #555;
    margin-bottom: 10px;
}
.card {
    background: #fff;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    padding: 20px;
    margin-bottom: 20px;
    overflow-x: auto;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
th, td { padding: 8px 12px; text-align: right; }
th { background: #f8f9fa; font-weight: 600; border-bottom: 2px solid #dee2e6; }
td { border-bottom: 1px solid #eee; }
tr:nth-child(even) td { background: #fdfdfe; }
th:first-child, td:first-child { text-align: left; }
.best-val { font-weight: 700; color: #27ae60; }
@media print {
    .tabs, .controls-bar, #export-pdf-btn { display: none !important; }
    .container { max-width: none; padding: 10px; }
    .section-details { break-before: page; }
    .section-details:not([open]) { break-before: auto; display: block; }
    .section-details:not([open]) > *:not(summary) { display: none !important; }
    .section-details:not([open]) > summary { color: #999; font-style: italic; }
    .section-details:first-of-type { break-before: auto; }
    .section-details > summary {
        pointer-events: none;
        break-after: avoid;
    }
    .section-details > summary::before { content: ""; }
    .card, .dataset-card {
        box-shadow: none;
        border: 1px solid #dee2e6;
        overflow: visible !important;
        break-inside: avoid;
        margin-bottom: 12px;
        padding: 14px;
    }
    table { font-size: 0.8rem; }
    th, td { padding: 4px 8px; }
    img { max-width: 100%; height: auto; }
    header { border-bottom-width: 2px; padding: 12px 0; }
    header h1 { font-size: 1.3rem; }
    .meta { font-size: 0.75rem; }
}
"""

TAB_JS = """
document.addEventListener('DOMContentLoaded', function() {
    var btns = document.querySelectorAll('.tab-btn');
    var sections = document.querySelectorAll('.section-details');
    var tabBar = document.querySelector('.tabs');
    var tabBarHeight = tabBar ? tabBar.offsetHeight + 10 : 60;

    // Click handler: smooth scroll to section
    btns.forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            var target = document.getElementById(btn.dataset.tab);
            if (target) {
                // Ensure section is open
                target.open = true;
                var y = target.getBoundingClientRect().top + window.pageYOffset - tabBarHeight - 60;
                window.scrollTo({ top: y, behavior: 'smooth' });
            }
        });
    });

    // IntersectionObserver: highlight active tab on scroll
    if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.id;
                    btns.forEach(function(b) {
                        b.classList.toggle('active', b.dataset.tab === id);
                    });
                }
            });
        }, {
            rootMargin: '-' + tabBarHeight + 'px 0px -60% 0px',
            threshold: 0
        });
        sections.forEach(function(sec) { observer.observe(sec); });
    }

    // Unified visibility update
    function updateVisibility() {
        var checkedDs = new Set();
        document.querySelectorAll('.ds-cb:checked').forEach(function(cb) { checkedDs.add(cb.value); });
        var checkedClf = new Set();
        document.querySelectorAll('.clf-cb:checked').forEach(function(cb) { checkedClf.add(cb.value); });
        var checkedPerf = new Set();
        document.querySelectorAll('.perf-cb:checked').forEach(function(cb) { checkedPerf.add(cb.value); });
        var checkedFair = new Set();
        document.querySelectorAll('.fair-cb:checked').forEach(function(cb) { checkedFair.add(cb.value); });

        document.querySelectorAll('.dataset-card').forEach(function(card) {
            var show = true;
            var ds = card.dataset.dataset;
            var clf = card.dataset.classifier;
            var metric = card.dataset.metric;
            var mtype = card.dataset.metricType;

            if (ds && !checkedDs.has(ds)) show = false;
            if (clf && !checkedClf.has(clf)) show = false;
            if (metric && mtype === 'performance' && !checkedPerf.has(metric)) show = false;
            if (metric && mtype === 'fairness' && !checkedFair.has(metric)) show = false;

            card.style.display = show ? '' : 'none';
        });

        // Filter table columns by metric checkboxes
        var allMetrics = new Set([...checkedPerf, ...checkedFair]);
        document.querySelectorAll('th[data-metric], td[data-metric]').forEach(function(cell) {
            var m = cell.dataset.metric;
            cell.style.display = allMetrics.has(m) ? '' : 'none';
        });
    }

    document.querySelectorAll('.ds-cb, .clf-cb, .perf-cb, .fair-cb').forEach(function(cb) {
        cb.addEventListener('change', updateVisibility);
    });

    // Select All / Deselect All
    var selBtn = document.getElementById('toggle-select-btn');
    if (selBtn) {
        selBtn.addEventListener('click', function() {
            var allCbs = document.querySelectorAll('.ds-cb, .clf-cb, .perf-cb, .fair-cb');
            var allChecked = Array.from(allCbs).every(function(cb) { return cb.checked; });
            allCbs.forEach(function(cb) { cb.checked = !allChecked; });
            selBtn.textContent = allChecked ? 'Select All' : 'Deselect All';
            updateVisibility();
        });
    }

    // Export PDF
    var pdfBtn = document.getElementById('export-pdf-btn');
    if (pdfBtn) {
        pdfBtn.addEventListener('click', function() {
            window.print();
        });
    }

});
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scikit-fair Comparison Report</title>
<style>{css}</style>
</head>
<body>
<header>
<div class="container">
<div style="display:flex;align-items:center;justify-content:space-between;">
<h1>scikit-fair Comparison Report</h1>
<button id="export-pdf-btn" class="toggle-btn">Export PDF</button>
</div>
<div class="meta">
<span>Generated: {date}</span>
<span>{n_datasets} dataset(s)</span>
<span>{n_methods} method(s)</span>
<span>{n_classifiers} classifier(s)</span>
<span>{n_metrics} metric(s)</span>
</div>
</div>
</header>
<div class="container">
<div class="tabs">
{tab_buttons}
</div>
<div class="controls-bar">
<div class="controls-btns">
<button id="toggle-select-btn" class="toggle-btn">Deselect All</button>
{std_toggle}
</div>
{checkbox_rows}
</div>
{sections}
</div>
<script>{tabjs}</script>
<script>{extra_js}</script>
</body>
</html>
"""


# Self-contained (no CDN): std error-bar toggle + click-to-sort tables.
EXTRA_JS = """
document.addEventListener('DOMContentLoaded', function () {
    // --- std error-bar toggle -------------------------------------------
    var stdCb = document.getElementById('std-toggle-cb');
    if (stdCb) {
        stdCb.addEventListener('change', function () {
            document.body.classList.toggle('show-std', this.checked);
        });
    }

    // --- click-to-sort tables -------------------------------------------
    function cellValue(row, idx) {
        var txt = row.children[idx].textContent.trim();
        if (txt === '' || txt === '\\u2014') return null;   // em dash = NaN
        var num = parseFloat(txt);
        return isNaN(num) ? txt.toLowerCase() : num;
    }
    document.querySelectorAll('table.sortable').forEach(function (table) {
        var headers = table.querySelectorAll('thead th');
        headers.forEach(function (th, idx) {
            th.addEventListener('click', function () {
                var asc = !th.classList.contains('sort-asc');
                headers.forEach(function (h) { h.classList.remove('sort-asc', 'sort-desc'); });
                th.classList.add(asc ? 'sort-asc' : 'sort-desc');
                var tbody = table.querySelector('tbody');
                var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
                rows.sort(function (a, b) {
                    var va = cellValue(a, idx), vb = cellValue(b, idx);
                    if (va === null) return 1;          // NaNs sink to the bottom
                    if (vb === null) return -1;
                    if (va < vb) return asc ? -1 : 1;
                    if (va > vb) return asc ? 1 : -1;
                    return 0;
                });
                rows.forEach(function (r) { tbody.appendChild(r); });
            });
        });
    });
});
"""


def _df_to_styled_html(df):
    """Convert a summary DataFrame to a styled HTML table with best-value bolding."""
    from ._utils import DEFAULT_METRIC_DIRECTION
    import numpy as np

    html = ['<table class="sortable">']
    html.append('<thead><tr><th>Method</th>')
    for col in df.columns:
        html.append(f'<th data-metric="{col}">{col}</th>')
    html.append('</tr></thead>')
    html.append('<tbody>')

    # Find best value per column
    best_idx = {}
    for col in df.columns:
        vals = df[col].dropna()
        if vals.empty:
            continue
        direction = DEFAULT_METRIC_DIRECTION.get(col, "higher")
        if direction == "higher":
            best_idx[col] = vals.idxmax()
        elif direction == "zero":
            best_idx[col] = vals.abs().idxmin()
        elif direction == "one":
            best_idx[col] = (vals - 1.0).abs().idxmin()
        else:
            best_idx[col] = vals.idxmax()

    for method in df.index:
        html.append(f'<tr><td>{method}</td>')
        for col in df.columns:
            val = df.loc[method, col]
            formatted = f"{val:.4f}" if not (isinstance(val, float) and np.isnan(val)) else "\u2014"
            if best_idx.get(col) == method:
                html.append(f'<td data-metric="{col}" class="best-val">{formatted}</td>')
            else:
                html.append(f'<td data-metric="{col}">{formatted}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)


def _render_metric_cards(charts_dict, datasets, metric_type):
    """Render cards for Performance or Fairness sections.

    Parameters
    ----------
    charts_dict : dict
        {metric: {dataset: img_html}}
    datasets : list of str
    metric_type : str
        "performance" or "fairness"
    """
    cards = []
    for metric, ds_dict in charts_dict.items():
        label = metric.replace("_", " ").title()
        for ds in datasets:
            if ds not in ds_dict:
                continue
            cards.append(
                f'<div class="dataset-card" data-dataset="{ds}" '
                f'data-metric="{metric}" data-metric-type="{metric_type}">'
                f'<h3>{label} — {ds}</h3>'
                f'{ds_dict[ds]}'
                f'</div>'
            )
    return "\n".join(cards)


def _render_table_cards(tables, datasets):
    """Render cards for Tables section.

    Parameters
    ----------
    tables : dict
        {dataset_name: {agg_name: df}}
    datasets : list of str
    """
    cards = []
    # Collect all classifier/agg keys in order
    agg_keys = []
    for ds in datasets:
        if ds in tables:
            for k in tables[ds]:
                if k not in agg_keys:
                    agg_keys.append(k)

    for key in agg_keys:
        for ds in datasets:
            if ds not in tables or key not in tables[ds]:
                continue
            tbl_html = _df_to_styled_html(tables[ds][key])
            cards.append(
                f'<div class="dataset-card" data-dataset="{ds}" data-classifier="{key}">'
                f'<h3>{key} — {ds}</h3>'
                f'{tbl_html}'
                f'</div>'
            )
    return "\n".join(cards)


def _render_ranking_cards(rank_charts, datasets):
    """Render cards for Rankings section.

    Ranking heatmaps are pre-rendered images showing cross-metric comparisons,
    so metric checkbox filtering does not apply to them — unlike Tables (HTML
    columns) and Performance/Fairness (per-metric cards).

    Parameters
    ----------
    rank_charts : dict
        {dataset: {agg: img_html}}
    datasets : list of str
    """
    cards = []
    # Collect agg keys in order
    agg_keys = []
    for ds in datasets:
        if ds in rank_charts:
            for k in rank_charts[ds]:
                if k not in agg_keys:
                    agg_keys.append(k)

    for key in agg_keys:
        for ds in datasets:
            if ds not in rank_charts or key not in rank_charts[ds]:
                continue
            cards.append(
                f'<div class="dataset-card" data-dataset="{ds}" data-classifier="{key}">'
                f'<h3>{key} — {ds}</h3>'
                f'{rank_charts[ds][key]}'
                f'</div>'
            )
    return "\n".join(cards)


def _render_tradeoff_cards(tradeoff_charts, datasets):
    """Render cards for Tradeoff section.

    Parameters
    ----------
    tradeoff_charts : dict
        {fairness_metric: {dataset: img_html}}
    datasets : list of str
    """
    cards = []
    for metric, ds_dict in tradeoff_charts.items():
        label = metric.replace("_", " ").title()
        for ds in datasets:
            if ds not in ds_dict:
                continue
            cards.append(
                f'<div class="dataset-card" data-dataset="{ds}" '
                f'data-metric="{metric}" data-metric-type="fairness">'
                f'<h3>{label} — {ds}</h3>'
                f'{ds_dict[ds]}'
                f'</div>'
            )
    return "\n".join(cards)


def render_html_report(perf_charts, fair_charts, rank_charts, tradeoff_charts,
                       tables, metadata, datasets, has_std=False):
    """Assemble the full HTML report.

    Parameters
    ----------
    perf_charts : dict
        {metric: {dataset: img_html}}
    fair_charts : dict
        {metric: {dataset: img_html}}
    rank_charts : dict
        {dataset: {agg: img_html}}
    tradeoff_charts : dict
        {fairness_metric: {dataset: img_html}}
    tables : dict
        {dataset_name: {"Average": df, "Best": df, clf_name: df, ...}}
    metadata : dict
        Keys: n_datasets, n_methods, n_classifiers, n_metrics
    datasets : list of str
        Ordered dataset names for checkboxes and card ordering.
    """
    tab_order = ["Tables", "Performance", "Fairness", "Rankings", "Tradeoff"]
    tab_buttons = []
    section_parts = []

    # Build checkbox rows
    cb_rows = []

    # Datasets
    ds_cbs = "".join(
        f'<label><input type="checkbox" class="ds-cb" value="{ds}" checked> {ds}</label>'
        for ds in datasets
    )
    cb_rows.append(f'<div class="cb-row"><span class="cb-label">Datasets:</span>{ds_cbs}</div>')

    # Classifiers (from tables agg keys)
    clf_keys = []
    for ds in datasets:
        if ds in tables:
            for k in tables[ds]:
                if k not in clf_keys:
                    clf_keys.append(k)
    if clf_keys:
        clf_cbs = "".join(
            f'<label><input type="checkbox" class="clf-cb" value="{k}" checked> {k}</label>'
            for k in clf_keys
        )
        cb_rows.append(f'<div class="cb-row"><span class="cb-label">Classifiers:</span>{clf_cbs}</div>')

    # Performance metrics
    perf_keys = list(perf_charts.keys())
    if perf_keys:
        perf_cbs = "".join(
            f'<label><input type="checkbox" class="perf-cb" value="{m}" checked> '
            f'{m.replace("_", " ").title()}</label>'
            for m in perf_keys
        )
        cb_rows.append(f'<div class="cb-row"><span class="cb-label">Perf. metrics:</span>{perf_cbs}</div>')

    # Fairness metrics
    fair_keys = list(fair_charts.keys())
    if fair_keys:
        fair_cbs = "".join(
            f'<label><input type="checkbox" class="fair-cb" value="{m}" checked> '
            f'{m.replace("_", " ").title()}</label>'
            for m in fair_keys
        )
        cb_rows.append(f'<div class="cb-row"><span class="cb-label">Fair. metrics:</span>{fair_cbs}</div>')

    for i, tab_name in enumerate(tab_order):
        active = " active" if i == 0 else ""
        tab_id = tab_name.lower()
        tab_buttons.append(
            f'<button class="tab-btn{active}" data-tab="{tab_id}">{tab_name}</button>'
        )

        if tab_name == "Tables":
            content_html = _render_table_cards(tables, datasets)
        elif tab_name == "Performance":
            content_html = _render_metric_cards(perf_charts, datasets, "performance")
        elif tab_name == "Fairness":
            content_html = _render_metric_cards(fair_charts, datasets, "fairness")
        elif tab_name == "Rankings":
            content_html = _render_ranking_cards(rank_charts, datasets)
        elif tab_name == "Tradeoff":
            content_html = _render_tradeoff_cards(tradeoff_charts, datasets)
        else:
            content_html = ""

        section_parts.append(
            f'<details id="{tab_id}" class="section-details">'
            f'<summary>{tab_name}</summary>'
            + content_html
            + "</details>"
        )

    std_toggle = (
        '<label class="std-toggle"><input type="checkbox" id="std-toggle-cb"> '
        "show ± std</label>"
        if has_std else ""
    )

    return HTML_TEMPLATE.format(
        css=CSS,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_datasets=metadata.get("n_datasets", 0),
        n_methods=metadata.get("n_methods", 0),
        n_classifiers=metadata.get("n_classifiers", 0),
        n_metrics=metadata.get("n_metrics", 0),
        tab_buttons="\n".join(tab_buttons),
        checkbox_rows="\n".join(cb_rows),
        sections="\n".join(section_parts),
        std_toggle=std_toggle,
        tabjs=TAB_JS,
        extra_js=EXTRA_JS,
    )
