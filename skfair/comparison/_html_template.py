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
.section {
    padding-top: 24px;
    margin-bottom: 32px;
    border-top: 2px solid #e9ecef;
}
.section:first-of-type { border-top: none; }
.section h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 16px;
    color: #2c3e50;
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
details { margin-bottom: 12px; }
summary {
    cursor: pointer;
    font-weight: 600;
    padding: 8px 0;
    font-size: 1rem;
}
details details summary {
    font-size: 0.95rem;
    padding-left: 12px;
    color: #555;
}
details details details summary {
    font-size: 0.9rem;
    padding-left: 24px;
    color: #666;
}
.expand-btn {
    padding: 6px 16px;
    font-size: 0.85rem;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    background: #fff;
    color: #555;
    cursor: pointer;
    margin: 12px 0;
}
.expand-btn:hover { background: #f0f0f0; }
@media print {
    .tabs { display: none; }
    .card { box-shadow: none; border: 1px solid #dee2e6; }
}
"""

TAB_JS = """
document.addEventListener('DOMContentLoaded', function() {
    var btns = document.querySelectorAll('.tab-btn');
    var sections = document.querySelectorAll('.section');
    var tabBar = document.querySelector('.tabs');
    var tabBarHeight = tabBar ? tabBar.offsetHeight + 10 : 60;

    // Click handler: smooth scroll to section
    btns.forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            var target = document.getElementById(btn.dataset.tab);
            if (target) {
                var y = target.getBoundingClientRect().top + window.pageYOffset - tabBarHeight;
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

    // Expand All / Collapse All
    var expandBtn = document.getElementById('expand-all-btn');
    if (expandBtn) {
        expandBtn.addEventListener('click', function() {
            var details = document.querySelectorAll('details');
            var allOpen = Array.from(details).every(function(d) { return d.open; });
            details.forEach(function(d) { d.open = !allOpen; });
            expandBtn.textContent = allOpen ? 'Expand All' : 'Collapse All';
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
<h1>scikit-fair Comparison Report</h1>
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
<button id="expand-all-btn" class="expand-btn">Expand All</button>
{sections}
</div>
<script>{tabjs}</script>
</body>
</html>
"""


def _df_to_styled_html(df, dataset_name):
    """Convert a summary DataFrame to a styled HTML table with best-value bolding."""
    from ._utils import DEFAULT_METRIC_DIRECTION
    import numpy as np

    html = ['<table>']
    html.append('<thead><tr><th>Method</th>')
    for col in df.columns:
        html.append(f'<th>{col}</th>')
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
                html.append(f'<td class="best-val">{formatted}</td>')
            else:
                html.append(f'<td>{formatted}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)


def _render_nested_chart_section(charts_dict, level="metric"):
    """Render nested <details> for metric → dataset → aggregation chart dicts.

    Parameters
    ----------
    charts_dict : dict
        For level="metric": {metric: {dataset: {agg: img_html}}}
        For level="dataset": {dataset: {agg: img_html}}
    level : str
        "metric" for Performance/Fairness (3 levels), "dataset" for Rankings (2 levels).
    """
    parts = []
    if level == "metric":
        for i, (metric_name, ds_dict) in enumerate(charts_dict.items()):
            metric_label = metric_name.replace("_", " ").title()
            ds_parts = []
            for j, (ds_name, agg_dict) in enumerate(ds_dict.items()):
                agg_parts = []
                for k, (agg_name, img_html) in enumerate(agg_dict.items()):
                    agg_parts.append(
                        f'<details><summary>{agg_name}</summary>'
                        f'<div class="card">{img_html}</div></details>'
                    )
                ds_parts.append(
                    f'<details><summary>{ds_name}</summary>'
                    + "\n".join(agg_parts)
                    + '</details>'
                )
            parts.append(
                f'<details><summary>{metric_label}</summary>'
                + "\n".join(ds_parts)
                + '</details>'
            )
    elif level == "dataset":
        for i, (ds_name, agg_dict) in enumerate(charts_dict.items()):
            agg_parts = []
            for agg_name, img_html in agg_dict.items():
                agg_parts.append(
                    f'<details><summary>{agg_name}</summary>'
                    f'<div class="card">{img_html}</div></details>'
                )
            parts.append(
                f'<details><summary>{ds_name}</summary>'
                + "\n".join(agg_parts)
                + '</details>'
            )
    return "\n".join(parts)


def _render_tradeoff_section(tradeoff_charts):
    """Render tradeoff section with expandable <details> per fairness metric.

    Parameters
    ----------
    tradeoff_charts : dict
        {fairness_metric_name: img_html}
    """
    if not tradeoff_charts:
        return ""
    parts = []
    for metric_name, img_html in tradeoff_charts.items():
        label = metric_name.replace("_", " ").title()
        parts.append(
            f'<details><summary>{label}</summary>'
            f'<div class="card">{img_html}</div></details>'
        )
    return "\n".join(parts)


def render_html_report(perf_charts, fair_charts, rank_charts, tradeoff_charts,
                       tables, metadata):
    """Assemble the full HTML report.

    Parameters
    ----------
    perf_charts : dict
        {metric: {dataset: {agg: img_html}}}
    fair_charts : dict
        {metric: {dataset: {agg: img_html}}}
    rank_charts : dict
        {dataset: {agg: img_html}}
    tradeoff_charts : dict
        {fairness_metric: img_html}
    tables : dict
        {dataset_name: {"Average": df, "Best": df, clf_name: df, ...}}
    metadata : dict
        Keys: n_datasets, n_methods, n_classifiers, n_metrics
    """
    tab_order = ["Tables", "Performance", "Fairness", "Rankings", "Tradeoff"]
    tab_buttons = []
    section_parts = []

    for i, tab_name in enumerate(tab_order):
        active = " active" if i == 0 else ""
        tab_id = tab_name.lower()
        tab_buttons.append(
            f'<button class="tab-btn{active}" data-tab="{tab_id}">{tab_name}</button>'
        )

        if tab_name == "Tables":
            content = []
            for ds_name, agg_dict in tables.items():
                ds_parts = []
                for agg_name, tbl_df in agg_dict.items():
                    tbl_html = _df_to_styled_html(tbl_df, ds_name)
                    ds_parts.append(
                        f'<details><summary>{agg_name}</summary>'
                        f'<div class="card">{tbl_html}</div></details>'
                    )
                content.append(
                    f'<details><summary>{ds_name}</summary>'
                    + "\n".join(ds_parts)
                    + '</details>'
                )
            content_html = "\n".join(content)
        elif tab_name == "Performance":
            content_html = _render_nested_chart_section(perf_charts, level="metric")
        elif tab_name == "Fairness":
            content_html = _render_nested_chart_section(fair_charts, level="metric")
        elif tab_name == "Rankings":
            content_html = _render_nested_chart_section(rank_charts, level="dataset")
        elif tab_name == "Tradeoff":
            content_html = _render_tradeoff_section(tradeoff_charts)
        else:
            content_html = ""

        section_parts.append(
            f'<div id="{tab_id}" class="section">'
            f'<h2>{tab_name}</h2>'
            + content_html
            + "</div>"
        )

    return HTML_TEMPLATE.format(
        css=CSS,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_datasets=metadata.get("n_datasets", 0),
        n_methods=metadata.get("n_methods", 0),
        n_classifiers=metadata.get("n_classifiers", 0),
        n_metrics=metadata.get("n_metrics", 0),
        tab_buttons="\n".join(tab_buttons),
        sections="\n".join(section_parts),
        tabjs=TAB_JS,
    )
