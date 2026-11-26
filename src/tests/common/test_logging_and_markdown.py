import logging
from pathlib import Path

import pandas as pd

from src.common.core.logging import configure_logging
from src.common.report.markdown import (
    dataframe_to_markdown,
    render_exit_code_table,
    render_plot_markdown,
)


def test_configure_logging() -> None:
    logger = configure_logging(debug=True)
    assert logger.level == logging.DEBUG
    quiet_logger = configure_logging(debug=False)
    assert quiet_logger.level == logging.INFO


def test_dataframe_to_markdown() -> None:
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x|y", "z"]})
    md = dataframe_to_markdown(df)
    assert "| col_a | col_b |" in md
    assert "x\\|y" in md
    assert dataframe_to_markdown(pd.DataFrame()) == "| (empty) |\n| --- |"


def test_render_plot_markdown(tmp_path: Path) -> None:
    from src.common.report.plotting import PlotArtifact

    img = tmp_path / "plot.png"
    img.write_bytes(b"png")
    plot_map = {
        "ok": PlotArtifact(
            plot_id="ok", title="Title", path=img, generated=True, caption="Caption"
        ),
        "skip": PlotArtifact(
            plot_id="skip",
            title="Title",
            path=None,
            generated=False,
            caption="Caption",
            skip_reason="No data",
        ),
    }
    md_ok = "\n".join(render_plot_markdown(plot_map, "ok", ".", "Fallback"))
    assert "![Title](./plot.png)" in md_ok
    md_skip = "\n".join(render_plot_markdown(plot_map, "skip", ".", "Fallback"))
    assert "No data" in md_skip


def test_render_exit_code_table() -> None:
    table = render_exit_code_table()
    assert "| code | name | meaning |" in table
    assert "| 0 |" in table
    assert "| 99 |" in table
