from pathlib import Path

import matplotlib.pyplot as plt

from src.common.report.plotting import (
    PlotSpec,
    create_grid,
    draw_barh,
    draw_heatmap,
    finish_plot,
    run_plot_jobs,
    save_figure,
    skip_plot,
)


def test_create_grid_and_save_figure(tmp_path: Path) -> None:
    fig, axes = create_grid(4, 2)
    assert len(axes) == 4
    out = save_figure(fig, tmp_path / "grid.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_skip_and_finish_plot(tmp_path: Path) -> None:
    spec = PlotSpec("demo", "demo.png", "Demo", "Demo caption")
    skipped = skip_plot(spec, "missing")
    assert skipped.path is None
    assert skipped.skip_reason == "missing"

    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    finished = finish_plot(spec, fig, tmp_path)
    assert finished.path == tmp_path / "demo.png"
    assert finished.path is not None
    assert finished.path.is_file()


def test_draw_barh_and_heatmap(tmp_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    import numpy as np

    draw_barh(ax1, ["a", "b"], [1.0, 2.0], "Barh", xlabel="Val")
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    draw_heatmap(ax2, matrix, ["x", "y"], ["x", "y"], "Heat")
    out = save_figure(fig, tmp_path / "charts.png")
    assert out.is_file()


def test_run_plot_jobs() -> None:
    spec = PlotSpec("job1", "job1.png", "Job 1", "Caption")
    artifacts = run_plot_jobs([(spec, lambda: skip_plot(spec, "test skip"))])
    assert len(artifacts) == 1
    assert artifacts[0].skip_reason == "test skip"
