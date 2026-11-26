from src.common.core.errors import AnalysisError, PipelineError


def test_analysis_error_exit_code() -> None:
    err = AnalysisError("analysis failed")
    assert isinstance(err, PipelineError)
    assert err.exit_code == 7
    assert err.message == "analysis failed"
