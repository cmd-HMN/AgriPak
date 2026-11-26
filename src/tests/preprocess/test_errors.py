from src.common.core.errors import PipelineError, TransformError


def test_transform_error_exit_code() -> None:
    err = TransformError("transform failed")
    assert isinstance(err, PipelineError)
    assert err.exit_code == 9
    assert err.message == "transform failed"
