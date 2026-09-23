import pytest
from pdblend.experimentation.lease import attempt_dir


def test_attempt_directories_are_immutable(tmp_path):
    path = attempt_dir(tmp_path, run_id="r", point="p", attempt=1)
    assert (path / "status.json").is_file()
    with pytest.raises(FileExistsError):
        attempt_dir(tmp_path, run_id="r", point="p", attempt=1)
