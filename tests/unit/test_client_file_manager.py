import pickle

import pytest

from client.client_file_manager import get_dataset_details

pytestmark = pytest.mark.unit


def test_get_dataset_details_finds_the_summary_file_for_an_absolute_path(tmp_path):
    # Regression test: get_dataset_details used to derive the summary
    # filename via `path.split(".")[1]`, which only works for a
    # "./relative/path.ext" shape (split(".") -> ["", "/relative/path",
    # "ext"], so [1] recovers the path). For an absolute path like
    # "/src/data/MNIST/x.pth" (split(".") -> ["/src/data/MNIST/x", "pth"],
    # no leading "" element), [1] silently grabbed just "pth" and looked for
    # a nonexistent "./pth_summary.data", always failing and returning None
    # -- surfaced during the Phase 4 real end-to-end run (client containers
    # use absolute dataset paths under /src/data).
    dataset_path = tmp_path / "MNIST" / "client_partition.pth"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.touch()

    summary = {"label_distribution": {0: 0.5, 1: 0.5}, "num_items": 42, "data_filename": str(dataset_path)}
    summary_path = tmp_path / "MNIST" / "client_partition_summary.data"
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)

    result = get_dataset_details(str(dataset_path))

    assert result == summary


def test_get_dataset_details_returns_none_when_summary_file_is_missing(tmp_path):
    dataset_path = tmp_path / "client_partition.pth"
    dataset_path.touch()

    assert get_dataset_details(str(dataset_path)) is None
