from pathlib import Path
import pytest
from app.services.data_loader import DataLoader


@pytest.fixture
def csv_dir(tmp_path):
    csv_content = "QA,CGA,CA,TC\n1.0,2.0,3.0,10.0\n4.0,5.0,6.0,20.0\n7.0,8.0,9.0,30.0\n"
    (tmp_path / "fruit-test-tc.csv").write_text(csv_content)
    leaf_content = "QA,CGA,CA,HDL\n1.0,2.0,3.0,50.0\n4.0,5.0,6.0,60.0\n"
    (tmp_path / "leaf-test-hdl.csv").write_text(leaf_content)
    return tmp_path


@pytest.fixture
def loader(csv_dir):
    return DataLoader(data_dir=csv_dir)


def test_load_preset_datasets_count(loader):
    metas = loader.load_preset_datasets()
    assert len(metas) == 2


def test_load_preset_metadata(loader):
    metas = loader.load_preset_datasets()
    tc = next(m for m in metas if m.target == "TC")
    assert tc.id == "fruit-test-tc"
    assert tc.plant_part == "Fruit"
    assert tc.target == "TC"
    assert tc.n_samples == 3
    assert tc.n_features == 3
    assert tc.feature_names == ["QA", "CGA", "CA"]
    assert tc.is_preset is True


def test_load_preset_leaf_detection(loader):
    metas = loader.load_preset_datasets()
    hdl = next(m for m in metas if m.target == "HDL")
    assert hdl.plant_part == "Leaf"
    assert hdl.filename == "leaf-test-hdl.csv"


def test_load_preset_nonexistent_dir(tmp_path):
    loader = DataLoader(data_dir=tmp_path / "nonexistent")
    metas = loader.load_preset_datasets()
    assert metas == []


def test_list_datasets(loader):
    loader.load_preset_datasets()
    datasets = loader.list_datasets()
    assert len(datasets) == 2
    assert all(d.id for d in datasets)


def test_get_dataset_preview(loader):
    loader.load_preset_datasets()
    detail = loader.get_dataset("fruit-test-tc")
    assert detail is not None
    assert len(detail.preview) == 3
    assert detail.preview[0]["QA"] == 1.0
    assert detail.preview[0]["TC"] == 10.0
    assert detail.preview[2]["TC"] == 30.0


def test_get_dataset_not_found(loader):
    loader.load_preset_datasets()
    assert loader.get_dataset("nonexistent") is None


def test_get_dataset_preview_capped_at_10(tmp_path):
    rows = "QA,TC\n" + "\n".join(f"{i}.0,{i * 10}.0" for i in range(1, 16)) + "\n"
    (tmp_path / "big.csv").write_text(rows)
    loader = DataLoader(data_dir=tmp_path)
    loader.load_preset_datasets()
    detail = loader.get_dataset("big")
    assert detail is not None
    assert len(detail.preview) == 10


def test_get_stats_values(loader):
    loader.load_preset_datasets()
    stats = loader.get_stats("fruit-test-tc")
    assert stats is not None
    assert stats.dataset_id == "fruit-test-tc"
    assert len(stats.stats) == 4  # QA, CGA, CA, TC

    tc_stat = next(s for s in stats.stats if s.column == "TC")
    assert tc_stat.mean == 20.0
    assert tc_stat.min == 10.0
    assert tc_stat.max == 30.0
    assert tc_stat.std == pytest.approx(10.0, abs=0.1)


def test_get_stats_not_found(loader):
    loader.load_preset_datasets()
    assert loader.get_stats("nonexistent") is None


def test_add_dataset(loader):
    loader.load_preset_datasets()
    csv = b"X,Y,Z\n1.0,2.0,10.0\n3.0,4.0,20.0\n"
    meta = loader.add_dataset("custom.csv", csv, name="My Custom")
    assert meta.name == "My Custom"
    assert meta.is_preset is False
    assert meta.n_samples == 2
    assert meta.n_features == 2
    assert loader.get_dataset(meta.id) is not None


def test_add_dataset_duplicate_id(loader):
    loader.load_preset_datasets()
    csv = b"QA,CGA,CA,TC\n1.0,2.0,3.0,10.0\n"
    meta = loader.add_dataset("fruit-test-tc.csv", csv)
    assert meta.id == "fruit-test-tc_1"


def test_delete_custom_dataset(loader):
    loader.load_preset_datasets()
    csv = b"X,Y,Z\n1.0,2.0,10.0\n"
    meta = loader.add_dataset("custom.csv", csv)
    result = loader.delete_dataset(meta.id)
    assert result == "deleted"
    assert loader.get_dataset(meta.id) is None


def test_delete_preset_fails(loader):
    loader.load_preset_datasets()
    result = loader.delete_dataset("fruit-test-tc")
    assert result == "is_preset"


def test_delete_not_found(loader):
    loader.load_preset_datasets()
    result = loader.delete_dataset("nonexistent")
    assert result == "not_found"


def test_header_whitespace_stripping(tmp_path):
    csv_with_spaces = "QA , CGA,CA  ,TC\n1.0,2.0,3.0,10.0\n4.0,5.0,6.0,20.0\n"
    (tmp_path / "spaced.csv").write_text(csv_with_spaces)
    loader = DataLoader(data_dir=tmp_path)
    loader.load_preset_datasets()
    detail = loader.get_dataset("spaced")
    assert detail is not None
    assert "QA" in detail.feature_names
    assert "CA" in detail.feature_names
    assert "TC" == detail.target


def test_meta_includes_all_columns(loader):
    loader.load_preset_datasets()
    detail = loader.get_dataset("fruit-test-tc")
    assert detail is not None
    assert detail.columns == ["QA", "CGA", "CA", "TC"]
