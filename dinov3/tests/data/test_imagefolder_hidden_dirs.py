from PIL import Image

from dinov3.data.loaders import make_dataset


def test_imagefolder_ignores_hidden_top_level_directories(tmp_path):
    class_dir = tmp_path / "class_a"
    class_dir.mkdir()
    Image.new("RGB", (4, 4)).save(class_dir / "sample.png")

    hidden_dir = tmp_path / ".manifest"
    hidden_dir.mkdir()
    (hidden_dir / "metadata.json").write_text("{}")

    dataset = make_dataset(dataset_str=f"ImageFolder:root={tmp_path}")

    assert len(dataset) == 1
    assert dataset.classes == ["class_a"]

