import os
from pathlib import Path

dataset_dir = Path(__file__).parents[1] / "external/dataset-audio"


def dataset():
    if os.getenv("CI"):
        return [dataset_dir / "two channel/ff-16b-2c-44100hz.wav"]
    data = list((dataset_dir / "BeeMoved").rglob("*.*"))
    data += list((dataset_dir / "two channel").rglob("*.*"))
    data.append(dataset_dir / "Vision of Her/24-88.flac")
    return data


def test_numpy(): ...
def test_librosa(): ...
def test_torch(): ...
