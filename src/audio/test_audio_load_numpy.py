import os
from pathlib import Path

import audiofile
import librosa
import pytest
import soundfile
from scipy.io import wavfile

dataset_dir = Path(__file__).parents[2] / "external/dataset-audio"

group = "Load to numpy: "


def dataset():
    if os.getenv("CI" ):
        return [
            dataset_dir / "single channel/ff-16b-1c-44100hz.wav",
            dataset_dir / "two channel/ff-16b-2c-44100hz.wav",
        ]
    else:  # pragma: nocover
        data = list((dataset_dir / "BeeMoved").rglob("*.flac"))
        data += list((dataset_dir / "two channel").rglob("*.wav"))
        data += list((dataset_dir / "two channel").rglob("*.flac"))
        data.append(dataset_dir / "../Vision of Her/24-88.flac")
        return data


@pytest.fixture(params=dataset(), scope="module")
def file(request):
    return request.param


def test_audiofile(benchmark, file):
    def foo():
        audiofile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "audiofile"
    benchmark(foo)


def test_librosa(benchmark, file):
    def foo():
        librosa.load(file, mono=False, sr=None)

    benchmark.group = group + file.name
    benchmark.name = "librosa"
    benchmark(foo)


def test_scipy(benchmark, file):
    def foo():
        wavfile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "scipy"
    benchmark(foo)


def test_soundfile(benchmark, file):
    def foo():
        soundfile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "soundfile"
    benchmark(foo)
