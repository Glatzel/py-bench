from pathlib import Path

import audiofile
import librosa
import pytest
import soundfile
import torchaudio
from scipy.io import wavfile

dataset_dir = Path(__file__).parents[2] / "external/dataset-audio"

group = "Load "


@pytest.fixture(
    params=list((dataset_dir / "BeeMoved").rglob("*.*")) + list((dataset_dir / "two channel").rglob("*.*")),
    scope="module",
)
def file(request):
    return request.param


@pytest.mark.xfail
def test_audiofile(benchmark, file):
    def foo():
        audiofile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "audiofile"
    benchmark(foo)


@pytest.mark.xfail
def test_librosa(benchmark, file):
    def foo():
        librosa.load(file, mono=False, sr=None)

    benchmark.group = group + file.name
    benchmark.name = "librosa"
    benchmark(foo)


@pytest.mark.xfail
def test_scipy(benchmark, file):
    def foo():
        wavfile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "scipy"
    benchmark(foo)


@pytest.mark.xfail
def test_soundfile(benchmark, file):
    def foo():
        soundfile.read(file)

    benchmark.group = group + file.name
    benchmark.name = "soundfile"
    benchmark(foo)


@pytest.mark.xfail
def test_torch(benchmark, file):
    def foo():
        torchaudio.load(file, backend="soundfile")

    benchmark.group = group + file.name
    benchmark.name = "torch"
    benchmark(foo)
