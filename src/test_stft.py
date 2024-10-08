import os
from pathlib import Path

import librosa
import numpy as np
import pyfftw
import pytest
import scipy
import scipy.signal
import soundfile
import ssqueezepy
import torch
import torchaudio

group = "STFT: "
n_fft = 1024
win_len = 1024
hop_len = 512
dataset_dir = Path(__file__).parents[1] / "external/dataset-audio"


def dataset():  # pragma: nocover
    if os.getenv("CI"):
        return [
            dataset_dir / "single channel/ff-16b-1c-44100hz.wav",
            dataset_dir / "two channel/ff-16b-2c-44100hz.wav",
        ]
    else:  # pragma: nocover
        return [
            dataset_dir / "BeeMoved/Sample_BeeMoved_96kHz24bit.flac",
            dataset_dir / "../Vision of Her/24-88.flac",
        ]


@pytest.fixture(params=dataset(), scope="module")
def audio(request):
    file = request.param
    wave, _ = soundfile.read(file, dtype="float32")
    if len(wave.shape) > 1:
        wave = wave.mean(1)
    return str(file.name), wave


def test_scipy(benchmark, audio):
    def foo(data):
        scipy.signal.stft(data, noverlap=hop_len, nfft=n_fft, nperseg=win_len)

    benchmark.group = group + f"{audio[0]}"
    benchmark.name = "scipy"
    benchmark(foo, audio[1])


def test_librosa(benchmark, audio):
    def foo(data):
        sp = librosa.stft(data, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
        np.power(np.abs(sp, out=sp), 2.0, out=sp)

    librosa.set_fftlib(pyfftw.interfaces.numpy_fft)
    benchmark.group = group + f"{audio[0]}"
    benchmark.name = "librosa"
    benchmark(foo, audio[1])


def test_torch(benchmark, audio):
    def foo(data):
        t = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len, win_length=win_len)
        t(data)

    benchmark.group = group + f"{audio[0]}"
    benchmark.name = "torch"
    benchmark(foo, torch.from_numpy(audio[1]))


def test_ssqueezepy(benchmark, audio):
    # almost no different between if pyfftw is installed
    def foo(data):
        sp = ssqueezepy.stft(
            data,
            win_len=win_len,
            hop_len=hop_len,
            n_fft=n_fft,
            dtype="float32",
            window=scipy.signal.windows.hann(win_len),
        )
        np.power(np.sqrt(sp, out=sp), 2.0, out=sp)  # type: ignore

    os.environ["SSQ_PARALLEL"] = "1"
    benchmark.group = group + f"{audio[0]}"
    benchmark.name = "ssqueezepy"
    benchmark(foo, audio[1])
