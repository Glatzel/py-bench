import os
from pathlib import Path

import librosa
import numpy as np
import pyfftw
import pytest
import scipy
import scipy.signal
import ssqueezepy
import torch
import torchaudio

group = "STFT: "
n_fft = 1024
win_len = 1024
hop_len = 512
dataset_dir = Path(__file__).parents[2] / "external/dataset-audio"


def data_figures():
    if os.getenv("CI"):
        return [5]
    else:  # pragma: nocover
        return range(5, 9)


@pytest.fixture(params=data_figures(), scope="module")
def audio(request):
    rng = np.random.default_rng(1337)
    return request.param, rng.random((10**request.param), dtype=np.float32)


def test_scipy(benchmark, audio):
    def foo(data):
        scipy.signal.stft(data, noverlap=hop_len, nfft=n_fft, nperseg=win_len)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "scipy"
    benchmark(foo, audio[1])


def test_librosa(benchmark, audio):
    def foo(data):
        amp = librosa.stft(data, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
        amp = np.abs(amp)
        np.power(amp, 2.0, out=amp)

    librosa.set_fftlib(pyfftw.interfaces.numpy_fft)
    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "librosa"
    benchmark(foo, audio[1])


def test_torch(benchmark, audio):
    def foo(data):
        t = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len, win_length=win_len)
        t(data)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "torch"
    benchmark(foo, torch.from_numpy(audio[1]))


def test_ssqueezepy(benchmark, audio):
    # almost no different between if pyfftw is installed
    def foo(data):
        os.environ["SSQ_PARALLEL"] = "1"
        amp = ssqueezepy.stft(
            data,
            win_len=win_len,
            hop_len=hop_len,
            n_fft=n_fft,
            dtype="float32",
            window=scipy.signal.windows.hann(win_len),
        )
        amp = np.abs(amp)  # type: ignore
        np.power(amp, 2.0, out=amp)

    benchmark.group = group + f"10^{audio[0]}"
    benchmark.name = "ssqueezepy"
    benchmark(foo, audio[1])
