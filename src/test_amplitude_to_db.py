import os
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torchaudio

dataset_dir = Path(__file__).parents[1] / "external/dataset-audio"
group = "Amplitude to DB: "
top_db = 100.0


def init_amplitude():
    n_fft: int
    win_len: int
    hop_len: int
    file: Path

    if os.getenv("CI"):
        file = dataset_dir / "two channel/ff-16b-2c-44100hz.wav"
        n_fft = 1024
        win_len = 1024
        hop_len = 512
    else:
        file = dataset_dir / "../Vision of Her/24-88.flac"
        n_fft = 4096
        win_len = 4096
        hop_len = 2048

    wave, _ = soundfile.read(file, dtype="float32")
    if len(wave.shape) > 1:
        wave = wave.mean(1)
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    p = transform(torch.from_numpy(wave))
    return file, p


def test_librosa(benchmark):
    def foo(amp):
        db = librosa.power_to_db(amp, ref=np.max, top_db=top_db, amin=1e-10)
        # normalize
        np.add(db, top_db, out=db)
        np.divide(db, top_db, out=db)

    file, amp = init_amplitude()
    amp = amp.numpy()
    benchmark.group = group + file.name
    benchmark.name = "librosa"
    benchmark(foo, amp)


def test_torch(benchmark):
    def foo(amp):
        torch.abs(amp, out=amp)
        db = torchaudio.functional.amplitude_to_DB(
            amp,
            multiplier=10.0,
            top_db=top_db,
            amin=1e-10,
            db_multiplier=torch.log10(torch.max(amp)),  # type: ignore
        )
        # normalize
        torch.add(db, top_db, out=db)
        torch.divide(db, top_db, out=db)  # type: ignore
        amp = amp.numpy()

    file, amp = init_amplitude()
    benchmark.group = group + file.name
    benchmark.name = "torch"
    benchmark(foo, amp)
