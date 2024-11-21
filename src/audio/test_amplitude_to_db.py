import os
from pathlib import Path

import librosa
import numexpr
import numpy as np
import pytest
import torch
import torchaudio

dataset_dir = Path(__file__).parents[2] / "external/dataset-audio"
group = "Amplitude to DB: "
top_db = 100.0


def data_figures():
    if os.getenv("CI"):
        return [1]
    else:  # pragma: nocover
        return range(3, 6)


@pytest.fixture(params=data_figures(), scope="module")
def magnitude(request):
    rng = np.random.default_rng(1337)

    return request.param, rng.random((4096, 10**request.param), dtype=np.float32)


def test_numpy(benchmark, magnitude):
    def foo(amp):
        amp = np.maximum(amp, 1e-10)
        ref = amp.max()  # noqa: F841
        db = numexpr.evaluate("10 * log10(amp/ref)")
        threshold = db.max() - top_db
        db = np.maximum(db, threshold)

    benchmark.group = group + f"10^{magnitude[0]}"
    benchmark.name = "numpy"
    benchmark(foo, magnitude[1])


def test_librosa(benchmark, magnitude):
    def foo(amp):
        db = librosa.power_to_db(amp, ref=np.max, top_db=top_db, amin=1e-10)
        # normalize
        np.add(db, top_db, out=db)
        np.divide(db, top_db, out=db)

    benchmark.group = group + f"10^{magnitude[0]}"
    benchmark.name = "librosa"
    benchmark(foo, magnitude[1])


def test_torch(benchmark, magnitude):
    def foo(amp):
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

    benchmark.group = group + f"10^{magnitude[0]}"
    benchmark.name = "torch"
    benchmark(foo, torch.tensor(np.abs(magnitude[1]) ** 2, dtype=torch.float32))
