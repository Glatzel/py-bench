import time

import util

audio_file = util.audio_file
t1 = time.perf_counter()
import os  # noqa: E402

import numexpr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import ssqueezepy  # noqa: E402
from scipy.signal.windows import hann  # noqa: E402

t1 = time.perf_counter() - t1
print(f"import: {t1}")

t2 = time.perf_counter()
wave, sr = soundfile.read(audio_file, dtype="float32")
if len(wave.shape) > 1:
    wave = wave.mean(1)
t2 = time.perf_counter() - t2
print(f"load: {t2}")

t3 = time.perf_counter()
os.environ["SSQ_PARALLEL"] = "1"
amp = ssqueezepy.stft(
    wave,
    win_len=2048,
    hop_len=1024,
    n_fft=2048,
    dtype="float32",
    window=hann(2048),
)
amp = np.abs(amp)  # type: ignore
np.power(amp, 2.0, out=amp)
t3 = time.perf_counter() - t3
print(f"spek: {t3}")

t4 = time.perf_counter()

amp = np.maximum(amp, 1e-10)
ref = amp.max()
db = numexpr.evaluate("10 * log10(amp/ref)")
threshold = db.max() - 100
db = np.maximum(db, threshold)

t4 = time.perf_counter() - t4
print(f"db: {t4}")

print(f"all: {t1 + t2 + t3 + t4}")
print(db)
print(db.shape)
