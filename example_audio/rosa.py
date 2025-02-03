import time

import util

audio_file = util.audio_file
t1 = time.perf_counter()
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pyfftw  # noqa: E402
import soundfile  # noqa: E402

t1 = time.perf_counter() - t1
print(f"import: {t1}")

t2 = time.perf_counter()
wave, sr = soundfile.read(audio_file, dtype="float32")
if len(wave.shape) > 1:
    wave = wave.mean(1)
t2 = time.perf_counter() - t2
print(f"load: {t2}")

t3 = time.perf_counter()
librosa.set_fftlib(pyfftw.interfaces.numpy_fft)
amp = librosa.stft(wave, n_fft=2048, hop_length=1024, win_length=2048, dtype="complex64")
amp = np.abs(amp)
np.power(amp, 2.0, out=amp)
t3 = time.perf_counter() - t3
print(f"spek: {t3}")
print(f"spek: {amp.shape}")

t4 = time.perf_counter()
db = librosa.power_to_db(
    amp,
    ref=np.max,
    top_db=100,
    amin=1e-10,
)
t4 = time.perf_counter() - t4
print(f"db: {t4}")

print(f"all: {t1 + t2 + t3 + t4}")
print(db.shape)
