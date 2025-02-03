import time

import util

audio_file = util.audio_file
t1 = time.perf_counter()
import torch  # noqa: E402
import torchaudio  # noqa: E402

t1 = time.perf_counter() - t1
print(f"import: {t1}")

t2 = time.perf_counter()
wave, sr = torchaudio.load(audio_file, backend="soundfile")
if len(wave.shape) > 1:
    wave = wave.mean(0)
t2 = time.perf_counter() - t2
print(f"load: {t2}")

t3 = time.perf_counter()
transform = torchaudio.transforms.Spectrogram(n_fft=2048, win_length=2048, hop_length=1024)
amp: torch.Tensor = transform(wave)
t3 = time.perf_counter() - t3
print(f"spek: {t3}")
print(amp)
t4 = time.perf_counter()
db = torchaudio.functional.amplitude_to_DB(
    amp,
    multiplier=10.0,
    top_db=100,
    amin=1e-10,
    db_multiplier=torch.log10(torch.max(amp)),  # type: ignore
)
t4 = time.perf_counter() - t4
print(f"db: {t4}")

print(f"all: {t1 + t2 + t3 + t4}")
print(db)
print(db.shape)
