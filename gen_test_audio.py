"""Generate test recordings for the dual-line mixer.

Timeline (seconds), sample rate 16 kHz, mono int16:
  0.0 - 2.0   normal silence (tiny noise floor)
  2.0 - 5.0   constant 1 kHz pilot tone
  5.0 - 7.0   main line damaged: freeze / dropout / garbage bursts
  7.0 - 8.0   both lines damaged (main garbage, backup dropout)
  8.0 - 10.0  main recovered, pilot continues

The backup line carries identical content delayed by BACKUP_DELAY_S,
with its own independent damage during 7.0 - 8.0 s.
"""
import numpy as np
import wave
import os

SR = 16000
DURATION_S = 10.0
PILOT_FREQ = 1000.0
PILOT_AMP = 0.5
NOISE_FLOOR = 0.002
BACKUP_DELAY_S = 0.040

OUT_DIR = "artifacts"


def content_timeline(n, rng):
    """The 'true' programme content both lines try to carry."""
    t = np.arange(n) / SR
    x = rng.normal(0.0, NOISE_FLOOR, n)
    pilot = PILOT_AMP * np.sin(2 * np.pi * PILOT_FREQ * t)
    pilot_on = t >= 2.0
    x[pilot_on] = pilot[pilot_on] + x[pilot_on]
    return x


def damage_main(x, rng):
    """Main line: 5.0-5.5 freeze, 5.5-6.2 dropout, 6.2-8.0 garbage."""
    y = x.copy()
    i5 = int(5.0 * SR)
    buf = x[i5 - 320:i5].copy()
    end = int(5.5 * SR)
    reps = (end - i5 + len(buf) - 1) // len(buf)
    y[i5:end] = np.tile(buf, reps)[:end - i5]
    y[int(5.5 * SR):int(6.2 * SR)] = 0.0
    g0, g1 = int(6.2 * SR), int(8.0 * SR)
    y[g0:g1] = np.clip(rng.normal(0, 0.9, g1 - g0), -1, 1)
    return y


def damage_backup(x, rng):
    """Backup line: 7.0-8.0 dropout (overlapping main garbage)."""
    y = x.copy()
    y[int(7.0 * SR):int(8.0 * SR)] = 0.0
    return y


def write_wav(path, x):
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(20261121)
    n = int(DURATION_S * SR)
    content = content_timeline(n, rng)

    delay = int(BACKUP_DELAY_S * SR)
    main_line = damage_main(content, rng)
    backup_line = damage_backup(np.concatenate([np.zeros(delay), content[:-delay]]), rng)

    write_wav(os.path.join(OUT_DIR, "main.wav"), main_line)
    write_wav(os.path.join(OUT_DIR, "backup.wav"), backup_line)
    print(f"wrote {OUT_DIR}/main.wav and {OUT_DIR}/backup.wav "
          f"({DURATION_S}s @ {SR} Hz, backup delay {BACKUP_DELAY_S*1000:.0f} ms)")


if __name__ == "__main__":
    main()
