"""Exhibit-device scenario: ambient bed + narration + two back-to-back alerts.

Timeline (seconds):
  0.0  ambient bed starts and loops forever
  0.5  narration begins (visitor approaches the exhibit)
  2.0  simulated visitor sensor event -> alert #1 (1.8 s) interrupts
  3.8  alert #1 ends; sidechain releases, narration playhead resumes
  4.3  alert #2 fires while the previous release is still in progress
  6.1  alert #2 ends; full release back to the narration

Outputs are written under artifacts/:
  mix_output.wav, gain_curve.csv, gain_and_waveform.svg, report.txt
"""

from __future__ import annotations

import os
import sys

from .analysis import analyze, format_report
from .mixer import Channel, Mixer
from .recording import write_gain_csv, write_svg, write_wav
from .synthesis import alert_chirp, brown_noise, pad_signal, syllable_train

SAMPLE_RATE = 48000
BLOCK = 256
DURATION_S = 10.5

ALERT1_AT = 2.0
ALERT1_LEN = 1.8
ALERT2_AT = ALERT1_AT + ALERT1_LEN + 0.5  # 4.3 s, mid-release re-trigger
ALERT2_LEN = 1.8


def build_mixer() -> tuple[Mixer, list[float]]:
    total = int(DURATION_S * SAMPLE_RATE)
    mixer = Mixer(SAMPLE_RATE, BLOCK)

    ambient = brown_noise(total, SAMPLE_RATE, level=0.5, cutoff_hz=350)
    narration = syllable_train(total, SAMPLE_RATE, level=0.85)
    alert_a = alert_chirp(int(ALERT1_LEN * SAMPLE_RATE), SAMPLE_RATE)
    alert_b = alert_chirp(int(ALERT2_LEN * SAMPLE_RATE), SAMPLE_RATE,
                          f0=740.0, f1=1180.0)

    mixer.add_channel(Channel(
        name='ambient', priority=0, signal=ambient,
        base_gain=0.55, duck_floor=0.12,
        attack_ms=30.0, release_ms=700.0, always_active=True))
    mixer.add_channel(Channel(
        name='narration', priority=1,
        signal=pad_signal([], total), base_gain=0.9, duck_floor=0.08,
        attack_ms=18.0, release_ms=450.0, pause_when_ducked=True))
    mixer.add_channel(Channel(
        name='alert', priority=2,
        signal=pad_signal([], total), base_gain=1.0, duck_floor=1.0,
        attack_ms=5.0, release_ms=80.0))

    # Visitor proximity event: narration starts at 0.5 s.
    mixer.play_on('narration', narration, at_seconds=0.5)
    # Two consecutive fault announcements.
    mixer.play_on('alert', alert_a, at_seconds=ALERT1_AT)
    mixer.play_on('alert', alert_b, at_seconds=ALERT2_AT)
    return mixer, alert_a


def main(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    mixer, _ = build_mixer()
    total = int(DURATION_S * SAMPLE_RATE)
    output = mixer.render(total)

    peak_db = write_wav(os.path.join(out_dir, 'mix_output.wav'),
                        output, SAMPLE_RATE)
    write_gain_csv(os.path.join(out_dir, 'gain_curve.csv'), mixer)
    write_svg(os.path.join(out_dir, 'gain_and_waveform.svg'), mixer,
              output, SAMPLE_RATE,
              title='Sidechain gain curves and mixed output waveform')
    report = analyze(mixer, output, SAMPLE_RATE,
                     alert_starts=[ALERT1_AT, ALERT2_AT])
    report['wav_peak_dbfs'] = peak_db
    with open(os.path.join(out_dir, 'report.txt'), 'w') as fh:
        fh.write(format_report(report) + '\n')
    return report


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
    print(format_report(main(target)))
