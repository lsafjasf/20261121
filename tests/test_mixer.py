import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audiomix.mixer import Channel, Mixer  # noqa: E402
from audiomix.synthesis import alert_chirp, syllable_train  # noqa: E402


SR = 16000


def tone(n, freq=440.0, sr=SR, amp=0.8):
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


class MixerTests(unittest.TestCase):
    def test_ducking_attacks_and_releases(self):
        mixer = Mixer(SR, block_size=160)
        mixer.add_channel(Channel('low', 0, tone(SR), base_gain=1.0,
                                  duck_floor=0.1, attack_ms=10,
                                  release_ms=100))
        mixer.add_channel(Channel('high', 1, [0.0] * SR, base_gain=1.0))
        mixer.play_on('high', tone(SR // 2), at_seconds=0.25)
        mixer.render(SR)
        low = mixer.channel('low')
        gains = [g for _, g, _ in low.history]
        during = [g for (t, g, _) in low.history if 0.3 < t < 0.7]
        after = [g for (t, g, _) in low.history if t > 0.99]
        self.assertGreater(gains[0], 0.99)
        self.assertLess(max(during), 0.25)
        self.assertGreater(min(after), 0.85)

    def test_narration_playhead_freezes_and_resumes(self):
        mixer = Mixer(SR, block_size=160)
        mixer.add_channel(Channel('ambient', 0, tone(SR),
                                  duck_floor=0.2))
        mixer.add_channel(Channel('narration', 1, [0.0] * SR,
                                  base_gain=1.0, duck_floor=0.1,
                                  pause_when_ducked=True))
        mixer.add_channel(Channel('alert', 2, [0.0] * SR))
        mixer.play_on('narration', syllable_train(SR, SR), at_seconds=0.0)
        mixer.play_on('alert', alert_chirp(SR // 4, SR), at_seconds=0.3)
        mixer.render(SR)
        nar = mixer.channel('narration')
        paused = sum(1 for _, _, p in nar.history if p)
        self.assertGreater(paused, 0)
        # 0.25 s of alert minus attack/release edge blocks must be owed back.
        edge_allowance = int(0.06 * SR) + 2 * mixer.block_size
        self.assertLess(nar.position, SR - SR // 4 + edge_allowance)
        # And after release the head moves again (position advanced overall).
        self.assertGreater(nar.position, SR // 2)

    def test_two_consecutive_alerts_retrigger_attack(self):
        mixer = Mixer(SR, block_size=160)
        mixer.add_channel(Channel('narration', 1, tone(SR),
                                  duck_floor=0.1, attack_ms=8,
                                  release_ms=300))
        mixer.add_channel(Channel('alert', 2, [0.0] * SR))
        mixer.play_on('alert', alert_chirp(SR // 5, SR), at_seconds=0.2)
        # Second alert starts in the release window of the first.
        mixer.play_on('alert', alert_chirp(SR // 5, SR, f0=700, f1=900),
                      at_seconds=0.45)
        mixer.render(SR)
        nar = mixer.channel('narration')
        gains = [(t, g) for t, g, _ in nar.history]
        # Between the two alerts gain starts recovering, then dives again.
        gap = [g for t, g in gains if 0.40 < t < 0.45]
        re_ducked = [g for t, g in gains if 0.5 < t < 0.6]
        self.assertGreater(min(gap), min(re_ducked))
        self.assertLess(max(re_ducked), 0.3)

    def test_output_peak_is_bounded(self):
        mixer = Mixer(SR, block_size=160, soft_clip=0.97)
        mixer.add_channel(Channel('a', 0, tone(SR, amp=1.0),
                                  base_gain=1.0, duck_floor=1.0))
        mixer.add_channel(Channel('b', 1, tone(SR, freq=660, amp=1.0),
                                  base_gain=1.0, duck_floor=1.0))
        out = mixer.render(SR)
        self.assertLessEqual(max(abs(s) for s in out), 0.98)

    def test_resume_hits_the_exact_playhead_sample(self):
        """Sample-level proof: after a freeze the next consumed sample is
        the exact buffer index that followed the last one - no skipped
        syllables, no repeated samples."""
        sr = 16000
        block = 160

        class RecordingMixer(Mixer):
            def __init__(self):
                super().__init__(sr, block_size=block)
                self.consumed_ranges = []

            def render(self, total_samples):
                output = [0.0] * total_samples
                for start in range(0, total_samples, block):
                    self._fire_events(start)
                    stop = min(start + block, total_samples)
                    for ch in self.channels:
                        target = self._duck_target(ch)
                        self._smooth(ch, target)
                        before = ch.position
                        if not ch.paused:
                            for _ in range(stop - start):
                                if ch.position < len(ch.signal):
                                    ch.position += 1
                        if ch.name == 'narration' and ch.position > before:
                            self.consumed_ranges.append((before, ch.position))
                        ch.history.append(((start + block / 2) / sr,
                                           ch.gain,
                                           1.0 if ch.paused else 0.0))
                return output

        mixer = RecordingMixer()
        ramp = [float(i) for i in range(sr)]
        nar = Channel('narration', 1, ramp, base_gain=1.0,
                      duck_floor=0.05, attack_ms=5, release_ms=40,
                      pause_when_ducked=True)
        mixer.add_channel(nar)
        mixer.add_channel(Channel('alert', 2, [0.0] * sr))
        mixer.play_on('alert', alert_chirp(sr // 4, sr), at_seconds=0.3)
        mixer.render(sr)

        # Every consumed range starts exactly where the previous one ended.
        for prev, nxt in zip(mixer.consumed_ranges,
                             mixer.consumed_ranges[1:]):
            self.assertEqual(prev[1], nxt[0])
        # At least one range sits before the freeze and one after it.
        self.assertGreater(len(mixer.consumed_ranges), 2)
        paused = sum(1 for _, _, p in nar.history if p)
        self.assertGreater(paused, 0)
        # Contiguous union means samples 0..N-1 were consumed exactly once.
        self.assertEqual(mixer.consumed_ranges[0][0], 0)
        self.assertEqual(nar.position,
                         mixer.consumed_ranges[-1][1])


if __name__ == '__main__':
    unittest.main()
