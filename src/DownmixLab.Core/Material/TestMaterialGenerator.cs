using DownmixLab.Core.Audio;

namespace DownmixLab.Core.Material;

/// <summary>
/// Built-in 5.1 test material with three monitoring segments:
///  1. Center-only dialogue-like passage (speech-flavored AM tone bursts).
///  2. Phase-inverted surrounds (Ls = -Rs) to expose mono fold-down cancellation.
///  3. Full mix: dialogue on C, music bed on L/R, inverted surrounds, LFE rumble.
/// </summary>
public static class TestMaterialGenerator
{
    public const int DefaultSampleRate = 48000;

    public static AudioBuffer Generate(int sampleRate = DefaultSampleRate, double secondsPerSegment = 2.0)
    {
        int segFrames = (int)(sampleRate * secondsPerSegment);
        var buffer = new AudioBuffer(sampleRate, 6, segFrames * 3L);
        var (l, r, c, lfe, ls, rs) =
            (buffer.Data[0], buffer.Data[1], buffer.Data[2], buffer.Data[3], buffer.Data[4], buffer.Data[5]);

        // Segment 1: center-only dialogue (formant-ish carrier with 4 Hz syllable gating)
        for (int n = 0; n < segFrames; n++)
        {
            double t = (double)n / sampleRate;
            double gate = 0.5 + 0.5 * Math.Sin(2 * Math.PI * 4 * t);
            gate = gate * gate;
            double voice = 0.45 * Math.Sin(2 * Math.PI * 220 * t)
                         + 0.25 * Math.Sin(2 * Math.PI * 440 * t)
                         + 0.12 * Math.Sin(2 * Math.PI * 880 * t);
            c[n] = (float)(0.6 * gate * voice);
        }

        // Segment 2: phase-inverted surrounds, Ls = -Rs (decorrelated noise burst pair)
        var rng = new Random(20261121);
        for (int n = 0; n < segFrames; n++)
        {
            double t = (double)n / sampleRate;
            double env = Math.Min(1.0, Math.Min(n, segFrames - n) / (0.02 * sampleRate));
            double noise = (rng.NextDouble() * 2 - 1) * 0.5
                         + 0.3 * Math.Sin(2 * Math.PI * 500 * t);
            ls[segFrames + n] = (float)(0.5 * env * noise);
            rs[segFrames + n] = (float)(-0.5 * env * noise);
        }

        // Segment 3: full mix
        for (int n = 0; n < segFrames; n++)
        {
            double t = (double)n / sampleRate;
            long o = 2L * segFrames + n;
            l[o] = (float)(0.35 * Math.Sin(2 * Math.PI * 330 * t));
            r[o] = (float)(0.35 * Math.Sin(2 * Math.PI * 392 * t));
            double gate = 0.5 + 0.5 * Math.Sin(2 * Math.PI * 4 * t);
            c[o] = (float)(0.5 * gate * gate * Math.Sin(2 * Math.PI * 250 * t));
            lfe[o] = (float)(0.4 * Math.Sin(2 * Math.PI * 45 * t));
            double noise = 0.3 * Math.Sin(2 * Math.PI * 700 * t);
            ls[o] = (float)(0.4 * noise);
            rs[o] = (float)(-0.4 * noise);
        }

        return buffer;
    }

    public static IReadOnlyList<Analysis.Segment> DefaultSegments(int sampleRate = DefaultSampleRate,
        double secondsPerSegment = 2.0) => new[]
    {
        new Analysis.Segment(0, (long)(sampleRate * secondsPerSegment), "Center dialogue only"),
        new Analysis.Segment((long)(sampleRate * secondsPerSegment), (long)(2 * sampleRate * secondsPerSegment),
            "Phase-inverted surrounds"),
        new Analysis.Segment((long)(2 * sampleRate * secondsPerSegment), (long)(3 * sampleRate * secondsPerSegment),
            "Full mix"),
    };
}
