using DownmixLab.Core.Analysis;
using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;
using DownmixLab.Core.Material;
using DownmixLab.Core.Session;
using Xunit;

namespace DownmixLab.Tests;

public class WavAndAnalysisTests
{
    [Fact]
    public void WavRoundTripPreservesSamplesPerFrame()
    {
        var buffer = TestMaterialGenerator.Generate(48000, 0.1);
        string path = Path.Combine(Path.GetTempPath(), $"dl_test_{Guid.NewGuid():N}.wav");
        try
        {
            WavWriter.Write(path, buffer, WavSampleFormat.Float32, writeBext: true);
            var read = WavFile.Read(path);
            Assert.True(read.IsBwf);
            Assert.Equal(6, read.Buffer.Channels);
            Assert.Equal(buffer.Frames, read.Buffer.Frames);
            for (int c = 0; c < 6; c++)
                for (long n = 0; n < buffer.Frames; n += 97)
                    Assert.Equal(buffer.Data[c][n], read.Buffer.Data[c][n], 6);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Pcm24RoundTripWithinOneLsb()
    {
        var buffer = new AudioBuffer(48000, 6, 64);
        for (int c = 0; c < 6; c++)
            for (int n = 0; n < 64; n++)
                buffer.Data[c][n] = (float)(0.8 * Math.Sin(n * 0.11 + c));
        string path = Path.Combine(Path.GetTempPath(), $"dl_test_{Guid.NewGuid():N}.wav");
        try
        {
            WavWriter.Write(path, buffer, WavSampleFormat.Pcm24);
            var read = WavFile.Read(path);
            Assert.Equal(WavSampleFormat.Pcm24, read.Format);
            for (int c = 0; c < 6; c++)
                for (int n = 0; n < 64; n++)
                    Assert.True(Math.Abs(buffer.Data[c][n] - read.Buffer.Data[c][n]) < 2e-7,
                        "PCM24 round-trip must stay within ~1 LSB");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AnalyzerFlagsPhaseInvertedSurrounds()
    {
        var source = TestMaterialGenerator.Generate();
        var segments = TestMaterialGenerator.DefaultSegments();
        var session = new MonitoringSession(source,
            DownmixMatrix.ItuStereo(), DownmixMatrix.DialogueFriendly());
        var comparisons = session.Run(segments);

        var inverted = comparisons[1];
        Assert.True(inverted.A.Correlation < -0.99, "inverted surrounds must correlate near -1");
        Assert.True(inverted.A.CancelledSampleRatio > 0.99);
        Assert.True(inverted.A.MonoSumLossDb > 60, "mono fold-down must nearly null");

        var dialogue = comparisons[0];
        Assert.True(dialogue.A.Correlation > 0.99, "center dialogue must correlate near +1");
        Assert.Equal(0.0, dialogue.B.CenterGainLtDb, 3); // unity center gain on both sides
        Assert.Equal(0.0, dialogue.B.CenterGainRtDb, 3);
        Assert.Equal(-3.0, dialogue.A.CenterGainLtDb, 1);
    }

    [Fact]
    public void BandEnergyPeaksInExpectedBand()
    {
        // 1 kHz tone on both stereo channels -> "1k" band must dominate.
        var stereo = new AudioBuffer(48000, 2, 48000);
        for (int n = 0; n < 48000; n++)
        {
            float s = (float)(0.5 * Math.Sin(2 * Math.PI * 1000 * n / 48000.0));
            stereo.Data[0][n] = s;
            stereo.Data[1][n] = s;
        }
        var analyzer = new SegmentAnalyzer();
        var metrics = analyzer.Analyze(stereo, new Segment(0, 48000, "tone"),
            "A", DownmixMatrix.ItuStereo());
        var maxBand = metrics.BandEnergyDb.MaxBy(kv => kv.Value).Key;
        Assert.Equal("1k", maxBand);
    }
}
