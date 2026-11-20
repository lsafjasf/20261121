namespace DownmixLab.Core.Analysis;

/// <summary>One monitoring segment over the source timeline.</summary>
public sealed record Segment(long StartFrame, long EndFrame, string Label)
{
    public double StartSeconds(int sampleRate) => (double)StartFrame / sampleRate;
    public double EndSeconds(int sampleRate) => (double)EndFrame / sampleRate;
}

/// <summary>Per-segment metrics for one downmix matrix.</summary>
public sealed class SegmentMetrics
{
    public string MatrixName { get; init; } = "";
    public Segment Segment { get; init; } = new(0, 0, "");

    // Stereo image
    public double Correlation { get; set; }          // Pearson r between Lt and Rt, -1..1
    public double MidSideRatioDb { get; set; }       // 20log10(midE/sideE); low => wide/decorrelated
    public double PanPosition { get; set; }          // -1 (left) .. +1 (right) by energy

    // Phase cancellation
    public double MonoSumLossDb { get; set; }        // level lost when summing Lt+Rt to mono
    public double CancelledSampleRatio { get; set; } // fraction of samples where |Lt+Rt| < 0.5*(|Lt|+|Rt|)

    // Center dialogue audibility
    public double CenterGainLtDb { get; set; }       // matrix C->Lt gain in dB
    public double CenterGainRtDb { get; set; }       // matrix C->Rt gain in dB

    // Band energies in dBFS (Lt/Rt combined), keyed by band label e.g. "1k"
    public Dictionary<string, double> BandEnergyDb { get; } = new();

    public double PeakLt { get; set; }
    public double PeakRt { get; set; }
}

/// <summary>A/B comparison of the two matrices for one segment.</summary>
public sealed class SegmentComparison
{
    public Segment Segment { get; init; } = new(0, 0, "");
    public SegmentMetrics A { get; init; } = new();
    public SegmentMetrics B { get; init; } = new();

    public double CorrelationDelta => A.Correlation - B.Correlation;
    public double MonoSumLossDeltaDb => A.MonoSumLossDb - B.MonoSumLossDb;

    public Dictionary<string, double> BandEnergyDeltaDb()
    {
        var d = new Dictionary<string, double>();
        foreach (var kv in A.BandEnergyDb)
            if (B.BandEnergyDb.TryGetValue(kv.Key, out var b))
                d[kv.Key] = kv.Value - b;
        return d;
    }
}
