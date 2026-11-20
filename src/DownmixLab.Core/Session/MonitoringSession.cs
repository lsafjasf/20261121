using DownmixLab.Core.Analysis;
using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;

namespace DownmixLab.Core.Session;

/// <summary>
/// Orchestrates a full A/B monitoring pass: downmix a 5.1 source with two user
/// matrices, analyze each segment, and hold the stereo outputs for export.
/// </summary>
public sealed class MonitoringSession
{
    public AudioBuffer Source { get; }
    public DownmixMatrix MatrixA { get; }
    public DownmixMatrix MatrixB { get; }
    public LfePolicy LfePolicy { get; }

    public AudioBuffer? MonitorA { get; private set; }
    public AudioBuffer? MonitorB { get; private set; }
    public List<SegmentComparison> Comparisons { get; } = new();

    public MonitoringSession(AudioBuffer source, DownmixMatrix matrixA, DownmixMatrix matrixB,
        LfePolicy? lfePolicy = null)
    {
        Source = source;
        MatrixA = matrixA;
        MatrixB = matrixB;
        LfePolicy = lfePolicy ?? new LfePolicy();
    }

    public IReadOnlyList<SegmentComparison> Run(IReadOnlyList<Segment> segments)
    {
        MonitorA = DownmixEngine.Downmix(Source, MatrixA, LfePolicy);
        MonitorB = DownmixEngine.Downmix(Source, MatrixB, LfePolicy);

        var effA = DownmixEngine.ApplyLfePolicy(MatrixA, LfePolicy);
        var effB = DownmixEngine.ApplyLfePolicy(MatrixB, LfePolicy);
        var analyzer = new SegmentAnalyzer();

        Comparisons.Clear();
        foreach (var segment in segments)
        {
            Comparisons.Add(new SegmentComparison
            {
                Segment = segment,
                A = analyzer.Analyze(MonitorA, segment, "A", effA),
                B = analyzer.Analyze(MonitorB, segment, "B", effB),
            });
        }
        return Comparisons;
    }

    /// <summary>Splits the source into N equal monitoring segments.</summary>
    public static List<Segment> EqualSegments(AudioBuffer source, int count)
    {
        var segments = new List<Segment>();
        long step = source.Frames / count;
        for (int i = 0; i < count; i++)
        {
            long start = i * step;
            long end = i == count - 1 ? source.Frames : (i + 1) * step;
            segments.Add(new Segment(start, end, $"Segment {i + 1}"));
        }
        return segments;
    }
}
