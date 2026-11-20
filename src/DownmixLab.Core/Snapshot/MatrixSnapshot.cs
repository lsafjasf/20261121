using System.Text.Json;
using System.Text.Json.Serialization;
using DownmixLab.Core.Analysis;
using DownmixLab.Core.Downmix;

namespace DownmixLab.Core.Snapshot;

/// <summary>JSON snapshot of both matrices, the LFE policy and per-segment metrics.</summary>
public sealed class MatrixSnapshot
{
    public string Tool { get; set; } = "DownmixLab";
    public string Version { get; set; } = "1.0";
    public DateTimeOffset CreatedUtc { get; set; } = DateTimeOffset.UtcNow;
    public string? SourceFile { get; set; }
    public int SampleRate { get; set; }
    public double[][] MatrixA { get; set; } = Array.Empty<double[]>();
    public double[][] MatrixB { get; set; } = Array.Empty<double[]>();
    public string LfePolicy { get; set; } = "";
    public List<SegmentMetricsSnapshot> Segments { get; set; } = new();

    public sealed class SegmentMetricsSnapshot
    {
        public string Label { get; set; } = "";
        public double StartSeconds { get; set; }
        public double EndSeconds { get; set; }
        public SegmentMetrics? A { get; set; }
        public SegmentMetrics? B { get; set; }
    }

    private static double[][] ToJagged(DownmixMatrix m) => Enumerable.Range(0, 6)
        .Select(i => new[] { m[i, 0], m[i, 1] }).ToArray();

    public static MatrixSnapshot Capture(Session.MonitoringSession session, string? sourceFile)
    {
        var snapshot = new MatrixSnapshot
        {
            SourceFile = sourceFile,
            SampleRate = session.Source.SampleRate,
            MatrixA = ToJagged(session.MatrixA),
            MatrixB = ToJagged(session.MatrixB),
            LfePolicy = session.LfePolicy.ToString() ?? "",
        };
        foreach (var comparison in session.Comparisons)
        {
            snapshot.Segments.Add(new SegmentMetricsSnapshot
            {
                Label = comparison.Segment.Label,
                StartSeconds = comparison.Segment.StartSeconds(session.Source.SampleRate),
                EndSeconds = comparison.Segment.EndSeconds(session.Source.SampleRate),
                A = comparison.A,
                B = comparison.B,
            });
        }
        return snapshot;
    }

    public void Save(string path)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(this, options));
    }
}
