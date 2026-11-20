using DownmixLab.Core.Analysis;
using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;
using DownmixLab.Core.Material;
using DownmixLab.Core.Session;
using DownmixLab.Core.Snapshot;
using DownmixLab.Core.Verification;

return args.Length == 0 ? Usage() : args[0] switch
{
    "verify" => Verify(),
    "gen-material" => GenMaterial(args),
    "analyze" => Analyze(args),
    _ => Usage(),
};

static int Usage()
{
    Console.WriteLine("""
        DownmixLab CLI
          verify                                   run per-sample known-vector checks
          gen-material <out.wav>                   write built-in 5.1 test material
          analyze <in.wav> --matrix-a "<12 nums>" --matrix-b "<12 nums>"
                  [--lfe exclude|include|attenuate:<dB>] [--segments N]
                  [--out <dir>] [--snapshot <file.json>]
        Matrix numbers are row-major L,R,C,LFE,Ls,Rs x Lt,Rt (12 values).
        """);
    return 1;
}

static int Verify()
{
    var results = KnownVectorVerifier.RunAll();
    bool all = true;
    foreach (var r in results)
    {
        Console.WriteLine($"{(r.Passed ? "PASS" : "FAIL")}  {r.Name,-36} maxErr={r.MaxAbsError:E3}  samples={r.SamplesChecked}");
        all &= r.Passed;
    }
    Console.WriteLine(all ? "All known vectors verified." : "Verification FAILED.");
    return all ? 0 : 2;
}

static int GenMaterial(string[] args)
{
    if (args.Length < 2) return Usage();
    var buffer = TestMaterialGenerator.Generate();
    WavWriter.Write(args[1], buffer, WavSampleFormat.Pcm24, writeBext: true);
    Console.WriteLine($"Wrote {args[1]} ({buffer.Channels}ch {buffer.SampleRate} Hz, {buffer.Frames} frames, BWF)");
    return 0;
}

static int Analyze(string[] args)
{
    if (args.Length < 2) return Usage();
    string input = args[1];
    string? matrixAText = Opt("--matrix-a"), matrixBText = Opt("--matrix-b");
    string lfeText = Opt("--lfe") ?? "exclude";
    int segmentCount = int.TryParse(Opt("--segments"), out var sc) ? sc : 4;
    string? outDir = Opt("--out");
    string? snapshotPath = Opt("--snapshot");

    var wav = WavFile.Read(input);
    Console.WriteLine($"Loaded {input}: {wav.Buffer.Channels}ch {wav.Buffer.SampleRate} Hz " +
        $"{wav.Buffer.Frames} frames, {wav.Format}{(wav.IsBwf ? ", BWF" : "")}");
    if (wav.Buffer.Channels != 6)
    {
        Console.Error.WriteLine("Input must be 5.1 (6 channels, order L R C LFE Ls Rs).");
        return 3;
    }

    var matrixA = matrixAText != null ? DownmixMatrix.Parse(matrixAText) : DownmixMatrix.ItuStereo();
    var matrixB = matrixBText != null ? DownmixMatrix.Parse(matrixBText) : DownmixMatrix.DialogueFriendly();
    var policy = ParseLfe(lfeText);

    var session = new MonitoringSession(wav.Buffer, matrixA, matrixB, policy);
    var segments = MonitoringSession.EqualSegments(wav.Buffer, segmentCount);
    var comparisons = session.Run(segments);

    Console.WriteLine($"LFE policy: {policy}");
    foreach (var cmp in comparisons)
        PrintComparison(cmp, wav.Buffer.SampleRate);

    if (outDir != null)
    {
        Directory.CreateDirectory(outDir);
        string a = Path.Combine(outDir, "monitor_A.wav");
        string b = Path.Combine(outDir, "monitor_B.wav");
        WavWriter.Write(a, session.MonitorA!, WavSampleFormat.Pcm24);
        WavWriter.Write(b, session.MonitorB!, WavSampleFormat.Pcm24);
        Console.WriteLine($"Exported {a} and {b}");
    }
    if (snapshotPath != null)
    {
        MatrixSnapshot.Capture(session, input).Save(snapshotPath);
        Console.WriteLine($"Snapshot saved to {snapshotPath}");
    }
    return 0;

    string? Opt(string name)
    {
        for (int i = 2; i < args.Length - 1; i++)
            if (args[i] == name) return args[i + 1];
        return null;
    }
}

static LfePolicy ParseLfe(string text)
{
    if (text.Equals("include", StringComparison.OrdinalIgnoreCase))
        return new LfePolicy { Mode = LfeInclusionMode.Include };
    if (text.StartsWith("attenuate:", StringComparison.OrdinalIgnoreCase) &&
        double.TryParse(text[10..], out double db))
        return new LfePolicy { Mode = LfeInclusionMode.Attenuate, AttenuationDb = db };
    return new LfePolicy { Mode = LfeInclusionMode.Exclude };
}

static void PrintComparison(SegmentComparison cmp, int sampleRate)
{
    Console.WriteLine($"\n== {cmp.Segment.Label} " +
        $"({cmp.Segment.StartSeconds(sampleRate):0.00}s - {cmp.Segment.EndSeconds(sampleRate):0.00}s) ==");
    Console.WriteLine($"{"metric",-24}{"A",12}{"B",12}{"A-B",12}");
    Row("correlation", cmp.A.Correlation, cmp.B.Correlation);
    Row("mid/side dB", cmp.A.MidSideRatioDb, cmp.B.MidSideRatioDb);
    Row("pan (-1..1)", cmp.A.PanPosition, cmp.B.PanPosition);
    Row("mono sum loss dB", cmp.A.MonoSumLossDb, cmp.B.MonoSumLossDb);
    Row("cancelled ratio", cmp.A.CancelledSampleRatio, cmp.B.CancelledSampleRatio);
    Row("center->Lt dB", cmp.A.CenterGainLtDb, cmp.B.CenterGainLtDb);
    Row("center->Rt dB", cmp.A.CenterGainRtDb, cmp.B.CenterGainRtDb);
    var delta = cmp.BandEnergyDeltaDb();
    foreach (var kv in cmp.A.BandEnergyDb)
        Row($"band {kv.Key} Hz dB", kv.Value, cmp.B.BandEnergyDb[kv.Key]);

    static void Row(string name, double a, double b) =>
        Console.WriteLine($"{name,-24}{a,12:F3}{b,12:F3}{a - b,12:F3}");
}
