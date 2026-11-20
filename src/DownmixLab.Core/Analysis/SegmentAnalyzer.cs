using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;

namespace DownmixLab.Core.Analysis;

/// <summary>
/// Computes stereo image, band energy and phase-cancellation metrics
/// per monitoring segment for a stereo (downmixed) buffer.
/// </summary>
public sealed class SegmentAnalyzer
{
    public static readonly (string Label, double LowHz, double HighHz)[] Bands =
    {
        ("125", 90, 180), ("250", 180, 355), ("500", 355, 710),
        ("1k", 710, 1400), ("2k", 1400, 2800), ("4k", 2800, 5600),
        ("8k", 5600, 11200), ("16k", 11200, 20000),
    };

    private readonly int _fftSize;

    public SegmentAnalyzer(int fftSize = 4096)
    {
        if ((fftSize & (fftSize - 1)) != 0)
            throw new ArgumentException("FFT size must be a power of two.");
        _fftSize = fftSize;
    }

    public SegmentMetrics Analyze(AudioBuffer stereo, Segment segment, string matrixName,
        DownmixMatrix effectiveMatrix)
    {
        if (stereo.Channels != 2)
            throw new ArgumentException("Analyzer expects a stereo buffer.");

        var lt = stereo.Data[0];
        var rt = stereo.Data[1];
        long start = Math.Clamp(segment.StartFrame, 0, stereo.Frames);
        long end = Math.Clamp(segment.EndFrame, start, stereo.Frames);
        long count = Math.Max(end - start, 1);

        var metrics = new SegmentMetrics
        {
            MatrixName = matrixName,
            Segment = new Segment(start, end, segment.Label),
            CenterGainLtDb = ToDb(Math.Abs(effectiveMatrix[2, 0])),
            CenterGainRtDb = ToDb(Math.Abs(effectiveMatrix[2, 1])),
        };

        double sumLt = 0, sumRt = 0, sumLtRt = 0, sumLt2 = 0, sumRt2 = 0;
        double midE = 0, sideE = 0, monoE = 0, sumAbs = 0;
        long cancelled = 0;
        double peakL = 0, peakR = 0;

        for (long n = start; n < end; n++)
        {
            double l = lt[n], r = rt[n];
            sumLt += l; sumRt += r;
            sumLtRt += l * r;
            sumLt2 += l * l; sumRt2 += r * r;
            double mid = 0.5 * (l + r);
            double side = 0.5 * (l - r);
            midE += mid * mid;
            sideE += side * side;
            monoE += mid * mid;
            sumAbs += 0.5 * (l * l + r * r);
            if (Math.Abs(l + r) < 0.5 * (Math.Abs(l) + Math.Abs(r)) && (l != 0 || r != 0))
                cancelled++;
            peakL = Math.Max(peakL, Math.Abs(l));
            peakR = Math.Max(peakR, Math.Abs(r));
        }

        double denom = Math.Sqrt((count * sumLt2 - sumLt * sumLt) * (count * sumRt2 - sumRt * sumRt));
        metrics.Correlation = denom > 0 ? (count * sumLtRt - sumLt * sumRt) / denom : 0;
        metrics.MidSideRatioDb = ToDb(Math.Sqrt(midE / Math.Max(sideE, 1e-12)));
        double eL = Math.Sqrt(sumLt2 / count), eR = Math.Sqrt(sumRt2 / count);
        metrics.PanPosition = (eL + eR) > 1e-9 ? (eR - eL) / (eR + eL) : 0;
        metrics.MonoSumLossDb = ToDb(Math.Sqrt(sumAbs / count)) - ToDb(Math.Sqrt(monoE / count));
        metrics.CancelledSampleRatio = (double)cancelled / count;
        metrics.PeakLt = peakL;
        metrics.PeakRt = peakR;

        ComputeBandEnergies(stereo, start, end, metrics);
        return metrics;
    }

    private void ComputeBandEnergies(AudioBuffer stereo, long start, long end,
        SegmentMetrics metrics)
    {
        var magL = new double[_fftSize / 2];
        var magR = new double[_fftSize / 2];
        var bandSum = new double[Bands.Length];
        int hop = _fftSize / 2;
        int windows = 0;

        for (long offset = start; offset + _fftSize <= end; offset += hop)
        {
            Fft.MagnitudeSpectrum(stereo.Data[0], (int)offset, _fftSize, magL);
            Fft.MagnitudeSpectrum(stereo.Data[1], (int)offset, _fftSize, magR);
            double binHz = (double)stereo.SampleRate / _fftSize;
            for (int b = 0; b < Bands.Length; b++)
            {
                int lo = Math.Max(1, (int)(Bands[b].LowHz / binHz));
                int hi = Math.Min(magL.Length - 1, (int)Math.Ceiling(Bands[b].HighHz / binHz));
                double e = 0;
                for (int k = lo; k <= hi; k++)
                    e += magL[k] * magL[k] + magR[k] * magR[k];
                bandSum[b] += e;
            }
            windows++;
        }

        for (int b = 0; b < Bands.Length; b++)
        {
            double mean = windows > 0 ? bandSum[b] / windows : 0;
            metrics.BandEnergyDb[Bands[b].Label] = ToDb(Math.Sqrt(mean));
        }
    }

    private static double ToDb(double linear) =>
        linear <= 1e-9 ? -180.0 : 20 * Math.Log10(linear);
}
