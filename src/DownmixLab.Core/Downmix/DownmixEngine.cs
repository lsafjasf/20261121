using DownmixLab.Core.Audio;

namespace DownmixLab.Core.Downmix;

/// <summary>
/// Applies a 6x2 matrix to a 5.1 buffer. The LFE row is pre-scaled by the LFE policy,
/// then Lt[n] = sum_i in_i[n] * M[i,0], Rt[n] = sum_i in_i[n] * M[i,1] per sample.
/// </summary>
public static class DownmixEngine
{
    public const int LfeRow = 3;

    /// <summary>Returns an effective copy of the matrix with the LFE policy applied.</summary>
    public static DownmixMatrix ApplyLfePolicy(DownmixMatrix matrix, LfePolicy policy)
    {
        var effective = matrix.Clone();
        double g = policy.GainFactor;
        effective[LfeRow, 0] *= g;
        effective[LfeRow, 1] *= g;
        return effective;
    }

    public static AudioBuffer Downmix(AudioBuffer source, DownmixMatrix matrix, LfePolicy policy)
    {
        if (source.Channels != DownmixMatrix.InputChannels)
            throw new ArgumentException(
                $"Expected 6 channels (5.1), got {source.Channels}.", nameof(source));

        var m = ApplyLfePolicy(matrix, policy);
        var output = new AudioBuffer(source.SampleRate, DownmixMatrix.OutputChannels, source.Frames);
        var lt = output.Data[0];
        var rt = output.Data[1];
        var src = source.Data;

        for (long n = 0; n < source.Frames; n++)
        {
            double l = 0, r = 0;
            for (int i = 0; i < DownmixMatrix.InputChannels; i++)
            {
                double s = src[i][n];
                l += s * m[i, 0];
                r += s * m[i, 1];
            }
            lt[n] = (float)l;
            rt[n] = (float)r;
        }
        return output;
    }

    /// <summary>Peak gain the matrix can produce per output (for headroom checks).</summary>
    public static (double Lt, double Rt) WorstCaseGain(DownmixMatrix matrix, LfePolicy policy)
    {
        var m = ApplyLfePolicy(matrix, policy);
        double lt = 0, rt = 0;
        for (int i = 0; i < DownmixMatrix.InputChannels; i++)
        {
            lt += Math.Abs(m[i, 0]);
            rt += Math.Abs(m[i, 1]);
        }
        return (lt, rt);
    }
}
