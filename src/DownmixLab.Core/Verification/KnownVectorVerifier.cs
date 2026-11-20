using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;

namespace DownmixLab.Core.Verification;

/// <summary>
/// Per-sample verification of the downmix engine against analytically known vectors.
/// Each case builds a 5.1 input whose expected stereo output is computable by hand,
/// then compares every output sample within a tight tolerance.
/// </summary>
public static class KnownVectorVerifier
{
    public sealed record CaseResult(string Name, bool Passed, double MaxAbsError, long SamplesChecked);

    public static IReadOnlyList<CaseResult> RunAll() => RunCases();

    private static IReadOnlyList<CaseResult> RunCases()
    {
        var results = new List<CaseResult>();
        var matrix = DownmixMatrix.ItuStereo();
        var policy = new LfePolicy { Mode = LfeInclusionMode.Exclude };

        // Case 1: unit impulse per input channel -> output equals the matrix row.
        for (int ch = 0; ch < 6; ch++)
        {
            var src = new AudioBuffer(48000, 6, 4);
            src.Data[ch][2] = 1f;
            var output = DownmixEngine.Downmix(src, matrix, policy);
            double err = 0;
            for (long n = 0; n < 4; n++)
            {
                double expL = n == 2 ? matrix[ch, 0] : 0;
                double expR = n == 2 ? matrix[ch, 1] : 0;
                err = Math.Max(err, Math.Abs(output.Data[0][n] - expL));
                err = Math.Max(err, Math.Abs(output.Data[1][n] - expR));
            }
            results.Add(new CaseResult($"impulse-ch{ch}-{DownmixMatrix.InputNames[ch]}",
                err < 1e-6, err, 8));
        }

        // Case 2: LFE include vs exclude with a constant 0.5 LFE signal.
        {
            var src = new AudioBuffer(48000, 6, 16);
            Array.Fill(src.Data[3], 0.5f);
            var m = DownmixMatrix.ItuStereo();
            m[3, 0] = 0.7; m[3, 1] = 0.7;
            var excluded = DownmixEngine.Downmix(src, m, new LfePolicy { Mode = LfeInclusionMode.Exclude });
            var included = DownmixEngine.Downmix(src, m, new LfePolicy { Mode = LfeInclusionMode.Include });
            double errEx = 0, errIn = 0;
            for (long n = 0; n < 16; n++)
            {
                errEx = Math.Max(errEx, Math.Abs(excluded.Data[0][n] - 0.0));
                errIn = Math.Max(errIn, Math.Abs(included.Data[0][n] - 0.35));
                errIn = Math.Max(errIn, Math.Abs(included.Data[1][n] - 0.35));
            }
            results.Add(new CaseResult("lfe-excluded-zero", errEx < 1e-6, errEx, 16));
            results.Add(new CaseResult("lfe-included-0.35", errIn < 1e-6, errIn, 32));
        }

        // Case 3: phase-inverted surrounds null in the mono sum.
        {
            var src = new AudioBuffer(48000, 6, 32);
            for (int n = 0; n < 32; n++)
            {
                src.Data[4][n] = 0.25f;   // Ls
                src.Data[5][n] = -0.25f;  // Rs
            }
            var m = DownmixMatrix.ItuStereo(); // Ls->Lt only, Rs->Rt only
            var output = DownmixEngine.Downmix(src, m, policy);
            double err = 0;
            double s = Math.Pow(10, -3.0 / 20.0);
            for (long n = 0; n < 32; n++)
            {
                err = Math.Max(err, Math.Abs(output.Data[0][n] - 0.25 * s));
                err = Math.Max(err, Math.Abs(output.Data[1][n] - (-0.25 * s)));
                err = Math.Max(err, Math.Abs(output.Data[0][n] + output.Data[1][n]));
            }
            results.Add(new CaseResult("inverted-surrounds-mono-null", err < 1e-6, err, 96));
        }

        // Case 4: center-only dialogue lands equally on Lt and Rt.
        {
            var src = new AudioBuffer(48000, 6, 8);
            for (int n = 0; n < 8; n++)
                src.Data[2][n] = (float)Math.Sin(n * 0.3);
            var m = DownmixMatrix.DialogueFriendly();
            var output = DownmixEngine.Downmix(src, m, policy);
            double err = 0;
            for (long n = 0; n < 8; n++)
            {
                err = Math.Max(err, Math.Abs(output.Data[0][n] - src.Data[2][n]));
                err = Math.Max(err, Math.Abs(output.Data[1][n] - src.Data[2][n]));
            }
            results.Add(new CaseResult("center-dialogue-both-sides", err < 1e-6, err, 16));
        }

        return results;
    }
}
