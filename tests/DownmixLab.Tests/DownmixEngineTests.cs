using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;
using DownmixLab.Core.Verification;
using Xunit;

namespace DownmixLab.Tests;

public class DownmixEngineTests
{
    [Fact]
    public void AllKnownVectorsPass()
    {
        var results = KnownVectorVerifier.RunAll();
        Assert.All(results, r => Assert.True(r.Passed,
            $"{r.Name} failed with max error {r.MaxAbsError}"));
    }

    [Fact]
    public void ImpulseResponseEqualsMatrixRow()
    {
        var matrix = DownmixMatrix.Parse(
            "1 0  0 1  0.7071 0.7071  0.5 0.5  0.7071 0  0 0.7071");
        var policy = new LfePolicy { Mode = LfeInclusionMode.Include };
        for (int ch = 0; ch < 6; ch++)
        {
            var src = new AudioBuffer(48000, 6, 2);
            src.Data[ch][0] = 1f;
            var output = DownmixEngine.Downmix(src, matrix, policy);
            Assert.Equal(matrix[ch, 0], output.Data[0][0], 6);
            Assert.Equal(matrix[ch, 1], output.Data[1][0], 6);
            Assert.Equal(0f, output.Data[0][1]);
            Assert.Equal(0f, output.Data[1][1]);
        }
    }

    [Fact]
    public void LfeAttenuateScalesLfeRowOnly()
    {
        var matrix = DownmixMatrix.ItuStereo();
        matrix[3, 0] = 1.0; matrix[3, 1] = 1.0;
        var policy = new LfePolicy { Mode = LfeInclusionMode.Attenuate, AttenuationDb = 20 };
        var effective = DownmixEngine.ApplyLfePolicy(matrix, policy);
        Assert.Equal(0.1, effective[3, 0], 9);
        Assert.Equal(0.1, effective[3, 1], 9);
        Assert.Equal(matrix[0, 0], effective[0, 0]); // other rows untouched
        Assert.Equal(1.0, matrix[3, 0]); // original matrix untouched
    }

    [Fact]
    public void RejectsNonSixChannelInput()
    {
        var stereo = new AudioBuffer(48000, 2, 8);
        Assert.Throws<ArgumentException>(() =>
            DownmixEngine.Downmix(stereo, DownmixMatrix.ItuStereo(), new LfePolicy()));
    }

    [Fact]
    public void ParseRequiresTwelveCoefficients()
    {
        Assert.Throws<FormatException>(() => DownmixMatrix.Parse("1 0 1 0"));
        var m = DownmixMatrix.Parse("1,0;0,1;0.5,0.5;0,0;0.5,0;0,0.5");
        Assert.Equal(0.5, m[2, 0]);
        Assert.Equal(0.5, m[5, 1]);
    }
}
