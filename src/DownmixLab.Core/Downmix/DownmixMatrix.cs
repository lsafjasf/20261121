namespace DownmixLab.Core.Downmix;

/// <summary>
/// 6x2 downmix matrix. Rows follow 5.1 order: L, R, C, LFE, Ls, Rs.
/// Columns are the monitoring outputs Lt and Rt.
/// </summary>
public sealed class DownmixMatrix
{
    public const int InputChannels = 6;
    public const int OutputChannels = 2;

    public static readonly string[] InputNames = { "L", "R", "C", "LFE", "Ls", "Rs" };
    public static readonly string[] OutputNames = { "Lt", "Rt" };

    /// <summary>Coefficients [input, output].</summary>
    public double[,] Coefficients { get; } = new double[InputChannels, OutputChannels];

    public double this[int input, int output]
    {
        get => Coefficients[input, output];
        set => Coefficients[input, output] = value;
    }

    /// <summary>ITU-R BS.775 style stereo downmix: C at -3 dB, surrounds at -3 dB, LFE excluded.</summary>
    public static DownmixMatrix ItuStereo()
    {
        var m = new DownmixMatrix();
        double s = Math.Pow(10, -3.0 / 20.0);
        m[0, 0] = 1.0; m[0, 1] = 0.0; // L
        m[1, 0] = 0.0; m[1, 1] = 1.0; // R
        m[2, 0] = s;   m[2, 1] = s;   // C
        m[3, 0] = 0.0; m[3, 1] = 0.0; // LFE
        m[4, 0] = s;   m[4, 1] = 0.0; // Ls
        m[5, 0] = 0.0; m[5, 1] = s;   // Rs
        return m;
    }

    /// <summary>LoRo with center at unity so dialogue stays clearly audible on both sides.</summary>
    public static DownmixMatrix DialogueFriendly()
    {
        var m = ItuStereo();
        m[2, 0] = 1.0;
        m[2, 1] = 1.0;
        return m;
    }

    public DownmixMatrix Clone()
    {
        var m = new DownmixMatrix();
        Array.Copy(Coefficients, m.Coefficients, Coefficients.Length);
        return m;
    }

    /// <summary>Parses 12 semicolon/comma/space separated numbers in row-major order (L..Rs x Lt,Rt).</summary>
    public static DownmixMatrix Parse(string text)
    {
        var parts = text.Split(new[] { ',', ';', ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != InputChannels * OutputChannels)
            throw new FormatException($"Expected {InputChannels * OutputChannels} coefficients, got {parts.Length}.");
        var m = new DownmixMatrix();
        for (int i = 0; i < InputChannels; i++)
            for (int o = 0; o < OutputChannels; o++)
                m[i, o] = double.Parse(parts[i * OutputChannels + o],
                    System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < InputChannels; i++)
            sb.AppendLine($"{InputNames[i],-4} Lt={Coefficients[i, 0],8:F4}  Rt={Coefficients[i, 1],8:F4}");
        return sb.ToString();
    }
}
