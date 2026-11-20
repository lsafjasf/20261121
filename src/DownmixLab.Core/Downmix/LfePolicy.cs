namespace DownmixLab.Core.Downmix;

/// <summary>How the LFE channel participates in the stereo downmix.</summary>
public enum LfeInclusionMode
{
    /// <summary>LFE row of the matrix is forced to zero (ITU default).</summary>
    Exclude,
    /// <summary>Matrix LFE row is used as-is.</summary>
    Include,
    /// <summary>Matrix LFE row is scaled by an additional attenuation in dB.</summary>
    Attenuate,
}

public sealed class LfePolicy
{
    public LfeInclusionMode Mode { get; set; } = LfeInclusionMode.Exclude;

    /// <summary>Extra attenuation in dB applied when Mode == Attenuate. Positive value reduces level.</summary>
    public double AttenuationDb { get; set; } = 10.0;

    public double GainFactor => Mode switch
    {
        LfeInclusionMode.Exclude => 0.0,
        LfeInclusionMode.Include => 1.0,
        LfeInclusionMode.Attenuate => Math.Pow(10, -AttenuationDb / 20.0),
        _ => 1.0,
    };

    public override string ToString() => Mode == LfeInclusionMode.Attenuate
        ? $"Attenuate {AttenuationDb:0.#} dB (x{GainFactor:0.###})"
        : Mode.ToString();
}
