namespace DownmixLab.Core.Analysis;

/// <summary>Iterative radix-2 FFT. Length must be a power of two.</summary>
public static class Fft
{
    public static void MagnitudeSpectrum(float[] samples, int offset, int length, double[] magnitudes)
    {
        if ((length & (length - 1)) != 0)
            throw new ArgumentException("Length must be a power of two.", nameof(length));

        var re = new double[length];
        var im = new double[length];

        // Hann window + bit-reversal copy
        int bits = (int)Math.Log2(length);
        for (int i = 0; i < length; i++)
        {
            int j = ReverseBits(i, bits);
            double w = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            re[j] = (offset + i < samples.Length ? samples[offset + i] : 0f) * w;
        }

        for (int size = 2; size <= length; size <<= 1)
        {
            double angle = -2 * Math.PI / size;
            double wr = Math.Cos(angle), wi = Math.Sin(angle);
            int half = size >> 1;
            for (int start = 0; start < length; start += size)
            {
                double curR = 1, curI = 0;
                for (int k = 0; k < half; k++)
                {
                    int even = start + k, odd = start + k + half;
                    double tr = re[odd] * curR - im[odd] * curI;
                    double ti = re[odd] * curI + im[odd] * curR;
                    re[odd] = re[even] - tr;
                    im[odd] = im[even] - ti;
                    re[even] += tr;
                    im[even] += ti;
                    double nextR = curR * wr - curI * wi;
                    curI = curR * wi + curI * wr;
                    curR = nextR;
                }
            }
        }

        int bins = length / 2;
        for (int i = 0; i < bins && i < magnitudes.Length; i++)
            magnitudes[i] = Math.Sqrt(re[i] * re[i] + im[i] * im[i]) * 2.0 / length;
    }

    private static int ReverseBits(int value, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }
}
