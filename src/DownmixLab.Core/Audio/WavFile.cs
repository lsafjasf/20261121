using System.Text;

namespace DownmixLab.Core.Audio;

public enum WavSampleFormat
{
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

/// <summary>
/// Interleaved-free multi-channel audio buffer. Samples are normalized to [-1, 1).
/// Channel order for 5.1 follows WAVEFORMATEXTENSIBLE: L, R, C, LFE, Ls, Rs.
/// </summary>
public sealed class AudioBuffer
{
    public int SampleRate { get; }
    public int Channels { get; }
    public long Frames { get; }
    public float[][] Data { get; }

    public AudioBuffer(int sampleRate, int channels, long frames)
    {
        SampleRate = sampleRate;
        Channels = channels;
        Frames = frames;
        Data = new float[channels][];
        for (int c = 0; c < channels; c++)
            Data[c] = new float[frames];
    }

    public static readonly string[] ChannelNames51 = { "L", "R", "C", "LFE", "Ls", "Rs" };
}

public sealed class WavFile
{
    public AudioBuffer Buffer { get; }
    public WavSampleFormat Format { get; }
    public bool IsBwf { get; }
    public string? BextDescription { get; }

    public WavFile(AudioBuffer buffer, WavSampleFormat format, bool isBwf, string? bextDescription)
    {
        Buffer = buffer;
        Format = format;
        IsBwf = isBwf;
        BextDescription = bextDescription;
    }

    public static WavFile Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

        string riffId = ReadFourCc(reader);
        if (riffId != "RIFF" && riffId != "RF64")
            throw new InvalidDataException("Not a RIFF/RF64 file.");
        reader.ReadUInt32(); // riff size
        if (ReadFourCc(reader) != "WAVE")
            throw new InvalidDataException("Missing WAVE signature.");

        ushort audioFormat = 0, channels = 0, bitsPerSample = 0, blockAlign = 0;
        int sampleRate = 0;
        byte[]? dataChunk = null;
        bool isBwf = false;
        string? bextDescription = null;

        while (stream.Position + 8 <= stream.Length)
        {
            string id = ReadFourCc(reader);
            uint size = reader.ReadUInt32();
            long next = stream.Position + size + (size & 1);
            switch (id)
            {
                case "fmt ":
                    audioFormat = reader.ReadUInt16();
                    channels = reader.ReadUInt16();
                    sampleRate = reader.ReadInt32();
                    reader.ReadInt32(); // byte rate
                    blockAlign = reader.ReadUInt16();
                    bitsPerSample = reader.ReadUInt16();
                    if (audioFormat == 0xFFFE && size >= 40) // EXTENSIBLE
                    {
                        reader.ReadUInt16(); // cbSize
                        reader.ReadUInt16(); // valid bits
                        reader.ReadUInt32(); // channel mask
                        var guid = reader.ReadBytes(16);
                        // SubFormat GUID first two bytes hold the format tag (PCM=1, FLOAT=3)
                        audioFormat = (ushort)(guid[0] | (guid[1] << 8));
                    }
                    break;
                case "data":
                    dataChunk = reader.ReadBytes((int)Math.Min(size, stream.Length - stream.Position));
                    break;
                case "bext":
                    isBwf = true;
                    var bext = reader.ReadBytes((int)Math.Min(size, stream.Length - stream.Position));
                    if (bext.Length >= 256)
                        bextDescription = Encoding.ASCII.GetString(bext, 0, 256).TrimEnd('\0', ' ');
                    break;
            }
            stream.Position = next;
        }

        if (dataChunk == null || channels == 0)
            throw new InvalidDataException("Missing fmt/data chunk.");

        var format = (audioFormat, bitsPerSample) switch
        {
            (1, 16) => WavSampleFormat.Pcm16,
            (1, 24) => WavSampleFormat.Pcm24,
            (1, 32) => WavSampleFormat.Pcm32,
            (3, 32) => WavSampleFormat.Float32,
            _ => throw new InvalidDataException($"Unsupported format tag={audioFormat} bits={bitsPerSample}."),
        };

        long frames = dataChunk.Length / blockAlign;
        var buffer = new AudioBuffer(sampleRate, channels, frames);
        Decode(dataChunk, buffer, format, blockAlign);
        return new WavFile(buffer, format, isBwf, bextDescription);
    }

    private static void Decode(byte[] raw, AudioBuffer buffer, WavSampleFormat format, int blockAlign)
    {
        int ch = buffer.Channels;
        for (long f = 0; f < buffer.Frames; f++)
        {
            int baseOffset = (int)(f * blockAlign);
            for (int c = 0; c < ch; c++)
            {
                int o = baseOffset + c * (blockAlign / ch);
                buffer.Data[c][f] = format switch
                {
                    WavSampleFormat.Pcm16 => (short)(raw[o] | (raw[o + 1] << 8)) / 32768f,
                    WavSampleFormat.Pcm24 => SignExtend24(raw[o] | (raw[o + 1] << 8) | (raw[o + 2] << 16)) / 8388608f,
                    WavSampleFormat.Pcm32 => BitConverter.ToInt32(raw, o) / 2147483648f,
                    WavSampleFormat.Float32 => BitConverter.ToSingle(raw, o),
                    _ => 0f,
                };
            }
        }
    }

    private static int SignExtend24(int v) => (v & 0x800000) != 0 ? v | unchecked((int)0xFF000000) : v;

    private static string ReadFourCc(BinaryReader reader) =>
        Encoding.ASCII.GetString(reader.ReadBytes(4));
}
