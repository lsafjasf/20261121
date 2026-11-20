using System.Text;

namespace DownmixLab.Core.Audio;

/// <summary>Writes interleaved PCM WAV (optionally with a minimal bext chunk for BWF).</summary>
public static class WavWriter
{
    public static void Write(string path, AudioBuffer buffer, WavSampleFormat format, bool writeBext = false)
    {
        int bytesPerSample = format switch
        {
            WavSampleFormat.Pcm16 => 2,
            WavSampleFormat.Pcm24 => 3,
            WavSampleFormat.Pcm32 => 4,
            WavSampleFormat.Float32 => 4,
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
        ushort tag = format == WavSampleFormat.Float32 ? (ushort)3 : (ushort)1;
        int blockAlign = buffer.Channels * bytesPerSample;
        long dataSize = buffer.Frames * blockAlign;

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);

        long bextSize = writeBext ? 8 + 602 : 0;
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write((uint)(4 + (8 + 16) + bextSize + (8 + dataSize)));
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write((uint)16);
        writer.Write(tag);
        writer.Write((ushort)buffer.Channels);
        writer.Write(buffer.SampleRate);
        writer.Write(buffer.SampleRate * blockAlign);
        writer.Write((ushort)blockAlign);
        writer.Write((ushort)(bytesPerSample * 8));

        if (writeBext)
        {
            var bext = new byte[602];
            var desc = Encoding.ASCII.GetBytes("DownmixLab monitoring export");
            Array.Copy(desc, bext, Math.Min(desc.Length, 256));
            var originator = Encoding.ASCII.GetBytes("DownmixLab");
            Array.Copy(originator, 0, bext, 256, Math.Min(originator.Length, 32));
            writer.Write(Encoding.ASCII.GetBytes("bext"));
            writer.Write((uint)602);
            writer.Write(bext);
        }

        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write((uint)dataSize);

        var interleaved = new byte[blockAlign * 4096];
        long framesWritten = 0;
        while (framesWritten < buffer.Frames)
        {
            int chunk = (int)Math.Min(4096, buffer.Frames - framesWritten);
            for (int f = 0; f < chunk; f++)
            {
                for (int c = 0; c < buffer.Channels; c++)
                {
                    float s = Math.Clamp(buffer.Data[c][framesWritten + f], -1f, 1f);
                    int o = f * blockAlign + c * bytesPerSample;
                    switch (format)
                    {
                        case WavSampleFormat.Pcm16:
                            short v16 = (short)Math.Round(s * 32767f);
                            interleaved[o] = (byte)v16;
                            interleaved[o + 1] = (byte)(v16 >> 8);
                            break;
                        case WavSampleFormat.Pcm24:
                            int v24 = (int)Math.Round(s * 8388607f);
                            interleaved[o] = (byte)v24;
                            interleaved[o + 1] = (byte)(v24 >> 8);
                            interleaved[o + 2] = (byte)(v24 >> 16);
                            break;
                        case WavSampleFormat.Pcm32:
                            BitConverter.GetBytes((int)Math.Round(s * 2147483647.0)).CopyTo(interleaved, o);
                            break;
                        case WavSampleFormat.Float32:
                            BitConverter.GetBytes(s).CopyTo(interleaved, o);
                            break;
                    }
                }
            }
            writer.Write(interleaved, 0, chunk * blockAlign);
            framesWritten += chunk;
        }
    }
}
