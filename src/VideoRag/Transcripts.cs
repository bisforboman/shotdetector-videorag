using ManagedCodecs;
using ManagedCodecs.Aac;
using ManagedCodecs.Containers;
using Whisper.net;

namespace VideoRag;

/// <summary>A stretch of speech: its times in the video and what was said.</summary>
public sealed record TranscriptSegment(TimeSpan Start, TimeSpan End, string Text);

/// <summary>Speech to text over a soundtrack given as 16 kHz mono float samples (what Whisper takes).</summary>
public interface ITranscriber
{
    /// <summary>The segments of <paramref name="samples"/>, timed from its first sample.</summary>
    IAsyncEnumerable<TranscriptSegment> TranscribeAsync(ReadOnlyMemory<float> samples, CancellationToken cancellationToken = default);
}

/// <summary>Whisper (whisper.cpp through Whisper.net) with a ggml model file, the language detected.</summary>
public sealed class WhisperTranscriber(string modelPath, string language = "auto") : ITranscriber, IDisposable
{
    readonly WhisperFactory _factory = WhisperFactory.FromPath(modelPath);

    public async IAsyncEnumerable<TranscriptSegment> TranscribeAsync(ReadOnlyMemory<float> samples, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await using var processor = _factory.CreateBuilder().WithLanguage(language).Build();
        await foreach (var segment in processor.ProcessAsync(samples, cancellationToken).ConfigureAwait(false))
        {
            if (segment.Text.Trim() is { Length: > 0 } text)
                yield return new(segment.Start, segment.End, text);
        }
    }

    public void Dispose() => _factory.Dispose();
}

/// <summary>
/// A video's soundtrack as 16 kHz mono, decoded in managed code by ManagedCodecs: AAC (LC or HE) in MP4 or ADTS.
/// Other containers and codecs have no soundtrack here.
/// </summary>
public static class Soundtrack
{
    /// <summary>Whisper's rate.</summary>
    public const int SampleRate = 16000;

    /// <summary>
    /// The first audio stream as 16 kHz mono and the time of its first sample, or null when the file has no audio
    /// ManagedCodecs reads (no audio stream, not AAC, or a container it doesn't open).
    /// </summary>
    public static (float[] Samples, TimeSpan Start)? Read(string path, CancellationToken cancellationToken = default)
    {
        IDemuxer demuxer;
        try
        {
            demuxer = Demuxer.Open(path);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException)
        {
            return null;
        }

        using (demuxer)
        {
            if (demuxer.Audio is not { } audio || audio.Codec != CodecIds.Aac)
                return null;
            var decoder = new AacDecoder(audio);
            var mono = new List<float>();
            int rate = 0;
            long? firstPts = null;
            foreach (var packet in demuxer.ReadPackets())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (packet.StreamIndex != audio.StreamIndex)
                    continue;
                var frame = decoder.Decode(packet);
                if (packet.Discard || frame.SampleCount == 0)
                    continue;
                firstPts ??= frame.Pts;
                rate = frame.SampleRate;
                Downmix(frame, mono);
            }

            if (firstPts is null)
                return null;
            var start = TimeSpan.FromSeconds(firstPts.Value * audio.TimeBase.ToDouble());
            return (Resample(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(mono), rate, SampleRate), start);
        }
    }

    /// <summary>The channels' mean, appended to <paramref name="mono"/>.</summary>
    static void Downmix(AudioFrame frame, List<float> mono)
    {
        float scale = 1f / frame.Channels;
        for (int i = 0; i < frame.SampleCount; i++)
        {
            float sum = 0;
            for (int c = 0; c < frame.Channels; c++)
                sum += frame.Samples[c][i];
            mono.Add(sum * scale);
        }
    }

    /// <summary>
    /// Band-limited resampling: a Blackman-windowed sinc, cut off at 0.45 of the lower rate, 16 zero crossings each
    /// side. Plenty for speech; not an audiophile's resampler.
    /// </summary>
    public static float[] Resample(ReadOnlySpan<float> input, int fromRate, int toRate)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fromRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(toRate);
        if (fromRate == toRate)
            return input.ToArray();
        const int Crossings = 16;
        double ratio = (double)fromRate / toRate;
        double cutoff = 0.45 * Math.Min(1.0, 1.0 / ratio); // cycles per input sample: 90% of the lower rate's Nyquist
        double halfWidth = Crossings / (2 * cutoff); // input samples each side
        int length = (int)Math.Floor(input.Length / ratio);
        var output = new float[length];
        for (int n = 0; n < length; n++)
        {
            double center = n * ratio;
            int first = Math.Max(0, (int)Math.Ceiling(center - halfWidth));
            int last = Math.Min(input.Length - 1, (int)Math.Floor(center + halfWidth));
            double sum = 0;
            for (int k = first; k <= last; k++)
            {
                double x = k - center;
                double sinc = x == 0 ? 2 * cutoff : Math.Sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
                double w = (x / halfWidth + 1) / 2; // 0 to 1 across the window
                double blackman = 0.42 - (0.5 * Math.Cos(2 * Math.PI * w)) + (0.08 * Math.Cos(4 * Math.PI * w));
                sum += input[k] * sinc * blackman;
            }

            output[n] = (float)sum;
        }

        return output;
    }
}
