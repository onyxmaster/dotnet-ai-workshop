using System.Buffers;
using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using BlingFire;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Text;
using Qdrant.Client.Grpc;

static class Program
{
    static ulong _pointId;

    static async Task Main()
    {
        // - Qdrant in Docker (e.g., `docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z -d qdrant/qdrant`)
        var tokenizer = BlingFireUtils.LoadModel("./xlm_roberta_base.bin");
        TextChunker.TokenCounter xlmrTokenCounter = (string text) =>
        {
            if (string.IsNullOrEmpty(text))
            {
                return 0;
            }

            var bytesPool = ArrayPool<byte>.Shared;
            var textBytes = bytesPool.Rent(text.Length * 4);
            var length = Encoding.UTF8.GetBytes(text, textBytes);
            var idsPool = ArrayPool<int>.Shared;
            var tokenIds = idsPool.Rent(length);
            int count = BlingFireUtils.TextToIds(
                tokenizer,
                textBytes,
                length,
                tokenIds,
                length,
                0);
            idsPool.Return(tokenIds);
            bytesPool.Return(textBytes);

            return count;
        };

        const string Prefix = "passage: ";
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
            new OpenAI.Embeddings.EmbeddingClient("intfloat/multilingual-e5-large", new("-"), new() { Endpoint = new Uri("http://127.0.0.1:8001/v1") }).AsIEmbeddingGenerator();

        var qdrantClient = new Qdrant.Client.QdrantClient("procyon10.bru");
        //if (await qdrantClient.CollectionExistsAsync("docs"))
        //{
        //    await qdrantClient.DeleteCollectionAsync("docs");
        //}

        //await qdrantClient.CreateCollectionAsync("docs", new VectorParams { Size = 1024, Distance = Distance.Cosine });

        var dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../data/content"));
        const int ContextLength = 512;

        const int Parallelism = 4;
        var tasks = new List<Task>(Parallelism);
        var count = 0;
        var totalLength = 0L;
        var pageLines = new List<string>();
        var docIdsBatch = new List<string>();
        var paragraphsBatch = new List<string>();
        var timer = Stopwatch.StartNew();
        foreach (var filePath in Directory.EnumerateFiles(dir, "*.txt.gz"))
        {
            var bytePool = ArrayPool<byte>.Shared;
            var charPool = ArrayPool<char>.Shared;
            using var stream = new GZipStream(File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.Read), CompressionMode.Decompress);
            while (true)
            {
                Span<byte> lengthBuffer = stackalloc byte[4];
                try
                {
                    stream.ReadExactly(lengthBuffer);
                }
                catch (EndOfStreamException)
                {
                    break;
                }

                var length = BitConverter.ToInt32(lengthBuffer);
                var contentBuffer = bytePool.Rent(length);
                var contentSpan = contentBuffer.AsSpan(0, length);
                stream.ReadExactly(contentSpan);
                var charBuffer = charPool.Rent(length);
                var contentLength = Encoding.UTF8.GetChars(contentSpan, charBuffer);
                bytePool.Return(contentBuffer);

                pageLines.Clear();
                string? docId = null;
                string? prefix = null;
                var text = charBuffer.AsSpan(0, contentLength);
                while (!text.IsEmpty)
                {
                    int idx = text.IndexOf('\n');
                    if (idx == -1)
                    {
                        pageLines.Add(text.ToString());
                        totalLength += text.Length;
                        break;
                    }

                    if (idx != 0)
                    {
                        var line = text.Slice(0, idx).ToString();
                        if (docId is null)
                        {
                            docId = line;
                        }
                        else if (prefix is null)
                        {
                            prefix = line;
                        }
                        else
                        {
                            pageLines.Add(line);
                        }

                        totalLength += idx;
                    }

                    text = text.Slice(idx + 1);
                }

                charPool.Return(charBuffer);
                ++count;

                if (docId is null || prefix is null)
                {
                    throw new InvalidDataException("Missing doc ID and/or prefix.");
                }

                if (pageLines.Count == 0)
                {
                    continue;
                }

                prefix = Prefix + prefix;
                // TextChunker.SplitPlainTextParagraphs does not appear to reliably handle chunk header length (it has the code, but it fails)
                var adjustedContextLength = ContextLength - xlmrTokenCounter(prefix);
                if (adjustedContextLength <= 0)
                {
                    throw new InvalidDataException($"Skipping {filePath} due to prefix length exceeding context length.");
                }

                var paragraphs = TextChunker.SplitPlainTextParagraphs(pageLines, adjustedContextLength, ContextLength / 8, chunkHeader: prefix, tokenCounter: xlmrTokenCounter);

                // SplitPlainTextParagraphs has a bug that merges paragraphs that are too long, so we need to check if any paragraph exceeds the adjusted context length
                if (paragraphs.Any(p => xlmrTokenCounter(p) > adjustedContextLength))
                {
                    paragraphs = TextChunker.SplitPlainTextParagraphs(pageLines, adjustedContextLength, ContextLength / 4, chunkHeader: prefix, tokenCounter: xlmrTokenCounter);
                }

                paragraphsBatch.AddRange(paragraphs);
                for (var i = 0; i < paragraphs.Count; i++)
                {
                    docIdsBatch.Add(docId);
                }

                if (paragraphsBatch.Count < 100)
                {
                    continue;
                }

                Console.WriteLine($"Processed {count} files, {totalLength} chars, {count / timer.Elapsed.TotalSeconds} files/s, {totalLength / timer.Elapsed.TotalSeconds} chars/s");
                var processDocIds = docIdsBatch.ToArray();
                docIdsBatch.Clear();
                var processParagraphs = paragraphsBatch.ToArray();
                paragraphsBatch.Clear();
                while (tasks.Count >= Parallelism)
                {
                    var completedTask = await Task.WhenAny(tasks);
                    tasks.Remove(completedTask);
                    await completedTask;
                }

                tasks.Add(Process(processDocIds, processParagraphs));
            }
        }

        await Task.WhenAll(tasks);
        Console.WriteLine($"Processed {count} files, {totalLength} chars, {totalLength / timer.Elapsed.TotalSeconds} chars/s, took {timer.Elapsed}");

        async Task Process(string[] docIds, string[] paragraphs)
        {
            var task = embeddingGenerator.GenerateAsync(paragraphs);
            var points = new PointStruct[paragraphs.Length];
            var id = Interlocked.Add(ref _pointId, (ulong)paragraphs.Length);
            var embeddings = await task;
            var index = 0;
            foreach (var docId in docIds)
            {
                points[index] = new()
                {
                    Id = --id,
                    Vectors = embeddings[index].Vector.ToArray(),
                    Payload =
                    {
                        ["text"] = paragraphs[index].Substring(Prefix.Length),
                        ["id"] = docId,
                    }
                };
                ++index;
            }

            await qdrantClient.UpsertAsync("docs", points);
        }
    }
}
