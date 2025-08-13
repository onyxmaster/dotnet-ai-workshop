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

        const string CollectionName = "docs-e5-large-test";
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
            new OpenAI.Embeddings.EmbeddingClient("intfloat/multilingual-e5-large", new("-"), new() { Endpoint = new Uri("http://127.0.0.1:8000/v1") }).AsIEmbeddingGenerator();

        var qdrantClient = new Qdrant.Client.QdrantClient("procyon10.bru");
        //if (await qdrantClient.CollectionExistsAsync(CollectionName))
        //{
        //    await qdrantClient.DeleteCollectionAsync(CollectionName);
        //}

        //await qdrantClient.CreateCollectionAsync(CollectionName, new VectorParams { Size = 1024, Distance = Distance.Cosine });

        var dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../data/content"));
        const int ContextLength = 512;
        const string Prefix = "passage: ";
        const int Parallelism = 4;
        var tasks = new List<Task>(Parallelism);
        var count = 0;
        var chunkCount = 0;
        var content = new StringBuilder();
        var docIdsBatch = new List<(string, string)>();
        var paragraphsBatch = new List<string>();
        var lengthBuffer = new byte[4];
        var timer = Stopwatch.StartNew();
        foreach (var filePath in Directory.EnumerateFiles(dir, "*.txt.gz"))
        {
            var bytePool = ArrayPool<byte>.Shared;
            var charPool = ArrayPool<char>.Shared;
            using var stream = new GZipStream(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.SequentialScan), CompressionMode.Decompress);
            while (true)
            {
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

                content.Clear();
                string? docId = null;
                string? prefix = null;
                var text = charBuffer.AsSpan(0, contentLength);
                while (!text.IsEmpty)
                {
                    int idx = text.IndexOf('\n');
                    if (idx == -1)
                    {
                        content.Append(text.ToString());
                        break;
                    }

                    if (idx != 0)
                    {
                        var line = text.Slice(0, idx);
                        if (docId is null)
                        {
                            docId = line.ToString();
                        }
                        else if (prefix is null)
                        {
                            prefix = line.ToString();
                        }
                        else
                        {
                            content.Append(line);
                        }
                    }

                    text = text.Slice(idx + 1);
                }

                charPool.Return(charBuffer);
                if (docId is null || prefix is null)
                {
                    throw new InvalidDataException("Missing doc ID and/or prefix.");
                }

                ++count;
                var chunkHeader = Prefix + prefix + "\n";
                if (content.Length != 0)
                {
                    // BlingFireUtils isn't too good at sampling
                    var adjustedContextLength = ContextLength * 8 / 9;
                    var contentString = content.ToString();
                    var lines = TextChunker.SplitPlainTextLines(contentString, adjustedContextLength, tokenCounter: xlmrTokenCounter);
                    var chunkHeader = Prefix + prefix + "\n";
                    var paragraphs = TextChunker.SplitPlainTextParagraphs(lines, adjustedContextLength, ContextLength / 8, chunkHeader: chunkHeader, tokenCounter: xlmrTokenCounter);

                    // SplitPlainTextParagraphs has a bug that merges paragraphs that are too long, so we need to check if any paragraph exceeds the adjusted context length
                    if (paragraphs.Any(p => p.Length > adjustedContextLength && xlmrTokenCounter(p) > adjustedContextLength))
                    {
                        paragraphs = TextChunker.SplitPlainTextParagraphs(pageLines, adjustedContextLength, ContextLength / 4, chunkHeader: chunkHeader, tokenCounter: xlmrTokenCounter);
                    }

                    paragraphsBatch.AddRange(paragraphs);
                    contentString = prefix + "\n" + contentString;
                    for (var i = 0; i < paragraphs.Count; i++)
                    {
                        docIdsBatch.Add((docId, contentString));
                    }
                }
                else
                {
                    paragraphsBatch.Add(Prefix + prefix);
                    docIdsBatch.Add((docId, prefix));
                }


                if (paragraphsBatch.Count < 128)
                {
                    continue;
                }

                chunkCount += paragraphsBatch.Count;
                Console.WriteLine($"Processed {count} files, chunks: {chunkCount}, {count / timer.Elapsed.TotalSeconds} records/s");
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

        chunkCount += paragraphsBatch.Count;
        tasks.Add(Process(docIdsBatch.ToArray(), paragraphsBatch.ToArray()));
        await Task.WhenAll(tasks);
        Console.WriteLine($"Processed {count} files, chunks: {chunkCount}, {count / timer.Elapsed.TotalSeconds} records/s, took {timer.Elapsed}");

        async Task Process((string, string)[] docIds, string[] paragraphs)
        {
            var task = embeddingGenerator.GenerateAsync(paragraphs);
            var points = new PointStruct[paragraphs.Length];
            var id = Interlocked.Add(ref _pointId, (ulong)paragraphs.Length);
            GeneratedEmbeddings<Embedding<float>> embeddings;
            try
            {
                embeddings = await task;
            }
            catch
            {
                Console.Error.WriteLine("Failure detected.");
                return;
            }

            var index = 0;
            foreach (var (docId, content) in docIds)
            {
                points[index] = new()
                {
                    Id = --id,
                    Vectors = embeddings[index].Vector.ToArray(),
                    Payload =
                    {
                        ["text"] = paragraphs[index].Substring(Prefix.Length),
                        ["docId"] = docId,
                        ["content"] = content,
                    }
                };
                ++index;
            }

            await qdrantClient.UpsertAsync(CollectionName, points);
        }
    }
}
