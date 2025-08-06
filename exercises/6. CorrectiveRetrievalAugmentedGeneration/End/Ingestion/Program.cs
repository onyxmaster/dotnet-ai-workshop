using System.Buffers;
using System.Diagnostics;
using System.Text;
using BlingFire;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Text;
using Qdrant.Client.Grpc;

static class Program
{
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

        const string Prefix = "query: ";
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
        var index = 0;
        var timer = Stopwatch.StartNew();
        foreach (var filePath in Directory.EnumerateFiles(dir, "*.csv"))
        {
            var bytePool = ArrayPool<byte>.Shared;
            var charPool = ArrayPool<char>.Shared;
            using var stream = File.OpenText(filePath);
            while (stream.ReadLine() is { } line)
            {
                pageLines.Clear();
                totalLength += line.Length;
                pageLines.Add(line);

                string? docId = $"q_{index++}";
                ++count;

                if (docId is null)
                {
                    throw new InvalidDataException("Missing doc ID and/or prefix.");
                }

                if (pageLines.Count == 0)
                {
                    continue;
                }

                var prefix = Prefix;
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

        tasks.Add(Process(docIdsBatch.ToArray(), paragraphsBatch.ToArray()));

        await Task.WhenAll(tasks);
        Console.WriteLine($"Processed {count} files, {totalLength} chars, {totalLength / timer.Elapsed.TotalSeconds} chars/s, took {timer.Elapsed}");

        async Task Process(string[] docIds, string[] paragraphs)
        {
            var task = embeddingGenerator.GenerateAsync(paragraphs);
            var points = new PointStruct[paragraphs.Length];
            var embeddings = await task;
            var index = 0;
            foreach (var docId in docIds)
            {
                points[index] = new()
                {
                    Id = new(Guid.NewGuid()),
                    Vectors = embeddings[index].Vector.ToArray(),
                    Payload =
                    {
                        ["text"] = paragraphs[index].Substring(Prefix.Length),
                        ["queryId"] = docId,
                    }
                };
                ++index;
            }

            await qdrantClient.UpsertAsync("docs", points);
        }
    }
}
