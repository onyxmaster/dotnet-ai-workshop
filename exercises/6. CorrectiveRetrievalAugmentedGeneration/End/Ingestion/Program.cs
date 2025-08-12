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
            new OpenAI.Embeddings.EmbeddingClient("intfloat/multilingual-e5-large", new("-"), new() { Endpoint = new Uri("http://127.0.0.1:8000/v1") }).AsIEmbeddingGenerator();

        var qdrantClient = new Qdrant.Client.QdrantClient("procyon10.bru");
        const string CollectionName = "queries";
        //if (await qdrantClient.CollectionExistsAsync(CollectionName))
        //{
        //    await qdrantClient.DeleteCollectionAsync(CollectionName);
        //}

        //await qdrantClient.CreateCollectionAsync(CollectionName, new VectorParams { Size = 1024, Distance = Distance.Cosine });

        var dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../data/content"));

        const int Parallelism = 4;
        var tasks = new List<Task>(Parallelism);
        var count = 0;
        var docIdsBatch = new List<int>();
        var paragraphsBatch = new List<string>();
        var timer = Stopwatch.StartNew();
        foreach (var filePath in Directory.EnumerateFiles(dir, "DoubleChecked.csv"))
        {
            var bytePool = ArrayPool<byte>.Shared;
            var charPool = ArrayPool<char>.Shared;
            using var stream = File.OpenText(filePath);
            stream.ReadLine();
            while (stream.ReadLine() is { } line)
            {
                paragraphsBatch.Add(Prefix + line);
                docIdsBatch.Add(count);

                ++count;
                if (paragraphsBatch.Count < 128)
                {
                    continue;
                }

                Console.WriteLine($"Processed {count} queries, {count / timer.Elapsed.TotalSeconds} q/s");
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
        Console.WriteLine($"Processed {count} queries, {count / timer.Elapsed.TotalSeconds} q/s");

        async Task Process(int[] queryIds, string[] paragraphs)
        {
            var task = embeddingGenerator.GenerateAsync(paragraphs);
            var points = new PointStruct[paragraphs.Length];
            var embeddings = await task;
            var index = 0;
            foreach (var queryId in queryIds)
            {
                points[index] = new()
                {
                    Id = 10_000_000_000UL + (ulong)queryId,
                    Vectors = embeddings[index].Vector.ToArray(),
                    Payload =
                    {
                        ["text"] = paragraphs[index].Substring(Prefix.Length),
                        ["queryId"] = queryId,
                    }
                };
                ++index;
            }

            await qdrantClient.UpsertAsync(CollectionName, points);
        }
    }
}
