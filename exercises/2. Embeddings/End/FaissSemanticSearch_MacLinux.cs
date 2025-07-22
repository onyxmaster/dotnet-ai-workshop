#if !USE_FAISS_NET
// Note: On Linux, first "sudo apt-get install libopenblas-dev"

using System.Diagnostics;
using Microsoft.Extensions.AI;

namespace Embeddings;

public class FaissSemanticSearch_MacLinux
{
    // The supplied test data contains 60,000 issues, but that may take too long to index
    // We'll work with a smaller set, but you can increase this if your machine can handle it
    private const int TestDataSetSize = 10000;

    // Keep in sync if you use a different model
    private const int EmbeddingDimension = 384;

    // Runs in process on CPU using a small embedding model
    // Alternatively use OllamaEmbeddingGenerator or OpenAIEmbeddingGenerator
    private IEmbeddingGenerator<string, Embedding<float>> EmbeddingGenerator { get; } =
        new LocalEmbeddingsGenerator(); 

    public async Task RunAsync()
    {
        var githubIssues = TestData.GitHubIssues.TakeLast(TestDataSetSize).ToDictionary(x => x.Number, x => x);
        var index = await LoadOrCreateIndexAsync(githubIssues);

        // Search
        while (true)
        {
            Console.Write("\nQuery: ");
            var input = Console.ReadLine()!;
            if (input == "") break;

            var inputEmbedding = await EmbeddingGenerator.GenerateVectorAsync(input);
            var sw = new Stopwatch();
            sw.Start();
            var closest = index.Search(inputEmbedding.ToArray(), 3).ToList();
            sw.Stop();
            foreach (var result in closest)
            {
                var id = (int)result.Label;
                var distance = result.Distance;
                Console.WriteLine($"({distance:F2}): {githubIssues[id].Title}");
            }
            Console.WriteLine($"Search duration: {sw.ElapsedMilliseconds:F2}ms");
        }
    }

    private async Task<FaissMask.IndexIDMap> LoadOrCreateIndexAsync(IDictionary<int, GitHubIssue> data)
    {
        var index = new FaissMask.IndexIDMap(new FaissMask.IndexFlatL2(EmbeddingDimension));

        // Build an index
        foreach (var issuesChunk in data.Chunk(1000))
        {
            Console.Write($"Embedding issues: {issuesChunk.First().Key} - {issuesChunk.Last().Key}");
            var embeddings = await EmbeddingGenerator.GenerateAsync(issuesChunk.Select(i => i.Value.Title));
            Console.WriteLine(" Inserting into index...");

            index.Add(
                embeddings.Select(e => e.Vector.ToArray()).ToArray(),
                issuesChunk.Select(i => (long)i.Key).ToArray());
        }

        return index;
    }
}

#endif // !USE_FAISS_NET
