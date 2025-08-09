using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Hosting;
using Qdrant.Client;

namespace RetrievalAugmentedGenerationApp;

public class Chatbot(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient)
    : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        var thread = new ChatbotThread(chatClient, embeddingGenerator, qdrantClient);

        using var csv = new CsvReader(File.OpenText("./../../../../../data/content/QdrantResults-BGE.csv"), System.Globalization.CultureInfo.InvariantCulture);
        var header = true;
        var relevanceMap = new Dictionary<int, int>();
        const int Parallelism = 3;
        var tasks = new List<Task>(Parallelism);
        var timer = Stopwatch.StartNew();
        while (csv.Read())
        {
            if (header)
            {
                header = false;
                continue;
            }

            var num = int.Parse(csv[0]!, NumberFormatInfo.InvariantInfo);
            var query = csv[1]!;
            var doc = csv[3]!;
            while (tasks.Count >= Parallelism)
            {
                var completedTask = await Task.WhenAny(tasks);
                tasks.Remove(completedTask);
                await completedTask;
            }

            tasks.Add(Process(num, query, doc));
            if (num % 100 == 0)
            {
                Console.Error.WriteLine($"Processed {num} queries, {num / timer.Elapsed.TotalSeconds} q/s");
            }

            //if (num == 1000)
            //{
            //    break;
            //}
        }

        await Task.WhenAll(tasks);

        Console.WriteLine();
        foreach (var item in relevanceMap.OrderBy(kvp => kvp.Key))
        {
            Console.WriteLine($"{item.Key}: {item.Value}");
        }

        async Task Process(int num, string query, string doc)
        {
            var relevance = await thread.AnswerAsync(query, doc, cancellationToken);
            lock (relevanceMap)
            {
                relevanceMap.TryGetValue(relevance, out var count);
                relevanceMap[relevance] = count + 1;
            }

            Console.WriteLine($"{num},{relevance}");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
        => Task.CompletedTask;
}
