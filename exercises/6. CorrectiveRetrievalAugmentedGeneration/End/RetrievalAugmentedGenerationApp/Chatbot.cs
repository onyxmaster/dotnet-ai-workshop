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

        Console.ForegroundColor = ConsoleColor.Green;

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("\nЗапрос: ");
            var userMessage = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userMessage))
            {
                continue;
            }

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"{DateTime.UtcNow:o} отправлено");
            var answer = await thread.AnswerAsync(userMessage, cancellationToken);

            Console.WriteLine($"{DateTime.UtcNow:o} получен ответ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Ответ: {answer.Text}\n");

            Console.ForegroundColor = ConsoleColor.Blue;
            foreach (var citation in answer.Citations)
            {
                Console.WriteLine($"Источник {citation.SourceId}: {citation.Quote}");
            }
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
        => Task.CompletedTask;
}
