using Microsoft.Extensions.AI;
using Qdrant.Client;

namespace RetrievalAugmentedGenerationApp;

public class ChatbotThread(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient)
{
    public async Task<int> AnswerAsync(string query, string doc, CancellationToken cancellationToken = default)
    {
        var chatOptions = new ChatOptions
        {
            //Instructions?!
        };

        var messages = new List<ChatMessage>(1);
        messages.Add(new(ChatRole.User, $$"""
            Ты — помощник для оценки релевантности результатов поиска.

            Перед тобой поисковый запрос пользователя и текст найденного документа с сайта Drive2.

            Твоя задача — оценить, насколько этот документ соответствует запросу, по следующей шкале:
            - 0 — совсем не релевантно
            - 1 — почти не релевантно
            - 2 — слабо релевантно
            - 3 — умеренно релевантно
            - 4 — в целом релевантно
            - 5 — полностью релевантно

            Ответ строго в формате JSON: {"relevance": X}, где X — число от 0 до 5. Никаких пояснений, только JSON.
            Не добавляй форматирование, обёртки или комментарии.

            Запрос:
            {{query}}

            Документ:
            {{doc}}
            """));
        var response = await chatClient.GetResponseAsync<ChatBotAnswer>(messages, chatOptions, cancellationToken: cancellationToken);

        //_messages.Add(response);
        if (response.TryGetResult(out var answer))
        {
            return answer.Relevance;
        }
        else
        {
            return -1;
        }
    }

    private record ChatBotAnswer(int Relevance);
}
