using System.ComponentModel;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace RetrievalAugmentedGenerationApp;

public class ChatbotThread(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient)
{
    private List<ChatMessage> _messages =
    [
        /*
        Answer the user question using ONLY information found by searching product manuals.
            If the product manual doesn't contain the information, you should say so. Do not make up information beyond what is
            given in the product manual.
            
            If this is a question about the product, ALWAYS search the product manual before answering.
            Only search across all product manuals if the user explicitly asks for information about all products.
        */
    ];

    public async Task<(string Text, Citation[] Citations, string[] AllContext)> AnswerAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        var chatOptions = new ChatOptions
        {
            Tools = [AIFunctionFactory.Create(SearchAsync, "search")],
            ToolMode = ChatToolMode.RequireSpecific("search"),
            AllowMultipleToolCalls = true,
            //Instructions?!
        };

        _messages.Clear();
        _messages.Add(new(ChatRole.System, $$"""
            You are an intelligent document search and summarization assistant. Your role is to help users answer their questions.            
            1. First, understand the user's question.
            2. Using the provided `search` tool, search for relevant information in the knowledge database.
            3. Once you have retrieved the information, ignore the documents that are not fully relevant to the question.
            3. Using only the relevant documents, formulate a clear and detailed answer.
            4. If the retrieved information is insufficient to answer the question, state that you do not have enough information. Do not make up an answer.
            5. Always respond in Russian language.
            """));

        _messages.Add(new(ChatRole.User, $$"""
            User question: {{userMessage}}

            Respond as a JSON object in this format: {
                "Sources": [ // The list of the sources used in the answer
                    {
                        "Id": string, // The ID of the source that was used
                        "KeyQuote": string // The most relevant quote from the source, up to 15 words
                    }
                ],
                "IgnoredSources": [ // The list of the ignored sources not used in the answer
                    {
                        "Id": string // The ID of the source that was ignored
                    }
                ],
                "Answer": string  // Answer based on the found documents
            }
            """));
        var response = await chatClient.GetResponseAsync<ChatBotAnswer>(_messages, chatOptions, cancellationToken: cancellationToken);

        //_messages.Add(response);
        if (response.TryGetResult(out var answer))
        {
            Citation[] citations = answer.Sources
                .Select(s => new Citation(s.Id, s.KeyQuote))
                .Where(s => s is not null)
                .ToArray()!;

            return (answer.Answer, citations, []);
        }
        else
        {
            return ("Sorry, there was a problem.", [], []);
        }
    }

    [Description("Searches for information that is relevant to the question")]
    private async Task<SearchResult[]> SearchAsync(
        [Description("The search phrase or keywords")] string searchPhrase)
    {
        Console.WriteLine($"{DateTime.UtcNow:o} агентный запрос: {searchPhrase}");
        var searchPhraseEmbedding = (await embeddingGenerator.GenerateAsync(["Query: " + searchPhrase]))[0];
        var closestChunks = await qdrantClient.SearchAsync(
            collectionName: "manuals",
            vector: searchPhraseEmbedding.Vector.ToArray(),
            limit: 10);
        foreach (var result in closestChunks)
        {
            Console.WriteLine($"{DateTime.UtcNow:o} результат {result.Payload["productId"].StringValue}, вес: {result.Score}, символов: {result.Payload["text"].StringValue.Length}");
        }

        return closestChunks.Select(c => new SearchResult(c.Payload["productId"].StringValue, c.Payload["text"].StringValue)).ToArray();
    }

    public record Citation(string SourceId, string Quote);
    private record SearchResult(string SourceId, string Text);
    private record ChatBotAnswer(ChatBotAnswerSource[] Sources, ChatBotAnswerIgnoredSource[] IgnoredSources, string Answer);
    private record ChatBotAnswerSource(string Id, string KeyQuote);
    private record ChatBotAnswerIgnoredSource(string Id);
}
