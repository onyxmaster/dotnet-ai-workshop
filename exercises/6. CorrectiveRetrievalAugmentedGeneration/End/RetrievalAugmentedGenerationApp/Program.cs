using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OpenAI;
using Qdrant.Client;
using RetrievalAugmentedGenerationApp;

// Set up app host
var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddUserSecrets<Program>();
builder.Services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Warning));

IChatClient innerChatClient = new OpenAI.Chat.ChatClient("RedHatAI/gemma-3-12b-it-quantized.w4a16", new("-"), new() { Endpoint = new("http://127.0.0.1:8000/v1") }).AsIChatClient();

// Register services
builder.Services.AddHostedService<Chatbot>();

builder.Services.AddEmbeddingGenerator(
    new OpenAI.Embeddings.EmbeddingClient("intfloat/multilingual-e5-large-instruct", new("-"), new() { Endpoint = new Uri("http://127.0.0.1:8001/v1") }).AsIEmbeddingGenerator());
builder.Services.AddSingleton(new QdrantClient("127.0.0.1"));
builder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation()
    .UseLogging();

// Go
await builder.Build().RunAsync();
