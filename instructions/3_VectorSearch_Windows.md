# Vector Search (Windows version)

Continued from [VectorSearch.md](./3_VectorSearch.md). Read that first.

You're going to use FAISS to build an index of many thousands of GitHub issue titles posted to the `dotnet/runtime` repo, and implement fast semantic search.

## Open the project

Open the project `exercises/2. Embeddings/Begin`.

In `Program.cs`, ensure only `FaissSemanticSearch` is uncommented:

```cs
//await new SentenceSimilarity().RunAsync();
//await new ManualSemanticSearch().RunAsync();
await new FaissSemanticSearch().RunAsync();
```

In `FaissSemanticSearch.cs`, in `RunAsync`, add a method call that will build the index:

```cs
var index = await LoadOrCreateIndexAsync("index.bin", githubIssues);
```

## Building an index

Start implementing `LoadOrCreateIndexAsync` as follows:

```cs
private async Task<FaissNet.Index> LoadOrCreateIndexAsync(string filename, IDictionary<int, GitHubIssue> data)
{
    var index = FaissNet.Index.Create(EmbeddingDimension,
       "IDMap2,Flat", FaissNet.MetricType.METRIC_INNER_PRODUCT);
}
```

This will set up a flat index that uses cosine distance (well, dot product, but it's the same because the vectors are normalized). So it's exactly equivalent to what you built manually in the previous exercise. Later we'll change the index type.

Continue by populating and returning it:

```cs
foreach (var issuesChunk in data.Chunk(1000))
{
    Console.Write($"Embedding issues: {issuesChunk.First().Key} - {issuesChunk.Last().Key}");
    var embeddings = await EmbeddingGenerator.GenerateAsync(issuesChunk.Select(i => i.Value.Title));

    Console.WriteLine(" Inserting into index...");
    index.AddWithIds(
        embeddings.Select(e => e.Vector.ToArray()).ToArray(),
        issuesChunk.Select(i => (long)i.Key).ToArray());
}

return index;
```

This asks `EmbeddingGenerator` for embeddings in groups of 1000. You could do it all at once, but most hosted services will have limits.

Try running it. Hopefully in 10 seconds or so, it will compute and index the embeddings for 10,000 issue titles.

### Sidenote: `LocalEmbeddingsGenerator`

Unlike the previous exercise, this does *not* use Ollama (or even OpenAI) to generate the embeddings. It's running an embedding model locally on your CPU, using the [ONNX runtime](https://onnxruntime.ai/). This is generally much faster than calling out to Ollama or OpenAI if you're only working with short strings. But in production, OpenAI or a similar service still has the advantage that it's not blocking your CPU with heavy work.

## Semantic search

Let's search that index. Back in `RunAsync`, add a search loop:

```cs
while (true)
{
    Console.Write("\nQuery: ");
    var input = Console.ReadLine()!;
    if (input == "") break;

    var inputEmbedding = await EmbeddingGenerator.GenerateVectorAsync(input);
    var (resultDistances, resultIds) = index.SearchFlat(1, inputEmbedding.ToArray(), 3);
    for (var i = 0; i < resultDistances.Length; i++)
    {
        var distance = resultDistances[i];
        var id = (int)resultIds[i];
        Console.WriteLine($"({distance:F2}): {githubIssues[id].Title}");
    }
}
```

Try it out! You can search for anything that might appear in dotnet/runtime issue titles, for example:

 * `is .net fast`
 * `why doesn't memory get reclaimed`
 * `please stop making Blazor so amazing, it's embarassing the rest of the ecosystem`

You should get relevant matches even when different wording or spellings are used.

## Persisting the index

Currently it will regenerate the index every time you restart the process. To avoid this, add the following to the end of `LoadOrCreateIndexAsync`, right before the `return` statement:

```cs
index.Save(filename);
```

... and correspondingly add the following at the very *start* of `LoadOrCreateIndexAsync`:

```cs
if (File.Exists(filename))
{
    var result = FaissNet.Index.Load(filename);
    Console.WriteLine($"Loaded index with {result.Count} entries");
    return result;
}
```

Now as long as you keep using the same filename, it will reload previously-populated indexes instead of regenerating from scratch.

## Test the speed

You can set up a proper benchmark if you want, but the following use of `Stopwatch` should be enough to give intuition. In `RunAsync`, use a stopwatch around the call to `SearchFlat`:

```cs
var sw = new Stopwatch();
sw.Start();
var (resultDistances, resultIds) = index.SearchFlat(1, inputEmbedding.ToArray(), 3);
sw.Stop();
```

... and put this after the `for` loop that displays results:

```cs
Console.WriteLine($"Search duration: {sw.ElapsedMilliseconds:F2}ms");
```

On my laptop, searching 10,000 items in this flat index takes about 2ms.

## Try a real index

Go back to `LoadOrCreateIndexAsync` and change how `index` is prepared:

```cs
var index = FaissNet.Index.Create(EmbeddingDimension,
   "IDMap2,HNSW32", FaissNet.MetricType.METRIC_INNER_PRODUCT);
```

Also change the filename being passed in from `RunAsync`:

```cs
var index = await LoadOrCreateIndexAsync("index_hnsw.bin", githubIssues);
```

You should find it takes slightly longer to build the HNSW index, but once it's done, searching is too fast to measure. It shows as `0.00ms` for me (except the first search).

HNSW stands for "Hierarchical Navigable Small Worlds" and refers to a vector indexing technique first developed in 2011. It's far too complicated to explain here, but [here's a good explanation including a video](https://www.pinecone.io/learn/series/faiss/hnsw/).
