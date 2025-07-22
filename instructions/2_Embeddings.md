# Embeddings

This session will explore one of the basic tools in AI app development, *embeddings*.

An embedding model converts some input - usually text or an image - into a numerical vector (e.g., an array of floats). This vector represents the *meaning* of the input, not the characters in the source string or the pixel colors in an image. Nearby vectors have similar meanings.

## Get your embedding model ready

For this exercise we'll use Ollama, and you can run it natively or with Docker. Embedding models can be small, and can run quickly even on CPU on any laptop.

### Option 1: Running Ollama natively

Pull the the [all-minilm](https://ollama.com/library/all-minilm) embedding model, which is pretty small and general-purpose:

```
ollama pull all-minilm
```

... and leave Ollama running:

```
ollama serve
```

If you get an error like *Error: listen tcp 127.0.0.1:11434: Only one usage of each socket address ... is normally permitted* that just means it's already running. Check your system tray or task bar and quit the existing instance before `ollama serve`.

### Option 2: Running Ollama in Docker

You can pull the ollama image with Docker on your local workstation with this command:

```bash
docker pull ollama/ollama
```

Then, start the ollama container with this command:

```bash
docker run -d --gpus=all -v ./ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

This command is going to create an ollama subfolder in the current folder to store model information.

Next, let's run the `all-minilm` model by executing this command:

```bash
docker exec -it ollama ollama run all-minilm
```

## Open the project

Open the project `exercises/2. Embeddings/Begin`.

In `Program.cs`, notice that there are three different entry points:

```cs
await new SentenceSimilarity().RunAsync();
//await new ManualSemanticSearch().RunAsync();
//await new FaissSemanticSearch().RunAsync();
```

Leave `SentenceSimilarity` uncommented, because that's where we'll start.

Open `SentenceSimilarity.cs`. First check you can generate an embedding for some text. Add this at the bottom of the `RunAsync` method:

```cs
var embedding = await embeddingGenerator.GenerateVectorAsync("Hello, world!");
Console.WriteLine($"Embedding dimensions: {embedding.Span.Length}");
foreach (var value in embedding.Span)
{
    Console.Write("{0:0.00}, ", value);
}
```

If you run this, you should see it produces a vector of length 384. It's a *normalized* vector (i.e., the sum of its squares adds to 1) so it represents a direction in 384-dimensional space. Any sentence that has similar meaning will be in a similar direction, and vice-versa.

## Compute similarity

To compute the similarity between two embeddings, many different metrics are possible.

The most commonly used is *cosine similarity*, which gives a number from -1 to 1 (higher means more similar). It's simply the cosine of the angle between the two vectors. Remember that cos(0)=1, so if the angle between two vectors is zero, this formula will return 1.

Compute similarity over a few strings as follows:

```cs
var catVector = await embeddingGenerator.GenerateVectorAsync("cat");
var dogVector = await embeddingGenerator.GenerateVectorAsync("dog");
var kittenVector = await embeddingGenerator.GenerateVectorAsync("kitten");

Console.WriteLine($"Cat-dog similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, dogVector.Span):F2}");
Console.WriteLine($"Cat-kitten similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, kittenVector.Span):F2}");
Console.WriteLine($"Dog-kitten similarity: {TensorPrimitives.CosineSimilarity(dogVector.Span, kittenVector.Span):F2}");
```

Be sure to add a using statement at the top of the file:

```csharp
using System.Numerics.Tensors;
```

You should see that "cat" is more related to "kitten" than it is to "dog".

## Semantic search

You can use this technique to find the closest text to a given search term. Let's do this for a set of documents for employees at a company, dealing with HR policies and such like. This company only has ~100 such policies, so we can easily implement semantic search in memory without needing any vector database or advanced indexing.

Back in `Program.cs`, comment out the line for `SentenceSimilarity` and uncomment the one for `ManualSemanticSearch`.

Now, in `ManualSemanticSearch.cs`, at the bottom of `RunAsync`, add:

```cs
var titlesWithEmbeddings = await embeddingGenerator.GenerateAndZipAsync(TestData.DocumentTitles.Values);
Console.WriteLine($"Got {titlesWithEmbeddings.Length} title-embedding pairs");
```

`TestData.DocumentTitles` is the dictionary of HR document titles, and `titlesWithEmbeddings` is now an array of `(text, embedding)` pairs. `GenerateAndZipAsync` is equivalent to generating embeddings for all the inputs and then using `.Zip` to combine them with all the inputs.

Verify this by setting a breakpoint after the line above and using the debugger to inspect the value of `titlesWithEmbeddings`.

Next let's implement a search REPL. Add this after the previous code:

```cs
while (true)
{
    Console.Write("\nQuery: ");
    var input = Console.ReadLine()!;
    if (input == "") break;

    // TODO: Compute embedding and search
}
```

Replace the `TODO` comment with the following. First compute the embedding of the current `input`:

```cs
var inputEmbedding = await embeddingGenerator.GenerateVectorAsync(input);
```

And now loop over all the `titlesWithEmbeddings` candidates. For each one, compute the similarity to the search term, and order by similarity descending:

```cs
var closest =
    from candidate in titlesWithEmbeddings
    let similarity = TensorPrimitives.CosineSimilarity(
        candidate.Embedding.Vector.Span, inputEmbedding.Span)
    orderby similarity descending
    select new { candidate.Value, Similarity = similarity };
```

Yes, it's LINQ query syntax! You don't see it a lot these days but it looks nice for something like this. If you want to re-express that with a load of `.Select` and lambdas, go for it. (But what a waste of time.)

Finally, display the closest three:

```cs
foreach (var result in closest.Take(3))
{
    Console.WriteLine($"({result.Similarity:F2}): {result.Value}");
}
```

Try any HR-like search term, such as:

 * exercise
 * where can I park
 * how to get my boss fired

Also try spelling mistakes. And opposites.

## Optional: Implement the similarity calculation manually

Cosine similarity is a really simple algorithm. And given that you're working with normalized vectors, it's mathematically identical to a dot product. This simply means multiplying together the corresponding elements in the two vectors, and adding together the results.

To prove this to yourself, try implementing it:

```cs
private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
{
    // TODO: Implement this
}
```

<details>
  <summary>SOLUTION</summary>

  ```cs
  private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
  {
      var result = 0f;
      for (int i = 0; i < a.Length; i++)
      {
          result += a[i] * b[i];
      }
      return result;
  }
  ```
</details>

Verify you can use this instead of `TensorPrimitives.CosineSimilarity` and still get the same result.

If you're really keen, have a go at vectorizing it (e.g., using `Vector256.Multiply`). There's a solution in `exercises/Embeddings/End`. Is it faster than your unvectorized version? How does the speed compare with `TensorPrimitives.CosineSimilarity`?

## Optional: Zero-shot classification

*Zero-shot classifiers* are given some input text and a set of candidate labels, and are asked to identify the most relevant label. They can be used to automate workflows or suggest categorization or actions in an app UI. The term "zero-shot" refers to the fact that you're asking it to classify into labels that it wasn't specifically trained to identify.

How well does your embedding model work as a zero-shot classifier? Try to implement a zero-shot classification method with this signature:

```cs
/// <summary>
/// Returns the most relevant candidate label.
/// </summary>
public async Task<string> ClassifyAsync(string text, IEnumerable<string> candidateLabels)
{
    // TODO: Implement it
}
```

Example test cases:

 * Given the candidates *Animals*, *Programming*, and *Music*:
   * *This is a Burmese Python* should give *Animals*
   * *This is a Python API* should give *Programming*
   * *This is my song about pythons* should give *Music*
 * Given the candidates *Help*, *Complaint*, and *Returns*:
   * *How can I reset my password?* should give *Help*
   * *I am unhappy with your service and demand a refund* should give *Complaint*
   * *I am sending this item back to you* should give *Returns*

To be clear, embedding models are not really trained for this use case, so don't expect it to do as well as a [proper zero-shot classification model](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=trending).

You can find a solution in `exercises/Embeddings/End/ZeroShotClassification.cs`.

## Optional: Semantic opposites

If cosine similarity is measured from -1 to +1, what are the most semantically different two strings you can find? Can you find any string pair whose cosine similarity is close to -1? What about two opposite-meaning statements?

This is a bit of a trick question so don't spend too long on it. Just give it a quick go if you can.

<details>
  <summary>SOLUTION</summary>

  Almost any pair of meaningful sentences you enter will have some positive "similarity" score, since they have more in common than you might think, e.g.:

   * "Opposite" statements tend to be very similar, as they refer to similar concepts. For example "*Ben will go to your party*" and "*Ben will not go to your party*" are extremely similar, as they are both statements about Ben, your party, and whether someone will do something.
     * This hints at one reason why prompt engineering can be difficult. Telling an LLM *not* to say something often makes it more likely to say that thing, since you've placed the idea in its context window.
   * Seemingly-unrelated sentences like "*7 is a prime number*" and "*where did you put my hat?*" have many points in common, e.g., they are both written in English, are of similar lengths, are both correctly spelled, both sound like lines of dialog, etc.
</details>
