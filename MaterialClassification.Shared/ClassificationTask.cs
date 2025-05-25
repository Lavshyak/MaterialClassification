using System.Text.Json;

namespace MaterialClassification.Shared;

public record ClassificationTaskResult(Guid TaskId, KeyValuePair<string, float>[] ClassNamesPredictionScores);

public static class Test
{
    public static void A()
    {
        Console.WriteLine(JsonSerializer.Serialize(new ClassificationTaskResult(Guid.NewGuid(), new []{new KeyValuePair<string, float>("steel", 0.56f), new KeyValuePair<string, float>("paper", 0.23f)})));
    }
}
