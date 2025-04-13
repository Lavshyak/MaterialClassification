namespace Shared;
public record ClassificationTaskResult(Guid TaskId, KeyValuePair<string, float> ClassNamesPredictionScores);