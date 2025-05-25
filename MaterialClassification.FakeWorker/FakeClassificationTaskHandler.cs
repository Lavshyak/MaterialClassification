using MaterialClassification.Worker.Library;
using MaterialClassification.Shared;

namespace MaterialClassification.FakeWorker;

public class FakeClassificationTaskHandler : IClassificationTaskHandler
{
    public Task<ClassificationTaskResult> HandleAsync(Guid taskId)
    {
        var result = new ClassificationTaskResult(taskId,
            [
                new KeyValuePair<string, float>("fakeMaterial1", 0.666f),
                new KeyValuePair<string, float>("fakeMaterial2", 0.777f),
            ]
        );
        return Task.FromResult(result);
    }
}