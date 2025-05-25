using MaterialClassification.Shared;

namespace MaterialClassification.Worker.Library;

public interface IClassificationTaskHandler
{
    Task<ClassificationTaskResult> HandleAsync(Guid taskId);
}