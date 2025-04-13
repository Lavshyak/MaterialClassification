using System.Text.Json;
using Microsoft.Extensions.Caching.Distributed;
using Shared;

namespace MaterialClassification.Main;

public class RedisClassificationTaskResultGetter
{
    private readonly IDistributedCache _distributedCache;

    public RedisClassificationTaskResultGetter(IDistributedCache distributedCache)
    {
        _distributedCache = distributedCache;
    }

    public async Task<ClassificationTaskResult?> GetResult(string correlationId, CancellationToken cancellationToken)
    {
        var resultJson = await _distributedCache.GetStringAsync($"classification_task_result_{correlationId}", cancellationToken);
        if (resultJson is null)
            return null;
        var result = JsonSerializer.Deserialize<ClassificationTaskResult>(resultJson);
        return result;
    }
}