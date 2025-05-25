using System.Text.Json;
using Microsoft.Extensions.Caching.Distributed;
using MaterialClassification.Shared;

namespace MaterialClassification.MainWebApi.Services;

public class RedisClassificationTaskResultGetter
{
    private readonly IDistributedCache _distributedCache;
    private readonly string _resultPrefix;
    public RedisClassificationTaskResultGetter(IDistributedCache distributedCache, IConfiguration configuration)
    {
        _distributedCache = distributedCache;
        _resultPrefix = configuration["Redis:TaskResultPrefix"] ?? throw new InvalidOperationException();
    }

    public async Task<ClassificationTaskResult?> GetResult(string correlationId, CancellationToken cancellationToken)
    {
        var resultJson = await _distributedCache.GetStringAsync($"{_resultPrefix}{correlationId}", cancellationToken);
        if (resultJson is null)
            return null;
        var result = JsonSerializer.Deserialize<ClassificationTaskResult>(resultJson);
        return result;
    }
}