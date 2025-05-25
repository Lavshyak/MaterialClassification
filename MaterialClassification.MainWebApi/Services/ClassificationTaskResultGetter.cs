using System.Text.Json;
using MaterialClassification.Shared;

namespace MaterialClassification.MainWebApi.Services;

public class ClassificationTaskResultGetter
{
    private readonly RedisClassificationTaskResultGetter _redisResultGetter;
    private readonly RabbitMqStringResultFromQueueByCorrelationIdListener _rabbitMqStringResultFromQueueByCorrelationIdListener;

    public ClassificationTaskResultGetter(RedisClassificationTaskResultGetter redisResultGetter, RabbitMqStringResultFromQueueByCorrelationIdListener rabbitMqStringResultFromQueueByCorrelationIdListener)
    {
        _redisResultGetter = redisResultGetter;
        _rabbitMqStringResultFromQueueByCorrelationIdListener = rabbitMqStringResultFromQueueByCorrelationIdListener;
    }

    public async Task<ClassificationTaskResult> WaitResult(Guid taskId, CancellationToken cancellationToken)
    {
        var rabbitWaitResultTask = _rabbitMqStringResultFromQueueByCorrelationIdListener.WaitResult(JsonSerializer.Serialize(taskId), cancellationToken);

        var rabbitResult = await rabbitWaitResultTask.WaitAsync(cancellationToken);
        
        if (rabbitResult is null)
        {
            throw new InvalidOperationException();
        }
        
        return rabbitResult;
    }
    
    public async Task<ClassificationTaskResult> TryGetOrWaitResult(Guid taskId, CancellationToken cancellationToken)
    {
        using var cts = new CancellationTokenSource();
        var rabbitWaitResultTask = _rabbitMqStringResultFromQueueByCorrelationIdListener.WaitResult(JsonSerializer.Serialize(taskId), cts.Token);
        var redisGetResultTask = _redisResultGetter.GetResult(taskId.ToString(), cts.Token);

        await Task.WhenAny(rabbitWaitResultTask, redisGetResultTask).WaitAsync(cancellationToken);

        if (redisGetResultTask.IsCompletedSuccessfully)
        {
            var redisResult = await redisGetResultTask;
            if (redisResult != null)
            {
                await cts.CancelAsync();
                return redisResult;
            }
        }

        var rabbitResult = await rabbitWaitResultTask.WaitAsync(cancellationToken);
        await cts.CancelAsync();
        
        if (rabbitResult is null)
        {
            throw new InvalidOperationException();
        }
        
        return rabbitResult;
    }

    public async Task<ClassificationTaskResult?> TryGetResult(Guid taskId)
    {
        var resultFromRedis = await _redisResultGetter.GetResult(taskId.ToString(), CancellationToken.None);
        return resultFromRedis;
    }
}