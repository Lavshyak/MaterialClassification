using Shared;

namespace MaterialClassification.Main;

public class ClassificationTaskSender
{
    private readonly RabbitMqClassificationTaskSenderService _rabbitMqClassificationTaskSenderService;
    private readonly MinioImageRepository _minioImageRepository;

    public ClassificationTaskSender(RabbitMqClassificationTaskSenderService rabbitMqClassificationTaskSenderService, MinioImageRepository minioImageRepository)
    {
        _rabbitMqClassificationTaskSenderService = rabbitMqClassificationTaskSenderService;
        _minioImageRepository = minioImageRepository;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="imageStream"></param>
    /// <returns>task id</returns>
    public async Task<Guid> SendOnly(Stream imageStream)
    {
        var taskId = Guid.NewGuid();

        await _minioImageRepository.SendImage(taskId, imageStream);
        await _rabbitMqClassificationTaskSenderService.Send(taskId);
        
        return taskId;
    }
}

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
        using var cts = new CancellationTokenSource();
        var rabbitWaitResultTask = _rabbitMqStringResultFromQueueByCorrelationIdListener.WaitResult(taskId.ToString(), cts.Token);
        var redisGetResultTask = _redisResultGetter.GetResult(taskId.ToString(), cts.Token);

        await Task.WhenAny(rabbitWaitResultTask, redisGetResultTask);

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