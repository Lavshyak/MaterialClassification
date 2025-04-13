using RabbitMQ.Client;

namespace MaterialClassification.Main;

public class RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService : IHostedService
{
    private readonly RabbitMqStringResultFromQueueByCorrelationIdListener
        _rabbitMqStringResultFromQueueByCorrelationIdListener;

    private readonly IConnection _rabbitMqConnection;
    private readonly ILogger<RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService> _logger;
    private readonly TaskCompletionSource _taskCompletionSource = new();
    private Task _listenerTask = null!;

    public RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService(
        RabbitMqStringResultFromQueueByCorrelationIdListener rabbitMqStringResultFromQueueByCorrelationIdListener,
        [FromKeyedServices("RabbitMqConnection")]
        IConnection rabbitMqConnection,
        ILogger<RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService> logger)
    {
        _rabbitMqStringResultFromQueueByCorrelationIdListener = rabbitMqStringResultFromQueueByCorrelationIdListener;
        _rabbitMqConnection = rabbitMqConnection;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _listenerTask = _rabbitMqStringResultFromQueueByCorrelationIdListener.InitAndListeningAsync(_rabbitMqConnection,
            "complete_task_queue", _taskCompletionSource.Task, cancellationToken);

        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _taskCompletionSource.SetCanceled(cancellationToken);
        _logger.LogDebug("Wait for stopping RabbitMqStringResultFromQueueByCorrelationIdListener with queue: \"{qn}\"", "complete_task_queue");
        await _listenerTask.WaitAsync(cancellationToken);
        _logger.LogDebug("RabbitMqStringResultFromQueueByCorrelationIdListener with queue: \"{qn}\" is stopped", "complete_task_queue");
    }
}