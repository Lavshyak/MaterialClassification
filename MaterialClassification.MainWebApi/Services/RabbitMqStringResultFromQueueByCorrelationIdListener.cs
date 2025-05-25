using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using MaterialClassification.Shared;

namespace MaterialClassification.MainWebApi.Services;

public class RabbitMqStringResultFromQueueByCorrelationIdListener
{
    private readonly ILogger<RabbitMqStringResultFromQueueByCorrelationIdListener> _logger;

    public RabbitMqStringResultFromQueueByCorrelationIdListener(
        ILogger<RabbitMqStringResultFromQueueByCorrelationIdListener> logger)
    {
        _logger = logger;
        _logger.LogDebug("constructor");
    }

    private ConcurrentDictionary<string, TaskCompletionSource<string>> TaskResultListeners { get; set; } = new();

    public async Task InitAndListeningAsync(IConnection connection, string queueName,
        Task waitForExitTask, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Beginning of InitAndListeningAsync");
        
        await using var channel = await connection.CreateChannelAsync(cancellationToken: cancellationToken);
        await channel.QueueDeclareAsync(queue: queueName,
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null,
            cancellationToken: cancellationToken);
        var consumer = new AsyncEventingBasicConsumer(channel);
        consumer.ReceivedAsync += (sender, ea) =>
        {
            try
            {
                var correlationId = ea.BasicProperties.CorrelationId;
                if (string.IsNullOrWhiteSpace(correlationId))
                {
                    _logger.LogDebug("Got result of task with NullOrWhiteSpace correlationId: \"{ci}\"", correlationId);
                    return Task.CompletedTask;
                }

                var message = Encoding.UTF8.GetString(ea.Body.Span);
                if (TaskResultListeners.TryRemove(correlationId, out var tcs))
                {
                    if (tcs.TrySetResult(message))
                    {
                        _logger.LogDebug("Result of task with correlationId: \"{ci}\" is set", correlationId);
                    }
                    else
                    {
                        _logger.LogWarning("Result of task with correlationId: \"{ci}\" is not set", correlationId);
                    }
                }
                else
                {
                    _logger.LogWarning("Can't remove correlationId: \"{ci}\" from ConcurrentDictionary", correlationId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "On handle received task result");
                throw;
            }

            return Task.CompletedTask;
        };

        var str = await channel.BasicConsumeAsync(queue: queueName, autoAck: true, consumer: consumer,
            cancellationToken: cancellationToken);

        _logger.LogInformation("Started to listening queue \"{qn}\"", queueName);

        await waitForExitTask.WaitAsync(cancellationToken);

        _logger.LogInformation("Finished to listening queue \"{qn}\"", queueName);
    }

    public async Task<ClassificationTaskResult?> WaitResult(string correlationId, CancellationToken cancellationToken)
    {
        var tsc = TaskResultListeners.GetOrAdd(correlationId, (_) => new TaskCompletionSource<string>());
        var task = tsc.Task.WaitAsync(cancellationToken);
        _logger.LogInformation("Waiting for result of task with correlationId: \"{ci}\"", correlationId);
        var resultStr = await task;
        _logger.LogInformation("Got result of task with correlationId: \"{ci}\"", correlationId);

        var result = JsonSerializer.Deserialize<ClassificationTaskResult>(resultStr);
        return result;
    }
}