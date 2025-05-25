using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using RabbitMQ.Client;
using MaterialClassification.Shared;

namespace MaterialClassification.Worker.Library;

public class ResultSender : IAsyncDisposable
{
    private readonly IConnection _connection;
    private readonly IDistributedCache _distributedCache;
    private IChannel? _channelOutput;
    
    public ResultSender([FromKeyedServices("RabbitMqConnection")]IConnection connection, IDistributedCache distributedCache, IConfiguration configuration)
    {
        _connection = connection;
        _distributedCache = distributedCache;
    }

    public async Task InitAsync()
    {
        _channelOutput = await _connection.CreateChannelAsync(null);
        // Создание очереди
        await _channelOutput.QueueDeclareAsync(queue: "complete_task_queue",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);
    }
    
    public async Task SendResult(Guid taskId, ClassificationTaskResult taskResult)
    {
        var taskIdStr = JsonSerializer.Serialize(taskId);

        Console.WriteLine($"Sending to rabbitmq {taskId}");
        await SendRabbitMq(taskIdStr, taskResult);
        Console.WriteLine($"Sending to redis {taskId}");
        await SendRedis(taskIdStr, taskResult);
    }

    private async Task SendRabbitMq(string taskIdStr, ClassificationTaskResult taskResult)
    {
        var taskResultJsonStr = JsonSerializer.Serialize(taskResult);
        var body = Encoding.UTF8.GetBytes(taskResultJsonStr);
        
        await using var channel = await _connection.CreateChannelAsync();
        
        // Отправка сообщения
        var basicProperties = new BasicProperties() { Persistent = true, CorrelationId = taskIdStr};
        
        await channel.BasicPublishAsync("", "completed_classification_tasks_queue", true, basicProperties, body);
    }

    private async Task SendRedis(string taskIdStr, ClassificationTaskResult taskResult)
    {
        await _distributedCache.SetStringAsync($"classification_task_result_{taskIdStr}",
            JsonSerializer.Serialize(taskResult));
    }
    
    public void Dispose()
    {
        _channelOutput?.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        if (_channelOutput is not null)
        {
            await _channelOutput.DisposeAsync();
        }
    }
}