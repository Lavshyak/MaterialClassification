using System.Text;
using System.Text.Json;
using RabbitMQ.Client;

namespace MaterialClassification.MainWebApi.Services;


public class RabbitMqClassificationTaskSenderService
{
    private readonly IConnection _connection;

    public RabbitMqClassificationTaskSenderService([FromKeyedServices("RabbitMqConnection")]IConnection connection)
    {
        _connection = connection;
    }
    
    public async Task Send(Guid taskId)
    {
        var taskIdStr = JsonSerializer.Serialize(taskId);
        var body = Encoding.UTF8.GetBytes(taskIdStr);
        
        await using var channel = await _connection.CreateChannelAsync();
        
        // Создание очереди
        await channel.QueueDeclareAsync(queue: "task_queue",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);
        
        // Отправка сообщения
        var basicProperties = new BasicProperties() { Persistent = true, CorrelationId = taskIdStr };
        await channel.BasicPublishAsync("", "task_queue", true, basicProperties, body);
    }
}