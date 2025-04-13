using System.Text;
using System.Text.Json;
using RabbitMQ.Client;
using Shared;

namespace MaterialClassification.Main;


public class RabbitMqClassificationTaskSenderService
{
    public async Task Send(Guid taskId)
    {
        await SendMessage(JsonSerializer.Serialize(taskId));
    }
    
    private async Task SendMessage(string message)
    {
        var factory = new ConnectionFactory() { HostName = "localhost" };
        await using var connection = await factory.CreateConnectionAsync();
        await using var channel = await connection.CreateChannelAsync();
        
        // Создание очереди
        await channel.QueueDeclareAsync(queue: "task_queue",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);
        
        var body = Encoding.UTF8.GetBytes(message);

        // Отправка сообщения
        var basicProperties = new BasicProperties() { Persistent = true };
        
        await channel.BasicPublishAsync("", "", true, basicProperties, body);
    }
}