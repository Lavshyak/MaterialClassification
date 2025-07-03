using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;

namespace MaterialClassification.Worker.Library;

public class ListenerHosted : IHostedService, IDisposable, IAsyncDisposable
{
    private readonly IConnection _connection;
    private readonly ResultSender _resultSender;
    private readonly IClassificationTaskHandler _classificationTaskHandler;
    private IChannel _channelInput = null!;

    public ListenerHosted([FromKeyedServices("RabbitMqConnection")]IConnection connection, ResultSender resultSender, IClassificationTaskHandler classificationTaskHandler)
    {
        _connection = connection;
        _resultSender = resultSender;
        _classificationTaskHandler = classificationTaskHandler;
        Console.WriteLine("ListenerHosted created");
    }
    
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _resultSender.InitAsync();
        cancellationToken.ThrowIfCancellationRequested();
        _channelInput = await _connection.CreateChannelAsync(null, cancellationToken);
        await _channelInput.QueueDeclareAsync(queue: "task_queue", durable: true, exclusive: false,
            autoDelete: false, arguments: null, cancellationToken: cancellationToken);
        
        await _channelInput.BasicQosAsync(prefetchSize: 0, prefetchCount: 5, global: false, cancellationToken);

        var consumer = new AsyncEventingBasicConsumer(_channelInput);
        
        consumer.ReceivedAsync += async (sender, ea) =>
        {
            try
            {
                Console.WriteLine("Received Message");
                var correlationId = ea.BasicProperties.CorrelationId ?? throw new InvalidOperationException();
                var taskIdStr = correlationId;
                var taskId = JsonSerializer.Deserialize<Guid>(taskIdStr);
            
                Console.WriteLine($" [x] Received taskId {taskId}");
            
                var taskResult = await _classificationTaskHandler.HandleAsync(taskId);
                Console.WriteLine($"sending result {taskId}");
                await _resultSender.SendResult(taskId, taskResult);
                // here channel could also be accessed as ((AsyncEventingBasicConsumer)sender).Channel
                Console.WriteLine($"BasicAckAsync {taskId}");
                await _channelInput.BasicAckAsync(deliveryTag: ea.DeliveryTag, multiple: false, cancellationToken);
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }
        };

        await _channelInput.BasicConsumeAsync("task_queue", autoAck: false, consumer: consumer, cancellationToken);
        Console.WriteLine("Listening started.");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await _channelInput.CloseAsync(cancellationToken);
    }

    public void Dispose()
    {
        _channelInput.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        await _channelInput.DisposeAsync();
    }
}