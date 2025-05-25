using MaterialClassification.FakeWorker;
using MaterialClassification.Worker.Library;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RabbitMQ.Client;
using StackExchange.Redis;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddEnvironmentVariables();
//builder.Configuration.AddJsonFile("appsettings.json", optional: true);

var configuration = builder.Configuration;

T GetRequiredConfigurationValue<T>(string key)
{
    return configuration.GetValue<T?>(key) ?? throw new InvalidOperationException();
}


builder.Services.AddStackExchangeRedisCache(options =>
{
    options.InstanceName = "local";
    options.ConfigurationOptions = new ConfigurationOptions()
    {
        EndPoints = { GetRequiredConfigurationValue<string>("Redis:Endpoints") },
        Password = GetRequiredConfigurationValue<string>("Redis:Password"),
        Ssl = GetRequiredConfigurationValue<bool>("Redis:SSL"),
    };
});

var rabbitMqConnectionFactory = new ConnectionFactory()
{
    HostName = GetRequiredConfigurationValue<string>("RabbitMQ:HostName"),
    Port = GetRequiredConfigurationValue<int>("RabbitMQ:Port"),
    UserName = GetRequiredConfigurationValue<string>("RabbitMQ:UserName"),
    Password = GetRequiredConfigurationValue<string>("RabbitMQ:Password"),
    Ssl = { Enabled = GetRequiredConfigurationValue<bool>("RabbitMQ:SSL_Enabled") }
};
builder.Services.AddKeyedSingleton("RabbitMqConnection", await rabbitMqConnectionFactory.CreateConnectionAsync());

builder.Services.AddSingleton<IClassificationTaskHandler, FakeClassificationTaskHandler>();
builder.Services.AddSingleton<ResultSender>();
builder.Services.AddHostedService<ListenerHosted>();

using IHost host = builder.Build();

host.Run();

Console.WriteLine("End.");