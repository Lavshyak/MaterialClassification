using MaterialClassification.WithImageClassification.Worker.Gpu;
using MaterialClassification.Worker.Library;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Minio;
using RabbitMQ.Client;
using StackExchange.Redis;

namespace MaterialClassification.WithImageClassification.Worker.Library;

public static class WorkerProgram
{
    public static async Task MainAsync(string[] args)
    {
        HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);
        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddYamlFile("appsettings.yaml");

        var configuration = builder.Configuration;

        T GetRequiredConfigurationValue<T>(string key)
        {
            try
            {
                return configuration.GetValue<T?>(key) ?? throw new InvalidOperationException();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"env key: {key}");
                throw;
            }
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
        Console.WriteLine("Wait for RabbitMQ connection established...");
        builder.Services.AddKeyedSingleton("RabbitMqConnection", await rabbitMqConnectionFactory.CreateConnectionAsync());
        Console.WriteLine("RabbitMQ connection established.");

        builder.Services.AddMinio(client =>
            client.WithEndpoint(GetRequiredConfigurationValue<string>("Minio:Endpoint"))
                .WithCredentials(GetRequiredConfigurationValue<string>("Minio:AccessKey"),
                    GetRequiredConfigurationValue<string>("Minio:SecretKey"))
                .WithSSL(GetRequiredConfigurationValue<bool>("Minio:SSL")));

        builder.Services.AddSingleton<MinioImagesReadOnlyRepository>();
        builder.Services.AddSingleton<IClassificationTaskHandler, ImageClassificationTaskHandler>();
        builder.Services.AddSingleton<ResultSender>();
        builder.Services.AddHostedService<ListenerHosted>();

        using IHost host = builder.Build();

        await host.RunAsync();

        Console.WriteLine("End.");
    }
}