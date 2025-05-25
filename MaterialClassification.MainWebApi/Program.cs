using System.Text.Json;
using MaterialClassification.MainWebApi;
using MaterialClassification.MainWebApi.Repositories;
using MaterialClassification.MainWebApi.Services;
using Minio;
using Minio.DataModel.Args;
using RabbitMQ.Client;
using Scalar.AspNetCore;
using Serilog;
using StackExchange.Redis;

var builder = WebApplication.CreateBuilder(args);
builder.Configuration.AddYamlFile("appsettings.yaml");

var defaultLogger = new LoggerConfiguration().ReadFrom
    .Configuration(builder.Configuration).CreateLogger();

Log.Logger = defaultLogger.ForContext("SourceContext", "Static");
builder.Logging.ClearProviders();
builder.Host.UseSerilog(defaultLogger);

try
{
    T GetRequiredConfigurationValue<T>(string key)
    {
        return builder.Configuration.GetValue<T?>(key) ?? throw new InvalidOperationException();
    }
    // Add services to the container.

    builder.Services.AddControllers();
    // Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
    builder.Services.AddOpenApi();


    var rabbitMqConnectionFactory = new ConnectionFactory()
    {
        HostName = GetRequiredConfigurationValue<string>("RabbitMQ:HostName"),
        Port = GetRequiredConfigurationValue<int>("RabbitMQ:Port"),
        UserName = GetRequiredConfigurationValue<string>("RabbitMQ:UserName"),
        Password = GetRequiredConfigurationValue<string>("RabbitMQ:Password"),
        Ssl = { Enabled = GetRequiredConfigurationValue<bool>("RabbitMQ:SSL_Enabled") }
    };
    builder.Services.AddKeyedSingleton("RabbitMqConnection", await rabbitMqConnectionFactory.CreateConnectionAsync());
    builder.Services.AddScoped<RabbitMqClassificationTaskSenderService>();
    builder.Services.AddSingleton<RabbitMqStringResultFromQueueByCorrelationIdListener>();
    builder.Services.AddHostedService<RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService>();
    builder.Services.AddScoped<ClassificationTaskSender>();
    builder.Services.AddScoped<MinioImagesRepository>();
    builder.Services.AddScoped<RedisClassificationTaskResultGetter>();
    builder.Services.AddScoped<ClassificationTaskResultGetter>();

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

    builder.Services.AddMinio(client =>
        client.WithEndpoint(GetRequiredConfigurationValue<string>("Minio:Endpoint"))
            .WithCredentials(GetRequiredConfigurationValue<string>("Minio:AccessKey"),
                GetRequiredConfigurationValue<string>("Minio:SecretKey"))
            .WithSSL(GetRequiredConfigurationValue<bool>("Minio:SSL")));

    builder.Services.AddCors(options =>
    {
        options.AddDefaultPolicy(policy =>
        {
            var originsJson = GetRequiredConfigurationValue<string>("Cors:DefaultPolicy:OriginsJsonArray");
            var origins = JsonSerializer.Deserialize<string[]>(originsJson) ?? throw new InvalidOperationException();
            policy.WithOrigins(origins);
        });
    });

    var app = builder.Build();

    using (var scope = app.Services.CreateScope())
    {
        var bucketName = GetRequiredConfigurationValue<string>("Minio:BucketNames:Images");
        var minioClient = scope.ServiceProvider.GetRequiredService<IMinioClient>();
        var exists = await minioClient.BucketExistsAsync(
            new BucketExistsArgs().WithBucket(bucketName));
        if (!exists)
        {
            await minioClient.MakeBucketAsync(new MakeBucketArgs().WithBucket(bucketName));
        }
    }

    // Configure the HTTP request pipeline.
    if (app.Environment.IsDevelopment())
    {
        app.MapOpenApi();
        app.MapScalarApiReference();
    }

    //app.UseHttpsRedirection();

    app.UseCors();

    app.UseAuthorization();

    app.MapControllers(); //.MapGroup("/backend/webapi/")

    app.MapGet("/", () => "Hello world!");

    app.Run();
}
catch (Exception ex)
{
    Log.Logger.Fatal(ex, "Exception in program.");
    throw;
}
finally
{
    Log.Logger.Information("End of application");
}