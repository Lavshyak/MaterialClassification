using MaterialClassification.Main;
using Minio;
using RabbitMQ.Client;
using Serilog;

var builder = WebApplication.CreateBuilder(args);
builder.Configuration.AddYamlFile("appsettings.yml");

var defaultLogger = new LoggerConfiguration().ReadFrom
    .Configuration(builder.Configuration).CreateLogger();

Log.Logger = defaultLogger.ForContext("SourceContext", "Static");
builder.Logging.ClearProviders();
builder.Host.UseSerilog(defaultLogger);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var rabbitMqConnectionFactory = new ConnectionFactory()
{
    HostName = "rabbitmq",
    Port = 5672,
    UserName = "rmquser",
    Password = "rmqpassword",
};
builder.Services.AddKeyedSingleton("RabbitMQConnection", await rabbitMqConnectionFactory.CreateConnectionAsync());
builder.Services.AddScoped<RabbitMqClassificationTaskSenderService>();
builder.Services.AddSingleton<RabbitMqStringResultFromQueueByCorrelationIdListener>();
builder.Services.AddHostedService<RabbitMqStringResultFromQueueByCorrelationIdListenerHostedService>();

builder.Services.AddStackExchangeRedisCache(options => {
    options.Configuration = "redis:6379, password=rpassword";
    options.InstanceName = "local";
});

builder.Services.AddMinio(client => client.WithEndpoint("minio:9000").WithCredentials("muser", "mpassword"));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();