using MaterialClassification.Shared;
using MaterialClassification.WithImageClassification.MetaInfo;
using MaterialClassification.WithImageClassification.ProductionDataModels;
using MaterialClassification.Worker.Library;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MaterialClassification.WithImageClassification.Worker.Gpu;

public class ImageClassificationTaskHandler : IClassificationTaskHandler
{
    private readonly MinioImagesReadOnlyRepository _imagesReadOnlyRepository;
    private readonly MLContext _mlContext;
    private readonly ITransformer _trainedModel;
    private readonly DataViewSchema _inputSchema;
    private readonly DataViewSchema _sourceInputSchema;

    public ImageClassificationTaskHandler(MinioImagesReadOnlyRepository imagesReadOnlyRepository,
        IConfiguration configuration)
    {
        _imagesReadOnlyRepository = imagesReadOnlyRepository;

        var modelPath = configuration["MLNet:ModelPath"] ?? throw new InvalidOperationException();
        _mlContext = new MLContext();
        Console.WriteLine($"Загрузка модели \"{modelPath}\".");
        _trainedModel = _mlContext.Model.Load(modelPath, out var inputSchema);
        _sourceInputSchema = inputSchema;
        Console.WriteLine("Модель загружена.");

        var dvsb = new DataViewSchema.Builder();
        dvsb.AddColumns([_sourceInputSchema[ColumnNames.SourceImageBytes]]);
        _inputSchema = dvsb.ToSchema();
    }

    public async Task<ClassificationTaskResult> HandleAsync(Guid taskId)
    {
        Console.WriteLine($"ImageClassificationTaskHandler.HandleAsync taskId:{taskId}");
        var imageBytes = await _imagesReadOnlyRepository.GetImage(taskId);
        var imageData = new ProductionImageDataInput { SourceImageBytes = imageBytes };
        var predictionEngine = _mlContext.Model.CreatePredictionEngine
            <ProductionImageDataInput, ProductionImageDataOutput>
            (_trainedModel, _inputSchema);
        Console.WriteLine("before predictionEngine.Predict");
        var prediction = predictionEngine.Predict(imageData);
        Console.WriteLine("predicted");

        var labelKeyColumn = _sourceInputSchema[ColumnNames.LabelKey];
        VBuffer<ReadOnlyMemory<char>> keyValues = default;
        labelKeyColumn.GetKeyValues(ref keyValues);
        var items = keyValues.Items().ToArray();
        if (items.Length != prediction.Score.Length)
        {
            throw new InvalidOperationException();
        }

        var classesScores = items.Zip(prediction.Score,
                (keyValuePair, predictionScore) => new
                    { ClassName = new string(keyValuePair.Value.Span), PredictionScore = predictionScore })
            .ToArray();
        
        var predictionScores = classesScores
            .OrderByDescending(cs => cs.PredictionScore).Take(5)
            .Select(cs => new KeyValuePair<string, float>(cs.ClassName, cs.PredictionScore)).ToArray();

        var result = new ClassificationTaskResult(taskId, predictionScores);
        return result;
    }
}