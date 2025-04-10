using System.Text;
using MaterialClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace MaterialClassification.WebApi;

public class MaterialClassificationService : IDisposable
{
    private readonly PredictionEngine<ProductionImageDataInput, ProductionImageDataOutput> _predictionEngine;
    private readonly DataViewSchema _outputSchema;
    
    public MaterialClassificationService(IConfiguration configuration)
    {
        var modelPath = configuration.GetValue<string>("ModelPath") ?? throw new InvalidOperationException();
        var mlContext = new MLContext();
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var inputSchema);
        Console.WriteLine("Модель загружена.");
        _predictionEngine = mlContext.Model.CreatePredictionEngine
            <ProductionImageDataInput, ProductionImageDataOutput>
            (trainedModel);
        Console.WriteLine("PredictionEngine создан.");
        
        var b = new DataViewSchema.Builder();
        b.AddColumns(inputSchema);
        b.AddColumn(nameof(ProductionImageDataInput.SourceImage), new ImageDataViewType(1,1));
        var schema = b.ToSchema();
        _outputSchema = trainedModel.GetOutputSchema(schema) ?? throw new InvalidOperationException();
    }

    public string Predict(Stream imageStream)
    {
        var imageData = new ProductionImageDataInput { SourceImage = MLImage.CreateFromStream(imageStream) };
        var sb = new StringBuilder();
        var prediction = _predictionEngine.Predict(imageData);
        sb.AppendLine(prediction.PredictedLabelValue);
        var labelKeyColumn = _outputSchema["LabelKey"];
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
        sb.AppendLine("Вероятности принадлежности к классам материалов:");
        sb.AppendLine(string.Join(", ",
            classesScores.OrderByDescending(cs => cs.PredictionScore).Take(5)
                .Select(cs => $"{cs.ClassName}: {cs.PredictionScore}")));
        return sb.ToString();
    }

    public void Dispose()
    {
        _predictionEngine.Dispose();
    }
}