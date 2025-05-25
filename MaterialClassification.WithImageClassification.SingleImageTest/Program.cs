using System.Text;
using MaterialClassification.WithImageClassification.MetaInfo;
using MaterialClassification.WithImageClassification.ProductionDataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

var modelPath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.WithImageClassification.Training.Single\\bin\\Debug\\net9.0\\model_withImageClassification_1_1_11.zip";
var mlContext = new MLContext();
ITransformer trainedModel = mlContext.Model.Load(modelPath, out var inputSchema);
Console.WriteLine("Модель загружена.");

Console.WriteLine("PredictionEngine создан.");

var imagePath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\data\\materials_under_microscope\\Copper-1B\\Copper-1B_6.jpg";

var imageBytes = File.ReadAllBytes(imagePath);

var dvsb = new DataViewSchema.Builder();
dvsb.AddColumns([inputSchema[ColumnNames.SourceImageBytes]]);
var schema = dvsb.ToSchema();

Console.WriteLine(DateTime.Now);
var imageData = new ProductionImageDataInput { SourceImageBytes = imageBytes };
var sb = new StringBuilder();
var predictionEngine = mlContext.Model.CreatePredictionEngine
    <ProductionImageDataInput, ProductionImageDataOutput>
    (trainedModel, schema);

var prediction = predictionEngine.Predict(imageData);

sb.AppendLine(prediction.PredictedLabelValue);
var labelKeyColumn = inputSchema[ColumnNames.LabelKey];
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
Console.WriteLine(sb.ToString());

Console.WriteLine("end.");
Console.WriteLine(DateTime.Now);