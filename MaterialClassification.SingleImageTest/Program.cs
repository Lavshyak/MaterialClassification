using System.Text;
using MaterialClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

var modelPath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.Training\\bin\\Debug\\net9.0\\model_10_90_11.zip";
var mlContext = new MLContext();
ITransformer trainedModel = mlContext.Model.Load(modelPath, out var inputSchema);
Console.WriteLine("Модель загружена.");
var predictionEngine = mlContext.Model.CreatePredictionEngine
    <ProductionImageDataInput, ProductionImageDataOutput>
    (trainedModel);
Console.WriteLine("PredictionEngine создан.");

var b = new DataViewSchema.Builder();
b.AddColumns(inputSchema);
b.AddColumn(nameof(ProductionImageDataInput.SourceImage), new ImageDataViewType(1,1));
var schema = b.ToSchema();
var outputSchema = trainedModel.GetOutputSchema(schema) ?? throw new InvalidOperationException();

var imagePath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.Training\\data\\materials_under_microscope\\Copper-1B\\Copper-1B_6.jpg";

MLImage image;
using (var stream = File.OpenRead(imagePath))
{
    image = MLImage.CreateFromStream(stream);
}

var imageData = new ProductionImageDataInput { SourceImage = image };
var sb = new StringBuilder();
var prediction = predictionEngine.Predict(imageData);
sb.AppendLine(prediction.PredictedLabelValue);
var labelKeyColumn = outputSchema["LabelKey"];
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
