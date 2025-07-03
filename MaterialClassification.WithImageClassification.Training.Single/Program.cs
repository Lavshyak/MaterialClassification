using System.Text.Json;
using MaterialClassification.WithImageClassification.Training.Library;
using Microsoft.ML;

namespace MaterialClassification.WithImageClassification.Training.Single;

public record Config(
    string ImagesRootPath,
    int ImagesForTestPerClass,
    int ImagesForTrainPerClass,
    int RandomSeed,
    int? GpuDeviceId,
    string? SaveModelToDirectory
)
{
    public int TotalImages { get; } = ImagesForTestPerClass + ImagesForTrainPerClass;
    public float TestFraction => (float)ImagesForTestPerClass / TotalImages;
}

public static class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("started");
        var config = JsonSerializer.Deserialize<Config>(File.ReadAllText("config.json")) ??
                     throw new InvalidOperationException();

        var imageDataGroups = Methods.CollectImagesDataFromDirectory(config.ImagesRootPath, config.TotalImages);

        Methods.TrainTestSplit(imageDataGroups, config.TestFraction, config.RandomSeed)
            .Deconstruct(out var trainPart, out var testPart);

        var mlContext = new MLContext(seed: config.RandomSeed);
        mlContext.GpuDeviceId = config.GpuDeviceId;

        var trainPartDataView = mlContext.Data.LoadFromEnumerable(trainPart);
        var testPartDataView = mlContext.Data.LoadFromEnumerable(testPart);

        var preparationEstimator = Methods.GenerateFromPathToImageBytesEstimator(mlContext);
        var preparationTransformer = preparationEstimator.Fit(trainPartDataView);

        var preparedTrainPartDataView = preparationTransformer.Transform(trainPartDataView);
        var preparedTestPartDataView = preparationTransformer.Transform(testPartDataView);

        var estimator = Methods.GenerateClassificationEstimator(mlContext);
        Console.WriteLine("training");
        var model = estimator.Fit(preparedTrainPartDataView);

        Console.WriteLine("testing");
        var transformedTestPartDataView = model.Transform(preparedTestPartDataView);
        Methods.CalculateAndPrintMetrics(mlContext, transformedTestPartDataView, config.ImagesForTrainPerClass,
            config.ImagesForTestPerClass);

        Console.WriteLine("saving");
        var modelFileName =
            $"model_withImageClassification_{config.ImagesForTestPerClass}_{config.ImagesForTrainPerClass}_{config.RandomSeed}.zip";
        var saveFilePath =
            Path.Combine(
                string.IsNullOrWhiteSpace(config.SaveModelToDirectory)
                    ? Directory.GetCurrentDirectory()
                    : config.SaveModelToDirectory,
                modelFileName);
        File.Delete(saveFilePath);
        mlContext.Model.Save(model, transformedTestPartDataView.Schema, saveFilePath);
        Console.WriteLine($"Saved to \"{saveFilePath}\"");
        Console.WriteLine("end");
    }
}