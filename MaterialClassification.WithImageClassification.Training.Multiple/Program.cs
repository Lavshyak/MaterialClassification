using System.Text.Json;
using MaterialClassification.WithImageClassification.Training.Library;
using Microsoft.ML;

namespace MaterialClassification.WithImageClassification.Training.Multiple;

public record Config(
    string ImagesRootPath,
    int RandomSeed,
    int? GpuDeviceId,
    string? SaveModelToDirectory
);

public static class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("started");
        var config = JsonSerializer.Deserialize<Config>(File.ReadAllText("config.json")) ??
                     throw new InvalidOperationException();

        const int maxImages = 100;
        const int imagesForTestPerClass = 10;
        int[] imagesForTrainPerClassArr = [2,4,8,16,32,64,90];
        foreach (var imagesForTrainPerClass in imagesForTrainPerClassArr)
        {
            Console.WriteLine($"{nameof(imagesForTrainPerClass)}: {imagesForTrainPerClass}");
            var totalImages = imagesForTestPerClass + imagesForTrainPerClass;
            var testFraction = (float)imagesForTestPerClass / totalImages;

            var imageDataGroups = Methods.CollectImagesDataFromDirectory(config.ImagesRootPath, totalImages);

            Methods.TrainTestSplit(imageDataGroups, testFraction, config.RandomSeed)
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
            Methods.CalculateAndPrintMetrics(mlContext, transformedTestPartDataView, imagesForTrainPerClass, imagesForTestPerClass);

            Console.WriteLine("saving");
            var modelFileName =
                $"model_withImageClassification_{imagesForTestPerClass}_{imagesForTrainPerClass}_{config.RandomSeed}.zip";
            var saveFilePath =
                Path.Combine(string.IsNullOrWhiteSpace(config.SaveModelToDirectory)
                        ? Directory.GetCurrentDirectory()
                        : config.SaveModelToDirectory,
                    modelFileName);
            File.Delete(saveFilePath);
            mlContext.Model.Save(model, transformedTestPartDataView.Schema, saveFilePath);
            Console.WriteLine($"Saved to \"{saveFilePath}\"");
            Console.WriteLine($"end of {imagesForTrainPerClass}");
        }

        Console.WriteLine("end");
    }
}