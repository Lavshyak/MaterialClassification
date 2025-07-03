using System.Text.Json;
using Microsoft.ML;

namespace MaterialClassification.WithLbfgs.Training;

public class Program
{
    public record Config(
        int ImageHeight,
        int ImageWidth,
        float OffsetColor,
        string InceptionTensorFlowModelFilePath,
        string ImagesRootPath,
        int ImagesForTestPerClass,
        int ImagesForTrainPerClass,
        int RandomSeed
    );

    static void Main(string[] args)
    {
        var config = JsonSerializer.Deserialize<Config>(File.ReadAllText("config.json")) ??
                     throw new InvalidOperationException();
        MLContext mlContext = new MLContext();

        // подготовка
        var imagesDataGroups =
            Methods.CollectImagesDataFromDirectory(config.ImagesRootPath,
                config.ImagesForTestPerClass + config.ImagesForTrainPerClass);
        Methods.TrainTestSplit(imagesDataGroups,
                config.ImagesForTestPerClass /
                (float)(config.ImagesForTestPerClass + config.ImagesForTrainPerClass),
                config.RandomSeed)
            .Deconstruct(out var trainPart, out var testPart);
        IDataView trainPartDataView = mlContext.Data.LoadFromEnumerable(trainPart);
        IDataView testPartDataView = mlContext.Data.LoadFromEnumerable(testPart);

        var preparationEstimator = Methods.GenerateFromPathToResizedImagesEstimator(mlContext);
        var preparationTransformer = preparationEstimator.Fit(trainPartDataView);

        var preparedTrainPartDataView = preparationTransformer.Transform(trainPartDataView);
        var preparedTestPartDataView = preparationTransformer.Transform(testPartDataView);

        // основной пайплайн
        var estimator = Methods.GenerateClassificationEstimator(mlContext, config.ImageHeight, config.ImageWidth,
            config.OffsetColor, config.InceptionTensorFlowModelFilePath);

        // тренировка
        var model = Methods.TrainModel(estimator, preparedTrainPartDataView);

        // тестирование
        var predictionsDataView = Methods.Test(mlContext, model, preparedTestPartDataView);

        var saveFilePath =
            Path.Combine(Directory.GetCurrentDirectory(),
                $"model_{config.ImagesForTestPerClass}_{config.ImagesForTrainPerClass}_{config.RandomSeed}.zip");
        File.Delete(saveFilePath);

        mlContext.Model.Save(model, predictionsDataView.Schema, saveFilePath);
        Console.WriteLine($"Model saved to {saveFilePath}");
    }
}