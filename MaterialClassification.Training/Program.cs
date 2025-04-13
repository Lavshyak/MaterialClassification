using System.Text.Json;
using MaterialClassification.DataModels;
using Microsoft.ML;
using TrainigLib;

namespace MaterialClassification.Training;

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
        var imagesDataGroups =
            Class1.CollectImagesDataFromDirectory(config.ImagesRootPath,
                config.ImagesForTestPerClass + config.ImagesForTrainPerClass);
        Class1.TrainTestSplit(imagesDataGroups,
                config.ImagesForTestPerClass /
                (float)(config.ImagesForTestPerClass + config.ImagesForTrainPerClass),
                config.RandomSeed)
            .Deconstruct(out var trainPart, out var testPart);
        IDataView trainPartDataView = mlContext.Data.LoadFromEnumerable(trainPart);
        IDataView testPartDataView = mlContext.Data.LoadFromEnumerable(testPart);

        var preparationEstimator = Class1.GenerateFromPathToResizedImagesEstimator(mlContext);
        var preparationTransformer = preparationEstimator.Fit(trainPartDataView);
        var preparedTrainPartDataView = preparationTransformer.Transform(trainPartDataView);
        var preparedTestPartDataView = preparationTransformer.Transform(testPartDataView);

        var estimator = Class1.GenerateClassificationEstimator(mlContext, config.ImageHeight, config.ImageWidth, config.OffsetColor, config.InceptionTensorFlowModelFilePath);
        var model = Class1.TrainModel(estimator, preparedTrainPartDataView);
        var predictionsDataView = Class1.Test(mlContext, model, preparedTestPartDataView);

        var saveFilePath =
            Path.Combine(Directory.GetCurrentDirectory(),
                $"model_{config.ImagesForTestPerClass}_{config.ImagesForTrainPerClass}_{config.RandomSeed}.zip");
        File.Delete(saveFilePath);

        mlContext.Model.Save(model, predictionsDataView.Schema, saveFilePath);
        Console.WriteLine($"Model saved to {saveFilePath}");
    } 
    
    public static void Main1()
    {
        var config = JsonSerializer.Deserialize<Config>(File.ReadAllText("config.json")) ??
                     throw new InvalidOperationException();

        for (int i = 1; i < 2; i++)
        {
            MLContext mlContext = new MLContext();
            var classesCount = (int) Math.Pow(2, i);
            var imagesForTrain = Class2.CreateImages(classesCount, 2).ToArray();
            var imagesForTest = Class2.CreateImages(classesCount, 1).ToArray();
            IDataView trainPartDataView = mlContext.Data.LoadFromEnumerable(imagesForTrain);
            IDataView testPartDataView = mlContext.Data.LoadFromEnumerable(imagesForTest);

            var estimator = Class1.GenerateClassificationEstimator(mlContext, config.ImageHeight, config.ImageWidth, config.OffsetColor, config.InceptionTensorFlowModelFilePath);
            var model = Class1.TrainModel(estimator, trainPartDataView);
            var predictionsDataView = Class1.Test(mlContext, model, testPartDataView);

            var predictionEngine = mlContext.Model.CreatePredictionEngine
                <ProductionImageDataInput, ProductionImageDataOutput>
                (model);

            predictionEngine.Predict(imagesForTest[0]);
            predictionEngine.Predict(imagesForTest[1]);
            Console.WriteLine($"end of {i}.");
        }
    } 
}