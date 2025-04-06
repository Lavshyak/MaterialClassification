using System.Text.Json;
using MaterialClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MaterialClassification.Training;


public class Program
{
    public record Config(
        int ImageHeight,
        int ImageWidth,
        float OffsetColor,
        bool ChannelsLast,
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
            CollectImagesDataFromDirectory(config.ImagesRootPath,
                config.ImagesForTestPerClass + config.ImagesForTrainPerClass);
        TrainTestSplit(imagesDataGroups,
                config.ImagesForTestPerClass / 
                (float)(config.ImagesForTestPerClass + config.ImagesForTrainPerClass),
                config.RandomSeed)
            .Deconstruct(out var trainPart, out var testPart);
        IDataView trainPartDataView = mlContext.Data.LoadFromEnumerable(trainPart);
        IDataView testPartDataView = mlContext.Data.LoadFromEnumerable(testPart);
        
        var preparationEstimator = GenerateFromPathToResizedImagesEstimator(mlContext, config);
        var preparationTransformer = preparationEstimator.Fit(trainPartDataView);
        var preparedTrainPartDataView = preparationTransformer.Transform(trainPartDataView);
        var preparedTestPartDataView = preparationTransformer.Transform(testPartDataView);
        
        var estimator = GenerateClassificationEstimator(mlContext, config);
        var model = TrainModel(estimator, preparedTrainPartDataView);
        Test(mlContext, model, preparedTestPartDataView);

        var saveFilePath =
            Path.Combine(Directory.GetCurrentDirectory(), $"model_{config.ImagesForTestPerClass}_{config.ImagesForTrainPerClass}_{config.RandomSeed}.zip");
        File.Delete(saveFilePath);
        mlContext.Model.Save(model, trainPartDataView.Schema,
            saveFilePath);
        Console.WriteLine($"Model saved to {saveFilePath}");
    }

    public record ImagesDataGroup(string ClassLabel, ImageData[] ImagesData);

    public static ImagesDataGroup[] CollectImagesDataFromDirectory(string directoryPath,
        int takeImagesPerClass = int.MaxValue)
    {
        var classesDirPaths = Directory.EnumerateDirectories(directoryPath);

        var imagesDataGroups =
            classesDirPaths.Select(classDirPath =>
                {
                    var filePaths = Directory.EnumerateFiles(classDirPath);
                    var classLabel = Path.GetFileName(classDirPath) ?? throw new InvalidOperationException();

                    return new ImagesDataGroup(classLabel, filePaths.Take(takeImagesPerClass).Select(filePath =>
                            new ImageData
                            {
                                ImagePath = filePath, Label = classLabel
                            })
                        .ToArray());
                })
                .ToArray();

        return imagesDataGroups;
    }

    public record TrainTestParts(ImageData[] TrainPart, ImageData[] TestPart);

    public static TrainTestParts TrainTestSplit(ImagesDataGroup[] imagesDataGroups, float testFraction,
        int randomSeed)
    {
        Console.WriteLine("Total number of classes: " + imagesDataGroups.Length);

        var random = new Random(randomSeed);

        var groupsWithParts = imagesDataGroups.Select(group =>
        {
            var arrayCopy = group.ImagesData.ToArray();
            random.Shuffle(arrayCopy);
            var testPart = arrayCopy.Take((int)Math.Round(arrayCopy.Length * testFraction)).ToArray();
            var trainPart = arrayCopy.Skip(testPart.Length).ToArray();

            Console.WriteLine($"{group.ClassLabel}. train: {trainPart.Length}. test: {testPart.Length}");

            return new { group.ClassLabel, testPart, trainPart };
        }).ToArray();

        var trainPartGroups = groupsWithParts
            .SelectMany(group => group.trainPart).ToArray();
        var testPartGroups = groupsWithParts
            .SelectMany(group => group.testPart).ToArray();

        return new TrainTestParts(trainPartGroups, testPartGroups);
    }

    public static IEstimator<ITransformer> GenerateFromPathToResizedImagesEstimator(MLContext mlContext, Config config)
    {
        IEstimator<ITransformer> pipeline =
            // 1
            mlContext.Transforms.LoadImages(inputColumnName: nameof(ImageData.ImagePath),
                    outputColumnName: "SourceImage",
                    imageFolder: "")
                // 2
                .Append(mlContext.Transforms.ResizeImages(
                    inputColumnName: "SourceImage", outputColumnName: "ResizedImage",
                    imageHeight: config.ImageHeight, imageWidth: config.ImageWidth));

        return pipeline;
    }

    // Создать модель для тренировки
    public static IEstimator<ITransformer> GenerateClassificationEstimator(MLContext mlContext, Config config)
    {
        IEstimator<ITransformer> pipeline =
            // 1
            mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedImage",
                    outputColumnName: "input", // загружаемая ниже модель принимает колонку input, поэтому здесь input.
                    interleavePixelColors: config.ChannelsLast, offsetImage: config.OffsetColor)
                // 4
                .Append(mlContext.Model.LoadTensorFlowModel(config.InceptionTensorFlowModelFilePath)
                    .ScoreTensorFlowModel(
                        outputColumnNames: ["softmax2_pre_activation"], inputColumnNames: ["input"],
                        addBatchDimensionInput: true))
                // 5
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label", outputColumnName: "LabelKey"))
                // 6
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                    labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"
                    /*outputColumnName: "PredictedLabel"*/))
                // 7
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    inputColumnName: "PredictedLabel", outputColumnName: "PredictedLabelValue"))
                .AppendCacheCheckpoint(mlContext);

        return pipeline;
    }

    public static ITransformer TrainModel(IEstimator<ITransformer> estimator, IDataView trainingData)
    {
        // Train the model
        Console.WriteLine("=============== Training classification model ===============");
        // Create and train the model
        ITransformer model = estimator.Fit(trainingData);

        Console.WriteLine("=============== End of Training classification model ===============");

        return model;
    }

    public static void Test(MLContext mlContext, ITransformer model, IDataView testData)
    {
        Console.WriteLine("=============== Test ===============");
        // Generate predictions from the test data, to be evaluated
        IDataView predictions = model.Transform(testData);

        // Create an IEnumerable for the predictions for displaying results
        IEnumerable<ImagePrediction> imagePredictionData =
            mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
        DisplayResults(imagePredictionData.Where(ipd => ipd.PredictedLabelValue != ipd.Label));

        // Get performance metrics on the model using training data
        Console.WriteLine("=============== Classification metrics ===============");

        MulticlassClassificationMetrics metrics =
            mlContext.MulticlassClassification.Evaluate(predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine("Кол-во классов: " + metrics.ConfusionMatrix.NumberOfClasses);

        Console.WriteLine(
            $"Precision is: {(metrics.ConfusionMatrix.PerClassPrecision.Sum() / metrics.ConfusionMatrix.PerClassPrecision.Count):F6}");
        //Console.WriteLine($"PerClassPrecision is: {string.Join(" ", metrics.ConfusionMatrix.PerClassPrecision.Select(precision => precision.ToString("F3")))}");
        //Console.WriteLine();

        Console.WriteLine(
            $"Recall is: {(metrics.ConfusionMatrix.PerClassRecall.Sum() / metrics.ConfusionMatrix.PerClassRecall.Count):F6}");
        //Console.WriteLine($"PerClassRecall is: {string.Join(" ", metrics.ConfusionMatrix.PerClassRecall.Select(precision => precision.ToString("F3")))}");
        //Console.WriteLine();

        Console.WriteLine($"LogLoss is: {metrics.LogLoss:F6}");
        //Console.WriteLine($"PerClassLogLoss is: {string.Join(" ", metrics.PerClassLogLoss.Select(precision => precision.ToString("F3")))}");
        //Console.WriteLine();

        Console.WriteLine($"MicroAccuracy is: {metrics.MicroAccuracy:F6}");
        Console.WriteLine($"MacroAccuracy is: {metrics.MacroAccuracy:F6}");

        Console.WriteLine("=============== End of Test ===============");
    }

    private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
    {
        int i = 0;
        foreach (ImagePrediction prediction in imagePredictionData)
        {
            Console.WriteLine(
                $"{i++}) Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} right: {(prediction.Label == prediction.PredictedLabelValue ? "True" : "False")}");
        }
    }
}