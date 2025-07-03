using MaterialClassification.WithLbfgs.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MaterialClassification.WithLbfgs.Training;

public static class Methods
{
    public static IEstimator<ITransformer> GenerateClassificationEstimator(MLContext mlContext, int imageHeight,
        int imageWidth, float colorOffset, string tensorflowModelFilePath)
    {
        IEstimator<ITransformer> pipeline =
                mlContext.Transforms.ResizeImages(
                        inputColumnName: ColumnNames.SourceImage, outputColumnName: "ResizedImage",
                        imageHeight: imageHeight, imageWidth: imageWidth)
                    .Append(mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedImage",
                        outputColumnName: "input", // загружаемая ниже модель принимает колонку input, поэтому здесь input.
                        interleavePixelColors: true, offsetImage: colorOffset))
                    .Append(mlContext.Model.LoadTensorFlowModel(tensorflowModelFilePath)
                        .ScoreTensorFlowModel(
                            outputColumnNames: ["softmax2_pre_activation"], inputColumnNames: ["input"],
                            addBatchDimensionInput: true))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(
                        inputColumnName: ColumnNames.LabelValue, outputColumnName: "LabelKey"))
                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                        labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"
                        /*outputColumnName: "PredictedLabel"*/))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                        inputColumnName: "PredictedLabel",
                        outputColumnName: ColumnNames.PredictedLabelValue))
            ;//.AppendCacheCheckpoint(mlContext);

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

    public static IDataView Test(MLContext mlContext, ITransformer model, IDataView testData)
    {
        Console.WriteLine("=============== Test ===============");
        IDataView predictions = model.Transform(testData);

        Console.WriteLine("=============== Classification metrics ===============");
        MulticlassClassificationMetrics metrics =
            mlContext.MulticlassClassification.Evaluate(predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

        List<(string, string)> toLog =
        [
            ("Кол-во классов", metrics.ConfusionMatrix.NumberOfClasses.ToString()),
            ("Precision",
                (metrics.ConfusionMatrix.PerClassPrecision.Sum() / metrics.ConfusionMatrix.PerClassPrecision.Count)
                .ToString("F6")),

            ("Recall",
                (metrics.ConfusionMatrix.PerClassRecall.Sum() / metrics.ConfusionMatrix.PerClassRecall.Count)
                .ToString("F6")),

            ("LogLoss", metrics.LogLoss.ToString("F6")),
            ("MicroAccuracy", metrics.MicroAccuracy.ToString("F6")),
            ("MacroAccuracy", metrics.MacroAccuracy.ToString("F6")),
            ("MacroAccuracy", metrics.MacroAccuracy.ToString("F6"))
        ];

        Console.WriteLine(string.Join(" ", toLog.Select(x => x.Item1.PadRight(16))));
        Console.WriteLine(string.Join(" ", toLog.Select(x => x.Item2.PadRight(16))));

        Console.WriteLine("=============== End of Test ===============");

        return predictions;
    }

    private static void DisplayResults(IEnumerable<TrainingImageDataOutput> imagePredictionData)
    {
        int i = 0;
        foreach (TrainingImageDataOutput prediction in imagePredictionData)
        {
            Console.WriteLine(
                $"{i++}) Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} right: {(prediction.LabelValue == prediction.PredictedLabelValue ? "True" : "False")}");
        }
    }

    public static IEstimator<ITransformer> GenerateFromPathToResizedImagesEstimator(MLContext mlContext)
    {
        IEstimator<ITransformer> pipeline =
            mlContext.Transforms.LoadImages(inputColumnName: ColumnNames.ImagePath,
                outputColumnName: ColumnNames.SourceImage,
                imageFolder: "");

        return pipeline;
    }

    public record TrainTestParts(PreparationImageDataInput[] TrainPart, PreparationImageDataInput[] TestPart);

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

    public record ImagesDataGroup(string ClassLabel, PreparationImageDataInput[] ImagesData);

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
                            new PreparationImageDataInput
                            {
                                ImagePath = filePath, LabelValue = classLabel
                            })
                        .ToArray());
                })
                .ToArray();

        return imagesDataGroups;
    }
}