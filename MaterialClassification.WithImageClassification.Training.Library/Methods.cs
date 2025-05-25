using MaterialClassification.WithImageClassification.MetaInfo;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace MaterialClassification.WithImageClassification.Training.Library;

public class PreparationImageDataInput
{
    [ColumnName(ColumnNames.ImagePath)]
    public string ImagePath { get; set; } = null!;

    [ColumnName(ColumnNames.LabelValue)]
    public string LabelValue { get; set; } = null!;
}

public class TrainingImageDataInput : PreparationImageDataInput
{
    [ColumnName(ColumnNames.SourceImageBytes)]
    public byte[] SourceImageBytes { get; set; } = null!;
}

public class TrainingImageDataOutput : TrainingImageDataInput
{
    // Вероятности принадлежности к классам
    [ColumnName(ColumnNames.Score)]
    public float[] Score { get; set; } = null!;

    // Результат предсказания
    [ColumnName(ColumnNames.PredictedLabelValue)]
    public string PredictedLabelValue { get; set; } = null!;
}

public class Methods
{
    public static void CalculateAndPrintMetrics(MLContext mlContext, IDataView transformedTestPartDataView,
        int imagesForTrainPerClassCount, int imagesForTestPerClassCount)
    {
        MulticlassClassificationMetrics metrics =
            mlContext.MulticlassClassification.Evaluate(transformedTestPartDataView,
                labelColumnName: ColumnNames.LabelKey,
                predictedLabelColumnName: ColumnNames.PredictedLabelKey,
                scoreColumnName: ColumnNames.Score);

        List<(string, string)> toLog =
        [
            ("Кол-во классов", metrics.ConfusionMatrix.NumberOfClasses.ToString()),
            ("forTestPerClass", imagesForTestPerClassCount.ToString()),
            ("forTrainPerClass", imagesForTrainPerClassCount.ToString()),
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

        Console.WriteLine(string.Join(" ", toLog.Select(x => x.Item1.PadRight(20))));
        Console.WriteLine(string.Join(" ", toLog.Select(x => x.Item2.PadRight(20))));
    }

    public static IEstimator<ITransformer> GenerateClassificationEstimator(MLContext mlContext)
    {
        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = ColumnNames.SourceImageBytes,
            LabelColumnName = ColumnNames.LabelKey,
            PredictedLabelColumnName = ColumnNames.PredictedLabelKey,
            ValidationSet = null,
            Arch = ImageClassificationTrainer.Architecture.InceptionV3,
            ScoreColumnName = ColumnNames.Score,
            //MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = true,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        var trainingPipeline =
            mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: ColumnNames.PredictedLabelKey,
                    outputColumnName: ColumnNames.PredictedLabelValue));

        return trainingPipeline;
    }

    public static IEstimator<ITransformer> GenerateFromPathToImageBytesEstimator(MLContext mlContext)
    {
        IEstimator<ITransformer> pipeline =
            mlContext.Transforms.LoadRawImageBytes(inputColumnName: ColumnNames.ImagePath,
                    outputColumnName: ColumnNames.SourceImageBytes,
                    imageFolder: "")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: ColumnNames.LabelValue,
                    outputColumnName: ColumnNames.LabelKey, keyOrdinality: Microsoft.ML.Transforms
                        .ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
                .Append(mlContext.Transforms.DropColumns(ColumnNames.LabelValue, ColumnNames.ImagePath));

        return pipeline;
    }

    public static IEstimator<ITransformer> GenerateFromImageAndLabelValuePreparationEstimator(MLContext mlContext)
    {
        IEstimator<ITransformer> pipeline =
            mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: ColumnNames.LabelValue,
                outputColumnName: ColumnNames.LabelKey, keyOrdinality: Microsoft.ML.Transforms
                    .ValueToKeyMappingEstimator.KeyOrdinality.ByValue);

        return pipeline;
    }

    public record ImagesDataGroup(string ClassLabel, PreparationImageDataInput[] ImagesData);

    public static ImagesDataGroup[] CollectImagesDataFromDirectory(string directoryPath,
        int takeImagesPerClass = int.MaxValue, int maxClassesCount = int.MaxValue)
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
                }).Take(maxClassesCount)
                .ToArray();

        return imagesDataGroups;
    }

    public record TrainTestParts(PreparationImageDataInput[] TrainPart, PreparationImageDataInput[] TestPart);

    public static TrainTestParts TrainTestSplit(ImagesDataGroup[] imagesDataGroups, float testFraction,
        int randomSeed)
    {
        //Console.WriteLine("Total number of classes: " + imagesDataGroups.Length);

        var random = new Random(randomSeed);

        var groupsWithParts = imagesDataGroups.Select(group =>
        {
            var arrayCopy = group.ImagesData.ToArray();
            random.Shuffle(arrayCopy);
            var testPart = arrayCopy.Take((int)Math.Round(arrayCopy.Length * testFraction)).ToArray();
            var trainPart = arrayCopy.Skip(testPart.Length).ToArray();

            //Console.WriteLine($"{group.ClassLabel}. train: {trainPart.Length}. test: {testPart.Length}");

            return new { group.ClassLabel, testPart, trainPart };
        }).ToArray();

        var trainPartGroups = groupsWithParts
            .SelectMany(group => group.trainPart).ToArray();
        var testPartGroups = groupsWithParts
            .SelectMany(group => group.testPart).ToArray();

        return new TrainTestParts(trainPartGroups, testPartGroups);
    }
}