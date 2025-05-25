using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace MaterialClassification.WithLbfgs.DataModels;

public static class ColumnNames
{
    public const string ImagePath = "ImagePath";
    public const string LabelValue = "LabelValue";
    public const string SourceImage = "SourceImage";
    public const string Score = "Score";
    public const string PredictedLabelValue = "PredictedLabelValue";
}

public class PreparationImageDataInput
{
    [ColumnName(ColumnNames.ImagePath)]
    public string ImagePath { get; set; } = null!;

    [ColumnName(ColumnNames.LabelValue)]
    public string LabelValue { get; set; } = null!;
}

public class TrainingImageDataInput : PreparationImageDataInput
{
    [ColumnName(ColumnNames.SourceImage)]
    [ImageType(1, 1)]
    public MLImage SourceImage { get; set; } = null!;
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

public class ProductionImageDataInput
{
    [ColumnName(ColumnNames.SourceImage)]
    [ImageType(1, 1)]
    public MLImage SourceImage { get; set; } = null!;
}

public class ProductionImageDataOutput
{
    // Вероятности принадлежности к классам
    [ColumnName(ColumnNames.Score)]
    public float[] Score { get; set; } = null!;

    // Результат предсказания
    [ColumnName(ColumnNames.PredictedLabelValue)]
    public string PredictedLabelValue { get; set; } = null!;
}