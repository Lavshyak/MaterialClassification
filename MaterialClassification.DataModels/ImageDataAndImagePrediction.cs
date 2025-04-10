using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace MaterialClassification.DataModels;

public class PreparationImageDataInput
{
    public string ImagePath { get; set; } = null!;
    // Правильный класс (для обучения)
    public string LabelValue { get; set; } = null!;
}

public class TrainingImageDataOutput : PreparationImageDataInput
{
    // Вероятности принадлежности к классам
    public float[] Score { get; set; } = null!;
    // Результат предсказания
    public string PredictedLabelValue { get; set; } = null!;
}

public class ProductionImageDataInput : PreparationImageDataInput
{
    [ImageType(1,1)]
    public MLImage SourceImage { get; set; } = null!;
}

public class ProductionImageDataOutput : ProductionImageDataInput
{
    // Вероятности принадлежности к классам
    public float[] Score { get; set; } = null!;
    // Результат предсказания
    public string PredictedLabelValue { get; set; } = null!;
}
