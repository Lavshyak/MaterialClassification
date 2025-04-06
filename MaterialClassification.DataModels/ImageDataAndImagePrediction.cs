using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace MaterialClassification.DataModels;

public class ImageData
{
    public string ImagePath { get; set; } = null!;
    // Правильный класс (для обучения)
    public string Label { get; set; } = null!;
    
    [ImageType(220,220)]
    public MLImage ResizedImage { get; set; } = null!;
}

public class ImagePrediction
{
    // Вероятности принадлежности к классам
    public float[] Score { get; set; } = null!;
    // Результат предсказания
    public string PredictedLabelValue { get; set; } = null!;
}

public class TrainingImagePrediction : ImageData
{
    // Вероятности принадлежности к классам
    public float[] Score { get; set; } = null!;
    // Результат предсказания
    public string PredictedLabelValue { get; set; } = null!;
}
