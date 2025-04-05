namespace MaterialClassification;

public class ImageData
{
    public string ImagePath { get; set; } = null!;
    // Правильный класс (для обучения)
    public string Label { get; set; } = null!;
}
public class ImagePrediction : ImageData
{
    // Вероятности принадлежности к классам
    public float[] Score { get; set; } = null!;
    // Результат предсказания
    public string PredictedLabelValue { get; set; } = null!;
}
