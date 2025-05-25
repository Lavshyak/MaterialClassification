using MaterialClassification.WithImageClassification.MetaInfo;
using Microsoft.ML.Data;

namespace MaterialClassification.WithImageClassification.ProductionDataModels;

public class ProductionImageDataOutput
{
    // Вероятности принадлежности к классам
    [ColumnName(ColumnNames.Score)]
    public float[] Score { get; set; } = null!;

    // Результат предсказания
    [ColumnName(ColumnNames.PredictedLabelValue)]
    public string PredictedLabelValue { get; set; } = null!;
}