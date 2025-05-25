using MaterialClassification.WithImageClassification.MetaInfo;
using Microsoft.ML.Data;

namespace MaterialClassification.WithImageClassification.ProductionDataModels;

public class ProductionImageDataInput
{
    [ColumnName(ColumnNames.SourceImageBytes)]
    public byte[] SourceImageBytes { get; set; } = null!;
}