using MaterialClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;

namespace TrainigLib;

public class Class2
{
    public static IEnumerable<ProductionImageDataInput> CreateImages(int classesCount, int imagesPerClassCount,
        int randomSeed = 1)
    {
        var bitmap = new SKBitmap(220, 220);
        var skImage = SKImage.FromBitmap(bitmap);
        var skData = skImage.Encode();
        using var stream = skData.AsStream();
        var image = MLImage.CreateFromStream(stream);
        var pixels = image.Pixels.ToArray();
        var pixelFormat = image.PixelFormat;

        var random = new Random(randomSeed);

        var dataInputs = Enumerable.Range(0, classesCount).SelectMany(i =>
        {
            var labelValue = $"class_{i}";
            var dataInputs = Enumerable.Range(0, imagesPerClassCount).Select(j =>
            {
                var newPixels = pixels.ToArray();
                random.NextBytes(newPixels);
                var newImage = MLImage.CreateFromPixels(220, 220, pixelFormat, newPixels);
                var dataInput = new ProductionImageDataInput()
                {
                    LabelValue = labelValue,
                    SourceImage = newImage,
                    ImagePath = $"{labelValue}_{j}"
                };
                return dataInput;
            });
            return dataInputs;
        });

        return dataInputs;
    }
}