using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;

var mlContext = new MLContext();

var bitmap = new SKBitmap(10, 10);
var skImage = SKImage.FromBitmap(bitmap);
var skData = skImage.Encode();
var stream = skData.AsStream();
var image = MLImage.CreateFromStream(stream);

MLImage[] images = [image];

var problemDatas = images.Select(image => new ProblemData() { Image = image }).ToArray();

var dataView = mlContext.Data.LoadFromEnumerable(problemDatas);

var enumerable1 = mlContext.Data.CreateEnumerable<ProblemData>(dataView, reuseRowObject: false);
var enumerable2 = mlContext.Data.CreateEnumerable<ProblemData>(dataView, reuseRowObject: false);

Console.WriteLine("ok");

public class ProblemData
{
    [ImageType(10, 10)]
    public MLImage Image { get; set; }
}