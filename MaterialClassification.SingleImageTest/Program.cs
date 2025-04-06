using System.Runtime.CompilerServices;
using System.Text;
using MaterialClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;

var modelPath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.Training\\bin\\Debug\\net9.0\\model_10_16_11.zip";
var mlContext = new MLContext();
ITransformer trainedModel = mlContext.Model.Load(modelPath, out var inputSchema);
Console.WriteLine("Модель загружена.");
var predictionEngine = mlContext.Model.CreatePredictionEngine
    <ImageData, ImagePrediction>
    (trainedModel);
Console.WriteLine("PredictionEngine создан.");

var b = new DataViewSchema.Builder();
b.AddColumns(inputSchema);
b.AddColumn("ResizedImage", new ImageDataViewType(220,220));
var schema = b.ToSchema();
var outputSchema = trainedModel.GetOutputSchema(schema) ?? throw new InvalidOperationException();

var imagePath =
    "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.Training\\data\\materials_under_microscope\\Copper-1B\\Copper-1B_6.jpg";

var skBitmap = SKBitmap.Decode(filename: imagePath);
var resized = ResizeWithCrop(skBitmap, 220, 220);
var image = MLImage.CreateFromPixels(220,220, MLPixelFormat.Rgba32, resized.Bytes);

var imageData = new ImageData { ResizedImage = image };
var sb = new StringBuilder();
var prediction = predictionEngine.Predict(imageData);
sb.AppendLine(prediction.PredictedLabelValue);
var labelKeyColumn = outputSchema["LabelKey"];
VBuffer<ReadOnlyMemory<char>> keyValues = default;
labelKeyColumn.GetKeyValues(ref keyValues);
var items = keyValues.Items().ToArray();
if (items.Length != prediction.Score.Length)
{
    throw new InvalidOperationException();
}
var classesScores = items.Zip(prediction.Score,
        (keyValuePair, predictionScore) => new
            { ClassName = new string(keyValuePair.Value.Span), PredictionScore = predictionScore })
    .ToArray();
sb.AppendLine("Вероятности принадлежности к классам материалов:");
sb.AppendLine(string.Join(", ",
    classesScores.OrderByDescending(cs => cs.PredictionScore).Take(5)
        .Select(cs => $"{cs.ClassName}: {cs.PredictionScore}")));
Console.WriteLine(sb.ToString());


SKBitmap ResizeWithCrop(SKBitmap image, int width, int height, ImageResizeModeClone mode = ImageResizeModeClone.CropAnchorCentral){
    float widthAspect = (float)width / image.Width;
                float heightAspect = (float)height / image.Height;
                int destX = 0;
                int destY = 0;
                float aspect;
    
                if (heightAspect < widthAspect)
                {
                    aspect = widthAspect;
                    switch (mode)
                    {
                        case ImageResizeModeClone.CropAnchorTop:
                            destY = 0;
                            break;
                        case ImageResizeModeClone.CropAnchorBottom:
                            destY = (int)(height - (image.Height * aspect));
                            break;
                        default:
                            destY = (int)((height - (image.Height * aspect)) / 2);
                            break;
                    }
                }
                else
                {
                    aspect = heightAspect;
                    switch (mode)
                    {
                        case ImageResizeModeClone.CropAnchorLeft:
                            destX = 0;
                            break;
                        case ImageResizeModeClone.CropAnchorRight:
                            destX = (int)(width - (image.Width * aspect));
                            break;
                        default:
                            destX = (int)((width - (image.Width * aspect)) / 2);
                            break;
                    }
                }
    
                int destWidth = (int)(image.Width * aspect);
                int destHeight = (int)(image.Height * aspect);
    
                SKBitmap dst = new SKBitmap(width, height, isOpaque: true);
    
                SKRect srcRect = new SKRect(0, 0, image.Width, image.Height);
                SKRect destRect = new SKRect(destX, destY, destX + destWidth, destY + destHeight);
    
                using SKCanvas canvas = new SKCanvas(dst);
                using SKPaint paint = new SKPaint() { FilterQuality = SKFilterQuality.High };
    
                canvas.DrawBitmap(image, srcRect, destRect, paint);
    
                return dst;
}

enum ImageResizeModeClone
{
    /// <summary>
    /// Pads the resized image to fit the bounds of its container.
    /// </summary>
    Pad,
    /// <summary>
    /// Ignore aspect ratio and squeeze/stretch into target dimensions.
    /// </summary>
    Fill,
    /// <summary>
    /// Resized image to fit the bounds of its container using cropping with top anchor.
    /// </summary>
    CropAnchorTop,
    /// <summary>
    /// Resized image to fit the bounds of its container using cropping with bottom anchor.
    /// </summary>
    CropAnchorBottom,
    /// <summary>
    /// Resized image to fit the bounds of its container using cropping with left anchor.
    /// </summary>
    CropAnchorLeft,
    /// <summary>
    /// Resized image to fit the bounds of its container using cropping with right anchor.
    /// </summary>
    CropAnchorRight,
    /// <summary>
    /// Resized image to fit the bounds of its container using cropping with central anchor.
    /// </summary>
    CropAnchorCentral,
}