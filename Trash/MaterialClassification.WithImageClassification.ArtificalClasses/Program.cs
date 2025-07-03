using System.Diagnostics;
using System.Text.Json;
using MaterialClassification.WithImageClassification.Training.Library;
using Microsoft.ML;
using Microsoft.ML.Data;
using SkiaSharp;

namespace MaterialClassification.WithImageClassification.ArtificalClasses;

public class Program
{
    public record Config(
        int ImageHeight,
        int ImageWidth,
        float OffsetColor,
        string InceptionTensorFlowModelFilePath,
        string ImagesRootPath,
        int ImagesForTestPerClass,
        int ImagesForTrainPerClass,
        int RandomSeed
    );

    static void Main(string[] args)
    {
        /*tf.compat.v1.disable_eager_execution();

        var a = tf.constant(np.arange(1e7).reshape((1000, 10000)), TF_DataType.TF_FLOAT);
        var b = tf.constant(np.arange(1e7).reshape((10000, 1000)), TF_DataType.TF_FLOAT);
        var c = tf.matmul(a, b);

        using (var sess = tf.Session())
        {
            Console.WriteLine("running");
            var result = sess.run(c);
            sess.run(c);
            sess.run(c);
            sess.run(c);
            sess.run(c);
            sess.run(c);
            Console.WriteLine(result.shape);
        }*/
        
        
        Console.WriteLine("Hello World!");
        
        var config = JsonSerializer.Deserialize<Config>(File.ReadAllText("config.json")) ??
                     throw new InvalidOperationException();
        
        // прогрев
        {
            for (int iTest = 1; iTest <= 5; iTest++)
            {
                var classesCount = (int)Math.Pow(2, iTest);

                MLContext mlContext = new MLContext();
                mlContext.GpuDeviceId = 0;

                Console.WriteLine("-------images generation");
                // подготовка
                var trainPartDataInputsDataView = Generate1(mlContext, classesCount, 1, 1);
                var testPartDataInputsDataView = Generate1(mlContext, classesCount, 10, 1);
                Console.WriteLine("-------end of images generation");

                var preparationEstimator = Methods.GenerateFromImageAndLabelValuePreparationEstimator(mlContext);
                var preparationTransformer = preparationEstimator.Fit(trainPartDataInputsDataView);
                
                trainPartDataInputsDataView = preparationTransformer.Transform(trainPartDataInputsDataView);
                testPartDataInputsDataView = preparationTransformer.Transform(testPartDataInputsDataView);
                
                // основной пайплайн
                var estimator = Methods.GenerateClassificationEstimator(mlContext);

                Console.WriteLine("-------train");
                // тренировка
                var model = estimator.Fit(trainPartDataInputsDataView);
                Console.WriteLine("-------end of train");

                Console.WriteLine("-------test");
                var stopwatch = Stopwatch.StartNew();
                // тестирование
                IDataView predictions = model.Transform(testPartDataInputsDataView);
                stopwatch.Stop();
                Console.WriteLine("-------end of test");
                Console.WriteLine($"{classesCount}\t{stopwatch.Elapsed.TotalMilliseconds}");
            }

            Console.WriteLine("прогрев окончен");
        }

        for (int iTest = 1; iTest <= 12; iTest++)
        {
            var classesCount = (int)Math.Pow(2, iTest);

            MLContext mlContext = new MLContext();
            mlContext.GpuDeviceId = 0;
            // подготовка
            var trainPartDataInputsDataView = Generate1(mlContext, classesCount, 1, 1);
            var testPartDataInputsDataView = Generate1(mlContext, 1, 1, 1);

            var preparationEstimator = Methods.GenerateFromImageAndLabelValuePreparationEstimator(mlContext);
            var preparationTransformer = preparationEstimator.Fit(trainPartDataInputsDataView);
                
            trainPartDataInputsDataView = preparationTransformer.Transform(trainPartDataInputsDataView);
            testPartDataInputsDataView = preparationTransformer.Transform(testPartDataInputsDataView);
            
            // основной пайплайн
            var estimator = Methods.GenerateClassificationEstimator(mlContext);

            // тренировка
            var model = estimator.Fit(trainPartDataInputsDataView);

            var stopwatch = Stopwatch.StartNew();
            // тестирование
            IDataView predictions = model.Transform(testPartDataInputsDataView);
            stopwatch.Stop();
            Console.WriteLine($"{classesCount}\t{stopwatch.Elapsed.TotalMilliseconds}");
        }
    }
    
    public static IDataView Generate1(MLContext mlContext, int classesCount,
        int imagesPerClass, int randomSeed)
    {
        var random = new Random(randomSeed);
        const int widthAndHeight = 300;
        var dataInputs = Enumerable.Range(1, classesCount).SelectMany(iClass =>
        {
            return Enumerable.Range(0, imagesPerClass).Select(iImage =>
            {
                var pixelBytes = new byte[widthAndHeight * widthAndHeight * 4];
                random.NextBytes(pixelBytes);
         
                var skImage = SKImage.FromPixels(new SKImageInfo(widthAndHeight, widthAndHeight, SKColorType.Bgra8888), SKData.CreateCopy(pixelBytes));
                var sourceImageBytes = skImage.Encode(SKEncodedImageFormat.Jpeg, 1).ToArray();
                
                var dataInput = new TrainingImageDataInput { LabelValue = $"label_{iClass}", SourceImageBytes = sourceImageBytes };
                return dataInput;
            });
        });

        var dataInputsDataView = mlContext.Data.LoadFromEnumerable(dataInputs);

        return dataInputsDataView;
    }
}