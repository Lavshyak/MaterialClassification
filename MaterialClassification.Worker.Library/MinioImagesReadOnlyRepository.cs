using Microsoft.Extensions.Configuration;
using Minio;
using Minio.DataModel.Args;

namespace MaterialClassification.Worker.Library;

public class MinioImagesReadOnlyRepository
{
    private readonly IMinioClient _minioClient;
    private readonly string _bucketName; // => "images-for-classification-tasks";

    public MinioImagesReadOnlyRepository(IMinioClient minioClient, IConfiguration configuration)
    {
        _minioClient = minioClient;
        _bucketName = configuration["Minio:BucketNames:Images"] ??
                      throw new InvalidOperationException("Minio:BucketNames:Images is missing in configuration");
    }

    [Obsolete("Не работает")]
    public async Task<byte[]> GetImage1(Guid taskId)
    {
        var tcs = new TaskCompletionSource<Stream>();
        var getObjectArgs = new GetObjectArgs().WithBucket(_bucketName)
            .WithObject(taskId.ToString()).WithCallbackStream(stream => { tcs.SetResult(stream); });
        var objectStat = await _minioClient.GetObjectAsync(getObjectArgs);
        if (objectStat is null)
        {
            throw new InvalidOperationException();
        }

        await using var stream = await tcs.Task;
        var memStream = new MemoryStream();
        await stream.CopyToAsync(memStream);
        var data = memStream.ToArray();
        return data;
    }

    
    public async Task<byte[]> GetImage(Guid taskId)
    {
        var tcs = new TaskCompletionSource<byte[]>();
        var objectStat = await _minioClient.GetObjectAsync(new GetObjectArgs().WithBucket(_bucketName)
            .WithObject(taskId.ToString()).WithCallbackStream(stream =>
            {
                var memStream = new MemoryStream();
                stream.CopyTo(memStream);
                var arr = memStream.ToArray();
                tcs.SetResult(arr);
            }));
        
        if (objectStat is null)
        {
            throw new InvalidOperationException();
        }

        byte[] data = await tcs.Task;

        return data;
    }
}