using System.Net;
using Minio;
using Minio.DataModel.Args;

namespace MaterialClassification.MainWebApi.Repositories;

public class MinioImagesRepository
{
    private readonly IMinioClient _minioClient;
    private readonly string _bucketName;// => "images-for-classification-tasks";
    
    public MinioImagesRepository(IMinioClient minioClient, IConfiguration configuration)
    {
        _minioClient = minioClient;
        _bucketName = configuration["Minio:BucketNames:Images"] ?? throw new InvalidOperationException();
    }

    public async Task SendImage(Guid taskId, Stream imageStream, long imageSize)
    {
        var resp = await _minioClient.PutObjectAsync(new PutObjectArgs().WithBucket(_bucketName).WithObject(taskId.ToString()).WithObjectSize(imageSize).WithStreamData(imageStream));
        if (resp is null || resp.ResponseStatusCode != HttpStatusCode.OK)
        {
            throw new InvalidOperationException();
        }
    }
}