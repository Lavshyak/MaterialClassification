using System.Net;
using Minio;
using Minio.DataModel.Args;

namespace MaterialClassification.Main;

public class MinioImageRepository
{
    private readonly IMinioClient _minioClient;

    public MinioImageRepository(IMinioClient minioClient)
    {
        _minioClient = minioClient;
    }

    public async Task SendImage(Guid taskId, Stream imageStream)
    {
        var resp = await _minioClient.PutObjectAsync(new PutObjectArgs().WithBucket("ImagesForClassificationTasks").WithObject(taskId.ToString()).WithStreamData(imageStream));
        if (resp is null || resp.ResponseStatusCode != HttpStatusCode.Accepted)
        {
            throw new InvalidOperationException();
        }
    }
}