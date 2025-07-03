using MaterialClassification.MainWebApi.Repositories;

namespace MaterialClassification.MainWebApi.Services;

public class ClassificationTaskSender
{
    private readonly RabbitMqClassificationTaskSenderService _rabbitMqClassificationTaskSenderService;
    private readonly MinioImagesRepository _minioImagesRepository;

    public ClassificationTaskSender(RabbitMqClassificationTaskSenderService rabbitMqClassificationTaskSenderService, MinioImagesRepository minioImagesRepository)
    {
        _rabbitMqClassificationTaskSenderService = rabbitMqClassificationTaskSenderService;
        _minioImagesRepository = minioImagesRepository;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="imageStream"></param>
    /// <param name="imageSize"></param>
    /// <param name="taskId"></param>
    /// <returns>task id</returns>
    public async Task Send(Stream imageStream, long imageSize, Guid taskId)
    {
        await _minioImagesRepository.SendImage(taskId, imageStream, imageSize);
        await _rabbitMqClassificationTaskSenderService.Send(taskId);
    }
}