using Microsoft.AspNetCore.Mvc;
using Shared;

namespace MaterialClassification.Main.Controllers;

[ApiController]
[Route("[controller]/[action]")]
public class MainController : ControllerBase
{
    private readonly ILogger<MainController> _logger;
    private readonly ClassificationTaskSender _classificationTaskSender;
    private readonly ClassificationTaskResultGetter _classificationTaskResultGetter;

    public MainController(ILogger<MainController> logger, ClassificationTaskSender classificationTaskSender,
        ClassificationTaskResultGetter classificationTaskResultGetter)
    {
        _logger = logger;
        _classificationTaskSender = classificationTaskSender;
        _classificationTaskResultGetter = classificationTaskResultGetter;
    }

    [HttpGet]
    public async Task<ClassificationTaskResult> ClassifySync(IFormFile formFile)
    {
        await using var stream = formFile.OpenReadStream();
        var taskId = await _classificationTaskSender.SendOnly(stream);
        var result = await _classificationTaskResultGetter.WaitResult(taskId, CancellationToken.None);
        return result;
    }

    [HttpGet]
    public async Task<List<ClassificationTaskResult>> ClassifySyncMultiply(IFormFile[] formFiles)
    {
        var results = new List<ClassificationTaskResult>();

        foreach (var formFile in formFiles)
        {
            await using var stream = formFile.OpenReadStream();
            var taskId = await _classificationTaskSender.SendOnly(stream);
            var result = await _classificationTaskResultGetter.WaitResult(taskId, CancellationToken.None);
            results.Add(result);
        }

        return results;
    }

    [HttpGet]
    public async Task<Guid> SendToClassify(IFormFile formFile)
    {
        await using var stream = formFile.OpenReadStream();
        var taskId = await _classificationTaskSender.SendOnly(stream);

        return taskId;
    }

    [HttpGet]
    public async Task<List<Guid>> SendToClassifyMultiply(IFormFile[] formFiles)
    {
        var taskIds = new List<Guid>();

        foreach (var formFile in formFiles)
        {
            await using var stream = formFile.OpenReadStream();
            var taskId = await _classificationTaskSender.SendOnly(stream);
            taskIds.Add(taskId);
        }

        return taskIds;
    }

    [HttpGet]
    public async Task<ClassificationTaskResult?> WaitForClassificationTaskResult(Guid taskId)
    {
        var result = await _classificationTaskResultGetter.TryGetResult(taskId);
        return result;
    }

    [HttpGet]
    public async Task<List<ClassificationTaskResult?>> WaitForClassificationTaskResultMultiply(Guid[] taskIds)
    {
        var results = new List<ClassificationTaskResult?>();

        foreach (var taskId in taskIds)
        {
            var result = await _classificationTaskResultGetter.TryGetResult(taskId);
            results.Add(result);
        }

        return results;
    }
}