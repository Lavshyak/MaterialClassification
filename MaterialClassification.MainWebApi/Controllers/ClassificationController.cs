using MaterialClassification.MainWebApi.Services;
using Microsoft.AspNetCore.Mvc;
using MaterialClassification.Shared;

namespace MaterialClassification.MainWebApi.Controllers;

[ApiController]
[Route("[controller]/[action]")]
public class ClassificationController : ControllerBase
{
    private readonly ILogger<ClassificationController> _logger;
    private readonly ClassificationTaskSender _classificationTaskSender;
    private readonly ClassificationTaskResultGetter _classificationTaskResultGetter;

    public ClassificationController(ILogger<ClassificationController> logger, ClassificationTaskSender classificationTaskSender,
        ClassificationTaskResultGetter classificationTaskResultGetter)
    {
        _logger = logger;
        _classificationTaskSender = classificationTaskSender;
        _classificationTaskResultGetter = classificationTaskResultGetter;
    }

    [HttpPost]
    public async Task<ClassificationTaskResult> ClassifySync(IFormFile formFile)
    {
        await using var stream = formFile.OpenReadStream();
        
        var taskId = Guid.NewGuid();
        var resultTask = _classificationTaskResultGetter.WaitResult(taskId, CancellationToken.None);
        await _classificationTaskSender.Send(stream, formFile.Length, taskId);
        var result = await resultTask;
        return result;
    }

    [HttpPost]
    public async Task<List<ClassificationTaskResult>> ClassifySyncMultiply(IFormFileCollection formFiles)
    {
        formFiles = this.HttpContext.Request.Form.Files;
        var results = new List<ClassificationTaskResult>();

        foreach (var formFile in formFiles)
        {
            var taskId = Guid.NewGuid();
            var resultTask = _classificationTaskResultGetter.WaitResult(taskId, CancellationToken.None);
            await using var stream = formFile.OpenReadStream();
            await _classificationTaskSender.Send(stream, formFile.Length, taskId);
            var result = await resultTask;
            results.Add(result);
        }

        return results;
    }

    [HttpPost]
    public async Task<Guid> SendToClassify(IFormFile formFile)
    {
        await using var stream = formFile.OpenReadStream();
        var taskId = Guid.NewGuid();
        await _classificationTaskSender.Send(stream, formFile.Length, taskId);

        return taskId;
    }

    [HttpPost]
    public async Task<List<Guid>> SendToClassifyMultiply(IFormFileCollection formFiles)
    {
        formFiles = this.HttpContext.Request.Form.Files;
        var taskIds = new List<Guid>();

        foreach (var formFile in formFiles)
        {
            await using var stream = formFile.OpenReadStream();
            var taskId = Guid.NewGuid();
            await _classificationTaskSender.Send(stream, formFile.Length, taskId);
            taskIds.Add(taskId);
        }

        return taskIds;
    }

    [HttpPost]
    public async Task<ClassificationTaskResult?> WaitForClassificationTaskResult(Guid taskId)
    {
        var result = await _classificationTaskResultGetter.TryGetOrWaitResult(taskId, CancellationToken.None);
        return result;
    }

    [HttpPost]
    public async Task<List<ClassificationTaskResult?>> WaitForClassificationTaskResultMultiply(Guid[] taskIds)
    {
        var results = new List<ClassificationTaskResult?>();

        foreach (var taskId in taskIds)
        {
            var result = await _classificationTaskResultGetter.TryGetOrWaitResult(taskId, CancellationToken.None);
            results.Add(result);
        }

        return results;
    }
}