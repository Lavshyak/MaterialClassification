using Microsoft.AspNetCore.Mvc;

namespace MaterialClassification.WebApi.Controllers;

[ApiController]
[Route("[controller]/[action]")]
public class MaterialClassificationController : ControllerBase
{
    [HttpPost]
    public string Classify([FromBody] string fileContentBase64,
        [FromServices] MaterialClassificationService materialClassificationService,
        [FromServices] IConfiguration configuration)
    {
        var filesDirectoryPath = configuration["ImageFilesDirectoryPath"]
                                 ?? throw new InvalidOperationException();
        var fileContent = Convert.FromBase64String(fileContentBase64);
        var filePath = Path.Combine(filesDirectoryPath, Guid.NewGuid().ToString());
        Directory.CreateDirectory(filesDirectoryPath);
        System.IO.File.WriteAllBytes(filePath, fileContent);
        var predictionResult = materialClassificationService.Predict(filePath);
        return predictionResult;
    }
}




