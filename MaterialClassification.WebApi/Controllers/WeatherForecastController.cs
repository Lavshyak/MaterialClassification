using Microsoft.AspNetCore.Mvc;

namespace MaterialClassification.WebApi.Controllers;

[ApiController]
[Route("[controller]/[action]")]
public class MaterialClassificationController : ControllerBase
{
    [HttpPost]
    public string Classify(IFormFile formFile,
        [FromServices] MaterialClassificationService materialClassificationService,
        [FromServices] IConfiguration configuration)
    {
        using var stream = formFile.OpenReadStream();
        var predictionResult = materialClassificationService.Predict(stream);
        return predictionResult;
    }
}




