Console.WriteLine("Введите путь к файлу с микрофотографией материала:");
var path = "C:\\CodeProjects\\NET\\MaterialClassification\\MaterialClassification.WithLbfgs.Training\\data\\materials_under_microscope\\Copper-1B\\Copper-1B_6.jpg";//Console.ReadLine();
if (!File.Exists(path))
{
    Console.WriteLine("Путь неверный");
    Console.ReadLine();
    return;
}
var fileContent = File.ReadAllBytes(path);
using var fileStream = File.OpenRead(path);
using var httpClient = new HttpClient();
using var content = new MultipartFormDataContent();
content.Add(new StreamContent(fileStream), "formFile", fileStream.Name);
using var response = await httpClient.PostAsync(
    "http://localhost:5139/MaterialClassification/Classify",
    content);
var predictionResult = await response.Content.ReadAsStringAsync();
Console.WriteLine(predictionResult);
Console.ReadLine();



