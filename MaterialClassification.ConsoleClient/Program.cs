using System.Net.Http.Json;
Console.WriteLine("Введите путь к файлу с микрофотографией материала:");
var path = Console.ReadLine();
if (!File.Exists(path))
{
    Console.WriteLine("Путь неверный");
    Console.ReadLine();
    return;
}
var fileContent = File.ReadAllBytes(path);
var fileContentBase64 = Convert.ToBase64String(fileContent);
using var httpClient = new HttpClient();
using var response = await httpClient.PostAsJsonAsync(
    "http://localhost:5139/MaterialClassification/Classify",
    fileContentBase64);
var predictionResult = await response.Content.ReadAsStringAsync();
Console.WriteLine(predictionResult);
Console.ReadLine();



