FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
USER $APP_UID
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["MaterialClassification.WebApi/MaterialClassification.WebApi.csproj", "MaterialClassification.WebApi/"]
RUN dotnet restore "MaterialClassification.WebApi/MaterialClassification.WebApi.csproj"
COPY . .
WORKDIR "/src/MaterialClassification.WebApi"
RUN dotnet build "MaterialClassification.WebApi.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "MaterialClassification.WebApi.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MaterialClassification.WebApi.dll"]
