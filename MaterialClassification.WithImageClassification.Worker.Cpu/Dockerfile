FROM mcr.microsoft.com/dotnet/runtime:9.0 AS base
USER $APP_UID
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["MaterialClassification.WithImageClassification.Worker.Cpu/MaterialClassification.WithImageClassification.Worker.Cpu.csproj", "MaterialClassification.WithImageClassification.Worker.Cpu/"]
COPY ["MaterialClassification.WithImageClassification.Worker.Library/MaterialClassification.WithImageClassification.Worker.Library.csproj", "MaterialClassification.WithImageClassification.Worker.Library/"]
COPY ["MaterialClassification.Worker.Library/MaterialClassification.Worker.Library.csproj", "MaterialClassification.Worker.Library/"]
COPY ["Shared/Shared.csproj", "Shared/"]
COPY ["MaterialClassification.WithImageClassification.ColumnNames/MaterialClassification.WithImageClassification.ColumnNames.csproj", "MaterialClassification.WithImageClassification.ColumnNames/"]
COPY ["MaterialClassification.WithImageClassification.ProductionDataModels/MaterialClassification.WithImageClassification.ProductionDataModels.csproj", "MaterialClassification.WithImageClassification.ProductionDataModels/"]
RUN dotnet restore "MaterialClassification.WithImageClassification.Worker.Cpu/MaterialClassification.WithImageClassification.Worker.Cpu.csproj"
COPY . .
WORKDIR "/src/MaterialClassification.WithImageClassification.Worker.Cpu"
RUN dotnet build "./MaterialClassification.WithImageClassification.Worker.Cpu.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./MaterialClassification.WithImageClassification.Worker.Cpu.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MaterialClassification.WithImageClassification.Worker.Cpu.dll"]
