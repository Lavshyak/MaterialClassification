﻿FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
USER $APP_UID
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["MaterialClassification.MainWebApi/MaterialClassification.MainWebApi.csproj", "MaterialClassification.MainWebApi/"]
COPY ["Shared/Shared.csproj", "Shared/"]
RUN dotnet restore "MaterialClassification.MainWebApi/MaterialClassification.MainWebApi.csproj"
COPY . .
WORKDIR "/src/MaterialClassification.MainWebApi"
RUN dotnet build "./MaterialClassification.MainWebApi.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./MaterialClassification.MainWebApi.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MaterialClassification.MainWebApi.dll"]
