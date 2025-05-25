FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["MaterialClassification.ImageClassification.Train/MaterialClassification.ImageClassification.Train.csproj", "MaterialClassification.ImageClassification.Train/"]
RUN dotnet restore "MaterialClassification.ImageClassification.Train/MaterialClassification.ImageClassification.Train.csproj"
COPY . .
WORKDIR "/src/MaterialClassification.ImageClassification.Train"
RUN dotnet build "./MaterialClassification.ImageClassification.Train.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./MaterialClassification.ImageClassification.Train.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM tensorflow/tensorflow:2.3.1-gpu AS base
USER $APP_UID

WORKDIR $HOME/tmp
RUN apt-get update
RUN apt-get install -y curl
RUN curl -L https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
RUN chmod +x ./dotnet-install.sh
ENV DOTNET_ROOT=$HOME/.dotnet
RUN ./dotnet-install.sh --version 9.0.4 --runtime dotnet
ENV PATH=$PATH:$DOTNET_ROOT:$DOTNET_ROOT/tools
#Process terminated. Couldn't find a valid ICU package installed on the system. 
#Please install libicu (or icu-libs) using your package manager and try again. 
#Alternatively you can set the configuration flag System.Globalization.Invariant 
#to true if you want to run with no globalization support. 
#Please see https://aka.ms/dotnet-missing-libicu for more information.
RUN apt-get install -y libicu-dev 

WORKDIR /app


FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MaterialClassification.ImageClassification.Train.dll"]
