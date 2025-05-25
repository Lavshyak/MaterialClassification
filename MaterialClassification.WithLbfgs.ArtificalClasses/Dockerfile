FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
ARG BUILD_CONFIGURATION=Debug
WORKDIR /src
COPY ["MaterialClassification.ArtificalClasses/MaterialClassification.ArtificalClasses.csproj", "MaterialClassification.ArtificalClasses/"]
COPY ["MaterialClassification.Training/MaterialClassification.Training.csproj", "MaterialClassification.Training/"]
COPY ["MaterialClassification.DataModels/MaterialClassification.DataModels.csproj", "MaterialClassification.DataModels/"]
RUN dotnet restore "MaterialClassification.ArtificalClasses/MaterialClassification.ArtificalClasses.csproj"
COPY . .
WORKDIR "/src/MaterialClassification.ArtificalClasses"
RUN dotnet build "./MaterialClassification.ArtificalClasses.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Debug
RUN dotnet publish "./MaterialClassification.ArtificalClasses.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 AS base

#USER root
WORKDIR $HOME/tmp
RUN apt update
RUN apt install -y curl
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

#RUN mkdir /app
#WORKDIR /app


FROM base AS final

#RUN apt-get update \
#    && apt-get install -y \
#        python3-pip \
#    && rm -rf /var/lib/apt/lists/*
#RUN ln -sf /usr/bin/python3 /usr/bin/python \
#    && python3 -m pip install tensorflow

WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MaterialClassification.ArtificalClasses.dll"]
