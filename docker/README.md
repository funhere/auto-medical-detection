Docker Support
==============
The Dockerfile in this directory will build Docker images with all the dependencies and code needed to run example notebooks included in this repository.

The following examples show how to build and run the Docker image for CPU and GPU environments. 
Note on some platforms, one needs to manually specify the environment variable for `DOCKER_BUILDKIT`to make sure the build runs well. 
For example, on a Windows machine, this can be done by the powershell command as below, before building the image
```
$env:DOCKER_BUILDKIT=1
```

Once the container is running you can access Jupyter notebooks at http://localhost:8888.

Building and Running with Docker
--------------------------------

<details>
<summary><strong><em>CPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t video_resnet:cpu --build-arg ENV="cpu" .
docker run -p 8888:8888 -d video_resnet:cpu
```

</details>

<details>
<summary><strong><em>GPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t video_resnet:gpu --build-arg ENV="gpu" .
docker run --runtime=nvidia -p 8888:8888 -d video_resnet:gpu
```

</details>

Build Arguments
---------------

There are several build arguments which can change how the image is built. Similar to the `ENV` build argument these are specified during the docker build command.

Build Arg|Description|
---------|-----------|
ENV|Environment to use, options: cpu, gpu|
ANACONDA|Anaconda installation script (defaults to miniconda3 4.6.14)|
