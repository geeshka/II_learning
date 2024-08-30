# Используем образ CUDA с поддержкой Ubuntu 22.04 и нужной версии CUDA
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Установим переменные среды для CUDA
ENV CUDA_HOME /usr/local/cuda
ENV PATH ${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Установка основных зависимостей и инструментов
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    build-essential cmake git python3-pip python3-dev \
    libglu1-mesa-dev libgles2-mesa-dev libxt-dev libx11-dev sudo v4l-utils xorg-dev \
    libglew-dev libglfw3-dev libflann-dev libeigen3-dev libboost-all-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgl1-mesa-dev libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.9.7.*-1+cuda12.2 \
    libcudnn8-dev=8.9.7.*-1+cuda12.2 \
    && apt-mark hold libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y dbus-x11

# Установка необходимых Python-библиотек
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow keras bing-image-downloader numpy pandas matplotlib opencv-python scipy
RUN pip3 install gdown
RUN pip3 install scikit-learn
# Скопировать скрипты в контейнер
COPY . /app

# Задать рабочую директорию
WORKDIR /app

# Запуск основного скрипта
CMD ["python3", "learning.py"]
