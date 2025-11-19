FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PyVips
RUN apt-get update && apt-get install -y \
    libvips-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user architecture /opt/app/architecture
COPY --chown=user:user modules /opt/app/modules
RUN  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# You can add any Python dependencies to requirements.txt
RUN python -m pip install albumentations==1.3.1 
RUN python -m pip install einops==0.8.0 
RUN python -m pip install pillow
RUN python -m pip install geojson==3.1.0 
RUN python -m pip install imagecodecs==2024.9.22 
RUN python -m pip install imageio==2.27.0 
RUN python -m pip install imageio-ffmpeg==0.5.1 
RUN python -m pip install imagesize==1.4.1
RUN python -m pip install imgaug==0.2.6 
RUN python -m pip install importlib_metadata==8.5.0
RUN python -m pip install lxml==4.9.4 
RUN python -m pip install numba==0.60.0 
RUN python -m pip install numpy==1.26.4 
RUN python -m pip install opencv-python==4.10.0.84 
RUN python -m pip install opencv-python-headless==4.11.0.86 
RUN python -m pip install openslide-python==1.4.1 
RUN python -m pip install pandas==2.2.3 
RUN python -m pip install pillow==10.2.0 
RUN python -m pip install PyYAML==6.0.2 
RUN python -m pip install scikit-fuzzy==0.5.0 
RUN python -m pip install scikit-image==0.22.0 
RUN python -m pip install scikit-learn==1.5.2 
RUN python -m pip install scipy==1.14.1 
RUN python -m pip install shapely==2.0.6 
RUN python -m pip install threadpoolctl==3.5.0 
RUN python -m pip install tifffile==2024.9.20 
RUN python -m pip install tiffslide==2.4.0 
RUN python -m pip install timm==1.0.3 

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

ENTRYPOINT ["python", "inference.py"]
