FROM continuumio/anaconda3
# Note: miniconda causes a segfault with pyhsmm Cython extensions — use anaconda3

LABEL org.opencontainers.image.source=https://github.com/llorente-lab/moseq2-app
LABEL org.opencontainers.image.description="MoSeq2 full pipeline — extract, PCA, model, viz, app"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# System deps: build tools + ffmpeg (video I/O) + libgl1 (headless OpenCV)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ git ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Pin Python 3.11 (satisfies >=3.9,<3.13 across all packages)
RUN conda install -y python=3.11 && conda clean -ay

# numpy<2 and Cython must be present before any pip build step:
# pybasicbayes/pyhsmm list them as build-system requires
RUN conda install -y "numpy<2.0.0" "cython>=3" && conda clean -ay

# PDM build backend (used by all 8 packages)
RUN pip install --no-cache-dir pdm-backend

# ── Source packages ─────────────────────────────────────────────────────────
# Build context is the directory containing all repos as subdirectories.
WORKDIR /src
COPY pybasicbayes          ./pybasicbayes
COPY pyhsmm                ./pyhsmm
COPY pyhsmm-autoregressive ./pyhsmm-autoregressive
COPY moseq2-extract        ./moseq2-extract
COPY moseq2-pca            ./moseq2-pca
COPY moseq2-model          ./moseq2-model
COPY moseq2-viz            ./moseq2-viz
COPY moseq2-app            ./moseq2-app

# ── Install in dependency order ──────────────────────────────────────────────

# 1. pybasicbayes — no internal deps; full install
RUN pip install --no-cache-dir --no-build-isolation /src/pybasicbayes

# 2. pyhsmm external deps not in anaconda base
RUN pip install --no-cache-dir "future>=1.0.0" six nose

# 3. pyhsmm — depends on pybasicbayes via git+https; --no-deps skips the pull
RUN pip install --no-cache-dir --no-build-isolation --no-deps /src/pyhsmm

# 4. pyhsmm-autoregressive — depends on pyhsmm+pybasicbayes via git+https
RUN pip install --no-cache-dir --no-build-isolation --no-deps /src/pyhsmm-autoregressive

# 5. moseq2-extract — no internal moseq2 deps; full install
#    Pulls in: click, cytoolz, h5py, joblib, opencv-python, ruamel.yaml,
#              scikit-image, scikit-learn, scipy, statsmodels, tqdm, tifffile
RUN pip install --no-cache-dir --no-build-isolation /src/moseq2-extract

# 6. moseq2-pca — no internal moseq2 deps; full install
#    Adds: bokeh, chest, dask, dask-jobqueue, distributed, pathspec, psutil, seaborn
RUN pip install --no-cache-dir --no-build-isolation /src/moseq2-pca

# 7. moseq2-viz — no internal moseq2 deps; full install
#    Adds: dtaidistance, networkx, pandas, pyarrow
RUN pip install --no-cache-dir --no-build-isolation /src/moseq2-viz

# 8. moseq2-model — depends on pyhsmm stack via git+https; all external deps
#    (cytoolz, opencv-python, pandas, statsmodels, etc.) already installed above
RUN pip install --no-cache-dir --no-build-isolation --no-deps /src/moseq2-model

# 9. moseq2-app external deps not yet covered by the packages above
RUN pip install --no-cache-dir \
    "fastparquet>=0.4.1" \
    "holoviews>=1.14.7" \
    "ipython>=7.14.0" \
    "ipywidgets<8.0.0" \
    "jinja2>=3.0.1" \
    "jupyter-bokeh>=2.0.3" \
    "jupyter>=1.0.0" \
    "panel>=0.12.6" \
    "plotly>=4.14.3" \
    "qgrid>=1.3.1"

# 10. moseq2-app — all deps satisfied; --no-deps skips internal git+https pulls
RUN pip install --no-cache-dir --no-build-isolation --no-deps /src/moseq2-app

# Jupyter notebook server
EXPOSE 8888
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
