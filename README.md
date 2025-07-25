<h2>3D Segmentation Multi Object with Point-SAM</h2>

이 저장소는 Point-SAM을 활용하여 원본 3D 포인트 클라우드(BLK306에서 얻은 데이터)에서 자동으로 인스턴스 마스크를 생성하는 맞춤형 파이프라인을 제공합니다.

- 수동 클릭이나 주석 작업이 필요 없습니다.
- 후속 3D 장면 이해를 위한 Segmentation을 가능하게 합니다.

목표는 어떠한 수동 프롬프트 없이도 객체 수준 마스크를 추출하는 것입니다.

## 📚 참고자료
[Point-SAM](https://github.com/zyc00/Point-SAM)


## 🧭 Installation

1. conda 환경 생성 및 활성화

```bash

conda create -n point-sam python=3.10 -y
conda activate point-sam
```
⚠️ Python ≥3.8만 요구하지만, PyTorch 2.1+의 안정성과 호환성을 위해 3.10 추천

2. 필수 패키지 설치(PyTorch, TorchVision, timm)

```bash
# PyTorch 2.1.0 + torchvision 0.16.0 (CUDA 12.1 기준)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# timm >= 0.9.0
pip install timm>=0.9.0
```

3. g++ 9.3.0 설치(Apex, Torkit3D 빌드용)

```bash
conda install -c conda-forge gxx_linux-64=9.3.0 -y
```
🔧 이걸로 g++ 9.3.0을 conda 내부에 설치하게 됨

4. Point-SAM Clone & Submodule update

```bash
git clone https://github.com/zyc00/Point-SAM.git
cd Point-SAM
git submodule update --init --recursive
```

5. Torkit3D 설치 전에 CUDA 12.1 toolkit 설치 진행

step 1) 저장소 추가 및 패키지 다운로드

```bash
# 저장소 PIN 설정
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# CUDA 12.1.1 local installer (.deb)
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 리포지터리 업데이트 & 설치
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

step 2) 설치 확인

```bash
ls /usr/local/ | grep cuda-12.1
  /usr/local/cuda-12.1/bin/nvcc --version
  cuda-12.1
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2023 NVIDIA Corporation
  Built on Mon_Apr__3_17:16:06_PDT_2023
  Cuda compilation tools, release 12.1, V12.1.105
  Build cuda_12.1.r12.1/compiler.32688072_0
```

step 3) 환경변수 설정(conda 세션용)

```bash
mkdir -p ~/.conda/envs/point-sam/etc/conda/activate.d
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
echo 'export CMAKE_PREFIX_PATH=$CUDA_HOME:$CMAKE_PREFIX_PATH' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
```

6. Torkit3D 설치 진행

```bash
git submodule update --init third_party/torkit3d
FORCE_CUDA=1 pip install third_party/torkit3d
```
📌 FORCE_CUDA=1은 CUDA extension 빌드 강제 플래그

7. apex 설치(inference.py만 할 때는 skipp)

8. 나머지 패키지 설치(inference.py할 때는 설치 진행)

```bash
pip install hydra-core omegaconf plyfile open3d einops
```
