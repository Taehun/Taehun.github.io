---
layout: post
title: "2022년 MLOps 툴 스택"
categories: mlops
tags: [mlops, mlops_tools, mlops_tool_stack, mymlops_com]
toc: true
comments: true
---

[![MyMLOps](https://github.com/Taehun/taehun.github.io/blob/master/imgs/MyMLOps.png?raw=true)](https://mymlops.com/stacks/2APSE7MUnl6A1cn7LAl8Ic5zKtg)
_<center>제가 선택한 MLOps 툴체인. 당신의 선택은?</center>_

최근에 [MyMLOps](https://mymlops.com/)이라는 흥미로운 사이트를 발견했습니다. MLOps의 각 항목별로 사용하는 도구를 선택하고, 선택된 도구들을 위와 같이 툴체인 형태로 시각화해주는 사이트 입니다. [MyMLOps](https://mymlops.com/)에 있는 도구들과 [MyMLOps](https://mymlops.com/)에는 없지만 도움이 될만한 MLOps 관련 도구들을 소개해 드리겠습니다. 여러 항목에 중복되는 도구들은 다양한 기능을 제공하는 통합 솔루션 입니다.

## MyMLOps의 MLOps 도구들

### 실험 추적 (Experiment tracking)

머신러닝 프로젝트는 다양한 모델, 파라메터 및 학습 데이터로 여러 실험을 실행해야 합니다. 실험 추적 툴은 다양한 실험에 필요한 모든 정보를 기록합니다. 이를 통해 실험 구성 요소의 버전과 결과를 추적하고 여러 실험을 비교할 수 있습니다.

- [Kedro](https://kedro.readthedocs.io/en/stable/): Kedro는 재현 가능하고 유지 가능한 모듈식 데이터 과학 코드를 만들기 위한 오픈 소스 Python 프레임워크입니다.
- [ModelDB](https://github.com/VertaAI/modeldb): ModelDB는 머신러닝 모델 버전 관리, 메타데이터 및 실험 관리를 위한 오픈 소스 시스템입니다.
- [MLflow](https://mlflow.org/): MLflow는 End-to-End 머신러닝 라이프싸이클을 관리하기 위한 오픈 소스 플랫폼입니다.
- [Determined](https://www.determined.ai/): Determined은 모델 생성을 빠르고 쉽게 만드는 오픈 소스 딥러닝 학습 플랫폼입니다.
- [Weights & Biases](https://wandb.ai/): Weights & Biases은 데이터 세트에서 프로덕션 모델에 이르기까지 머신러닝 파이프라인 각 부분을 추적하고 시각화하는 도구입니다.
- [Polyaxon](https://github.com/polyaxon/polyaxon): Polyaxon은 대규모 딥러닝 애플리케이션을 구축, 학습 및 모니터링하기 위한 플랫폼입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.
- [TensorBoard](https://github.com/tensorflow/tensorboard): TensorBoard는 ML 모델을 최적화하고 디버그하기 위한 시각화 툴킷입니다.

> TensorBoard는 ML 실험 결과 분석을 위해 가장 널리 사용하는 도구 중 하나 입니다. 하지만, TensorBoard의 기능 대부분은 하나의 실험에 한정되어 있어 여러 실험을 추적하려면 다른 도구와 조합해야 합니다.

### 실험 (Experimentation)

실험 (Experimentation)은 데이터를 분석하고, 프로토타입 모델을 만들며 학습을 구현하는 과정 입니다. 대부분 데이터 과학에 자주 사용하는 [Jupyter Notebook](https://jupyter.org/) 환경을 제공하는 도구들 입니다.

- [Kubeflow](https://www.kubeflow.org/): Kubeflow는 쿠버네티스 환경에서 ML 워크플로우를 단순하고, 이식 가능하며, 확장 가능하게 합니다.
- [Jupyter](https://jupyter.org/): Jupyter Notebook은 라이브 코드, 방정식, 시각화 및 텍스트가 포함된 문서를 만들고 공유할 수 있는 오픈 소스 웹 애플리케이션입니다.
- [Polynote](https://polynote.org/latest/): Polynote는 최고 수준의 Scala를 지원하는 데이터 과학 노트북입니다.
- [Spyder](https://www.spyder-ide.org/): Spyder는 Python을 위한, Python으로 작성된 무료 오픈 소스 과학 환경입니다.
- [Apache Zepellin](https://zeppelin.apache.org/): Apache Zepellin은 SQL, Scala, Python, R 등을 사용하여 데이터 기반의 대화형 데이터 분석 및 협업 문서를 지원하는 웹 기반 노트북입니다.

### 데이터 버전 관리 (Data versioning)

데이터 버전 관리 도구를 사용하면 다양한 버전의 데이터 세트를 잘 구성된 방식으로 관리 할 수 있습니다. 이를 통해 ML 팀은 데이터 변경이 모델 성능에 미치는 영향을 식별하고 데이터 세트가 어떻게 변화하는지 파악하는 것과 같은 통찰력을 얻을 수 있습니다. 데이터 버전 관리는 ML 모델 계보 재현 및 추적을 지원합니다.

- [Dolt](https://github.com/dolthub/dolt): Dolt는 데이터를 위한 Git 입니다. SQL 쿼리를 실행하거나 명령줄 인터페이스로 데이터를 조작할 수 있습니다.
- [DVC](https://dvc.org/): DVC(Data Version Control)는 머신러닝 프로젝트 재현성을 위한 오픈 소스 버전 관리 시스템입니다.
- [Weights & Biases](https://wandb.ai/): Weights & Biases은 데이터 세트에서 프로덕션 모델에 이르기까지 머신러닝 파이프라인 각 부분을 추적하고 시각화하는 도구입니다.
- [Git LFS](https://git-lfs.github.com/): Git LFS는 Git으로 대용량 파일의 버전 관리할 수 있는 오픈 소스 도구입니다.
- [Pachyderm](https://www.pachyderm.com/): Pachyderm은 MLOps용 데이터 버전 관리 및 파이프라인을 위한 도구입니다.
- [lakeFS](https://lakefs.io/): lakeFS는 모든 데이터 크기에서 작동하는 오브젝트 스토리지 버킷을 위한 Git입니다.

### 코드 버전 관리 (Code versioning)

데이터 분석이나 ML 실험에 사용한 노트북 파일이나 파이썬 소스 코드 파일등의 코드 버전 관리 도구들 입니다.

- [GitLab](https://about.gitlab.com/): GitLab은 무료 공개 및 비공개 Git 저장소를 제공하는 웹 기반 DevOps 플랫폼입니다.
- [Git](https://git-scm.com/): Git은 파일의 변경 사항을 추적하는 데 사용되는 오픈 소스 도구입니다.

> 요즘 코드 버전 관리에 Git을 사용하지 않는 조직은 없을것 입니다. 이 항목은 [GitHub](https://github.com), [GitLab](https://about.gitlab.com/)과 같은 Git 원격 저장소 호스팅 서비스를 어떤 것을 사용하는지 선택하는 항목이라고 보시는게 좋을것 같습니다. 인기는 없지만 AWS의 [AWS CodeCommit](https://aws.amazon.com/ko/codecommit/), GCP의 [Cloud Source Repositories](https://source.cloud.google.com/), Azure의 [Azure Repos](https://azure.microsoft.com/en-us/services/devops/repos/)등 공용 클라우드 Git 저장소 서비스들도 있습니다.

### 머신러닝 파이프라인 (Pipeline orchestration)

머신러닝 파이프라인 (ML Pipeline)은 상용 환경에서 복잡한 ML 학습과 추론 작업을 단계별로 구성하고 자동화 합니다. 데이터/컨셉 드리프트 발생시 파이프라인을 자동으로 트리거하여 최신 데이터에서 모델을 재학습합니다.

- [Apache Airflow](https://airflow.apache.org/): Airflow는 Apache 커뮤니티에서 프로그래밍 방식으로 워크플로우를 작성, 예약 및 모니터링하기 위해 만든 플랫폼입니다.
- [Argo Workflows](https://github.com/argoproj/argo-workflows): Argo Workflows는 쿠버네티스 환경에서 병렬 작업을 오케스트레이션하기 위한 오픈 소스 컨테이너 네이티브 워크플로우 엔진입니다.
- [Luigi](https://github.com/spotify/luigi): Luigi는 배치 작업의 복잡한 파이프라인을 구축하는데 도움이 되는 Python 패키지입니다.
- [Kubeflow](https://www.kubeflow.org/): Kubeflow는 쿠버네티스 환경에서 ML 워크플로우를 단순하고, 이식 가능하며, 확장 가능하게 합니다.
- [Kedro](https://kedro.readthedocs.io/en/stable/): Kedro는 재현 가능하고 유지 가능한 모듈식 데이터 과학 코드를 만들기 위한 오픈 소스 Python 프레임워크입니다.
- [Nextflow](https://www.nextflow.io/): Nextflow는 소프트웨어 컨테이너를 사용하여 확장 가능하고 재현 가능한 과학 워크플로우를 가능하게 합니다.
- [Dagster](https://github.com/dagster-io/dagster): Dagster는 데이터 자산을 개발, 생성 및 관찰하기 위한 오케스트레이션 플랫폼입니다.
- [Apache Beam](https://beam.apache.org/): Apache Beam은 미션 크리티컬 프로덕션 워크로드를 위한 배치 및 스트리밍 데이터 처리 도구입니다.
- [ZenML](https://github.com/zenml-io/zenml): ZenML은 프로덕션-준비 (production-ready) 된 머신러닝 파이프라인을 생성하기 위한 확장 가능한 오픈 소스 MLOps 프레임워크입니다.
- [Flyte](https://flyte.org/): Flyte는 미션 크리티컬 데이터 및 대규모 머신러닝 프로세스를 위한 워크플로우 자동화 플랫폼입니다.
- [Prefect](https://github.com/prefecthq/prefect): Prefect는 최신 인프라용으로 설계된 오픈 소스 워크플로우 관리 시스템입니다.
- [Ray](https://www.ray.io/): Ray는 컴퓨팅 집약적인 Python 워크로드를 간편하게 확장할 수 있는 오픈 소스 도구입니다.
- [DVC](https://dvc.org/): DVC(Data Version Control)는 머신러닝 프로젝트 재현성을 위한 오픈 소스 버전 관리 시스템입니다.
- [Polyaxon](https://github.com/polyaxon/polyaxon): Polyaxon은 대규모 딥러닝 애플리케이션을 구축, 학습 및 모니터링하기 위한 플랫폼입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.
- [Pachyderm](https://www.pachyderm.com/): Pachyderm은 MLOps용 데이터 버전 관리 및 파이프라인을 위한 도구입니다.
- [Orchest](https://github.com/orchest/orchest): Orchest는 Jupyter 노트북 및 Python 스크립트를 시각적 파이프라인 편집기와 결합하는 데이터 파이프라인 생성 도구입니다.

> 이 항목에 나열된 도구들은 통합 솔루션들이 많습니다.

### 런타임 엔진 (Runtime engine)

런타임 엔진 (Runtime engine)은 ML 코드가 실행되는 환경을 제공합니다. 멀티 노드 환경에서 병렬 실행을 위해 쿠버네티스 클러스터에 설치하여 API를 제공하거나 자체 클러스터를 런타임 환경으로 제공 합니다.

- [Ray](https://www.ray.io/): Ray는 컴퓨팅 집약적인 Python 워크로드를 간편하게 확장할 수 있는 오픈 소스 도구입니다.
- [Nuclio](https://github.com/nuclio/nuclio): Nuclio는 데이터, I/O 및 컴퓨팅 집약적 워크로드에 중점을 둔 고성능 서버리스 프레임워크 입니다.
- [Dask](https://www.dask.org/): Dask는 Python의 병렬 컴퓨팅을 위한 유연한 라이브러리입니다.
- [Horovod](https://github.com/horovod/horovod): Horovod를 사용하면 여러 GPU에서 모델을 병렬로 쉽게 훈련할 수 있습니다.
- [Apache Spark](https://spark.apache.org/): Apache Spark는 대규모 데이터 세트 및 분산 컴퓨팅을 위한 데이터 처리 프레임워크입니다.

### 아티팩트 추적 (Artifact tracking)

실험 및 ML 파이프라인 전반에 걸쳐 데이터세트, 모델 및 중간 결과물을 저장하고 추적합니다.

- [Kubeflow](https://www.kubeflow.org/): Kubeflow는 쿠버네티스 환경에서 ML 워크플로우를 단순하고, 이식 가능하며, 확장 가능하게 합니다.
- [MLflow](https://mlflow.org/): MLflow는 End-to-End 머신러닝 라이프싸이클을 관리하기 위한 오픈 소스 플랫폼입니다.
- [Polyaxon](https://github.com/polyaxon/polyaxon): Polyaxon은 대규모 딥러닝 애플리케이션을 구축, 학습 및 모니터링하기 위한 플랫폼입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.
- [Pachyderm](https://www.pachyderm.com/): Pachyderm은 MLOps용 데이터 버전 관리 및 파이프라인을 위한 도구입니다.

> 아티팩트? ML 파이프라인의 각 단계나 ML 실험의 실행 결과물들을 아티팩트 (Artifact)라고 합니다. 학습된 모델 체크포인트 파일, 변환된 데이터의 CSV/ Parquet/JSON 파일, 실행 로그 파일등이 여기에 해당 됩니다.

> 아티팩트 추적 전용 툴은 없습니다. 아티팩트 추적은 ML 파이프라인 도구나 ML 실험 추적 도구의 기능에 포함되어 있습니다.

### 모델 레지스트리 (Model registry)

모델 레지스트리 (Model registry)는 중앙 집중식 모델 저장소를 의미 합니다. 학습된 모델의 버전 관리 및 추적 기능을 제공하고, 배포를 위해 모델 입/출력 인터페이스 및 런타임 패키지들에 대한 정보들을 저장 합니다.

- [ModelDB](https://github.com/VertaAI/modeldb): ModelDB는 머신러닝 모델 버전 관리, 메타데이터 및 실험 관리를 위한 오픈 소스 시스템입니다.
- [MLflow](https://mlflow.org/): MLflow는 End-to-End 머신러닝 라이프싸이클을 관리하기 위한 오픈 소스 플랫폼입니다.
- [Determined](https://www.determined.ai/): Determined은 모델 생성을 빠르고 쉽게 만드는 오픈 소스 딥러닝 학습 플랫폼입니다.
- [Weights & Biases](https://wandb.ai/): Weights & Biases은 데이터 세트에서 프로덕션 모델에 이르기까지 머신러닝 파이프라인 각 부분을 추적하고 시각화하는 도구입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.

### 모델 서빙 (Model serving)

REST API나 gRPC 엔드포인트를 생성하여 요청를 받으면 추론 결과를 응답합니다. 모델의 배포 타겟이 클라우드일때 사용하는 도구들 입니다.

- [Seldon Core](https://seldon.io): Seldon Core는 쿠버네티스 환경에 머신러닝 모델을 대규모로 배포하기 위한 오픈 소스 플랫폼입니다.
- [BentoML](https://www.bentoml.com/): BentoML은 ML 모델 배포를 단순화하고 몇 분 만에 프로덕션 규모로 모델을 서빙할 수 있는 오픈 플랫폼입니다.
- [NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server): NVIDIA Triton Inference Server는 모델 배포 및 실행을 표준화하고 프로덕션 환경에서 빠르고 확장 가능한 AI를 제공하는 오픈 소스 추론 서빙 소프트웨어입니다.
- [TensorFlow Serving](https://github.com/tensorflow/serving): TensorFlow Serving은 프로덕션 환경을 위해 설계된 머신러닝 모델을 위한 유연한 고성능 서빙 시스템입니다.
- [KServe](https://kserve.github.io/): KServe는 쿠버네티스 환경에서 높은 확장성과 표준 기반 모델 추론 플랫폼입니다.
- [FastAPI](https://github.com/tiangolo/fastapi): FastAPI는 Python으로 API를 구축하기 위한 최신 고성능 웹 프레임워크입니다.
- [TorchServe](https://github.com/pytorch/serve): TorchServe는 프로덕션 환경에서 PyTorch 모델을 제공하고 확장하기 위한 유연하고 사용하기 쉬운 도구입니다.
- [Ray](https://www.ray.io/): Ray는 컴퓨팅 집약적인 Python 워크로드를 간편하게 확장할 수 있는 오픈 소스 도구입니다.
- [Cog](https://github.com/replicate/cog): Cog는 머신러닝 모델을 표준 프로덕션 준비 컨테이너로 패키징할 수 있는 오픈 소스 도구입니다.
- [ModelDB](https://github.com/VertaAI/modeldb): ModelDB는 머신러닝 모델 버전 관리, 메타데이터 및 실험 관리를 위한 오픈 소스 시스템입니다.
- [MLflow](https://mlflow.org/): MLflow는 End-to-End 머신러닝 라이프싸이클을 관리하기 위한 오픈 소스 플랫폼입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.
- [Nuclio](https://github.com/nuclio/nuclio): Nuclio는 데이터, I/O 및 컴퓨팅 집약적 워크로드에 중점을 둔 고성능 서버리스 프레임워크 입니다.

### 모델 모니터링 (Model monitoring)

모델 모니터링 도구는 배포된 모델의 성능 저하, 편향 및 데이터 드리프트를 감지합니다.

- [Prometheus](https://prometheus.io/): Prometheus는 오픈 소스 시스템 모니터링 및 경고 툴킷입니다.
- [Grafana](https://grafana.com/): Grafana는 온라인 시스템 분석 및 모니터링을 위한 오픈 소스 도구입니다.
- [Evidently](https://github.com/evidentlyai/evidently): Evidently는 모델 수명 주기 전반에 걸쳐 데이터 및 ML 모델 품질을 분석 및 추적하기 위한 도구입니다.
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect): Alibi Detect는 이상값, 적대적 및 드리프트 감지에 중점을 둔 오픈 소스 Python 라이브러리 입니다.
- [ModelDB](https://github.com/VertaAI/modeldb): ModelDB는 머신러닝 모델 버전 관리, 메타데이터 및 실험 관리를 위한 오픈 소스 시스템입니다.
- [ClearML](https://github.com/allegroai/clearml): ClearML은 ML 워크플로우 간소화하는 자동 마법 스위트가 포함된 End-to-End 플랫폼입니다.

## 기타 MLOps 도구들

### 머신러닝/딥러닝 프레임워크 (ML/DL framework)

복잡한 머신러닝/딥러닝 모델 개발을 쉽고 효율적으로 할 수 있는 프레임워크들 입니다.

- [Tensorflow](https://www.tensorflow.org): TensorFlow는 머신러닝을 위한 End-to-End 오픈소스 플랫폼입니다.
- [PyTorch](https://pytorch.org): PyTorch는 GPU와 CPU를 이용한 딥러닝을 위한 최적화된 텐서 라이브러리입니다.
- [Keras](https://keras.io): Keras는 기계 학습 플랫폼 TensorFlow에서 실행되는 Python으로 작성된 딥 러닝 API입니다.
- [scikit-learn](https://scikit-learn.org): scikit-learn은 SciPy를 기반으로 구축된 머신러닝을 위한 Python 모듈입니다.
- [JAX](https://github.com/google/jax): JAX는 CPU, GPU 및 TPU의 NumPy이며 고성능 머신러닝 연구를 위한 자동 미분 기능을 제공합니다.

### 피처 스토어 (Feature store)

피처 스토어란 피처들을 위한 중앙 저장소 입니다. 피처 스토어는 미가공 데이터를 피처들로 변환하고 저장하며, 모델 학습과 추론에 제공합니다.

- [Feast](https://feast.dev): Feast는 오픈 소스 피처 스토어 입니다. 모델 학습 및 온라인 추론을 위한 분석 데이터를 운용하는 가장 빠른 방법입니다.
- [Hopsworks](https://www.hopsworks.ai): Hopsworks는 머신러닝 모델을 대규모로 개발 및 운영하는데 사용되는 데이터 집약적 오픈 소스 AI 플랫폼입니다.
- [Tecton](https://www.tecton.ai): Tecton은 엔터프라이즈급 SLA를 사용하여 변환에서 온라인 서비스에 이르기까지 기능의 전체 수명 주기를 오케스트레이션하도록 구축된 완전 관리형 피처 플랫폼입니다.
- [Rasgo](https://www.rasgoml.com): Rasgo는 클라우드 데이터 웨어하우스내에서 데이터를 탐색, 변환 및 시각화하기 위한 웹 앱입니다.

> 피처 엔지니어링이 필요하다면 피처 스토어 적용을 검토해보세요.

### 데이터 검증 (Data validation)

아웃라이어 (Outlier) 데이터와 유효하지 않은 데이터는 ML 모델에 악영향을 끼칩니다. 데이터 유효성 검사에 도움이 되는 도구들 입니다.

- [TFDV](https://www.tensorflow.org/tfx/data_validation): TensorFlow Data Validation은 학습 및 서빙 데이터의 이상을 식별하고 데이터를 검사하여 자동으로 스키마를 생성할 수 있습니다.
- [Deequ](https://github.com/awslabs/deequ): Deequ는 대규모 데이터 세트의 데이터 품질을 측정하는 "데이터 유닛 테스트"를 정의하기 위해 Apache Spark를 기반으로 구축된 라이브러리입니다.
- [DVT](https://github.com/GoogleCloudPlatform/professional-services-data-validator): Data Validation Tool은 이기종 데이터 원본 테이블을 다단계 유효성 검사 기능과 비교하는 [Ibis 프레임워크](https://ibis-project.org/) 기반의 오픈 소스 Python CLI 도구입니다.

### 설명 가능한 AI (XAI, eXplainable AI)

XAI는 AI 모델이 특정 결정을 내린 원인과 그 작동 원리를 사람들이 쉽게 파악할 수 있도록 도와주는 도구와 기술 세트입니다.

- [LIME](https://github.com/marcotcr/lime): Local Interpretable Model-agnostic Explanation의 약자로 XAI 알고리즘의 일종 입니다.
- [SHAP](https://christophm.github.io/interpretable-ml-book/shap.html): SHapley Additive Descriptions는 모든 기계 학습 모델의 출력을 설명하기 위한 게임 이론적 접근 방식입니다.
- [ELI5](https://eli5.readthedocs.io/en/latest/overview.html): ELI5는 머신러닝 분류기를 디버깅하고 추론 결과를 설명하는 데 도움이 되는 파이썬 패키지입니다.

> XAI는 활발히 연구가 진행되고 있고, 필요성에 대한 기사들은 많지만 실제 적용 사례는 아직 많지 않은것 같습니다.

## 결론

제품 등급 (Production-grade) ML 서비스에는 여기 나열된 MLOps 도구들 이외에도 많은 도구들이 같이 사용 될 것 입니다. 예를 들면, [JMeter](https://jmeter.apache.org)나 [Locust](https://locust.io)와 같은 툴로 부하 테스트를 해야 하며, [Logstash](https://www.elastic.co/kr/logstash)나 [Fluentd](https://www.fluentd.org/)로 로그 수집도 해야 합니다. MLOps가 포함하는 DevOps와 데이터 엔지니어링 영역까지 고려하면 훨씬 더 많은 도구들이 같이 사용 될것 입니다. (_kubernetes, kubespray, Terraform, Ansible, Helm, Kustomize, istio, jaeger, envoy, gRPC, RabbitMQ, Kafka, Keyclock, Valut, Argo CD, dbt, BigQuery, Athena, MongoDB, PostgresQL, Redis, ..._)

이렇게 많은 MLOps 도구들 중에 우리 ML 팀에 필요한 최적의 도구 들을 찾는건 쉽지 않은 일입니다. 아래는 제 개인의 경험에 의거한 MLOps 도구 및 솔루션 적용 가이드 입니다.

1. _작게 시작하세요._ ML 실험에 사용하는 데이터가 많지 않고, 아직 실험 단계에 머물러 있는 ML 팀이라면 사내 서버에 [MLflow](https://mlflow.org/), [Jupyter Hub](https://jupyter.org/hub)등의 도구들만 설정해서 사용하는 것도 많은 도움이 됩니다.
2. _통합 솔루션을 활용해 보세요._ 공용 클라우드를 사용하신다면 사용하시는 클라우드 서비스에서 제공하는 All-in-One ML 서비스 ([SageMaker](https://aws.amazon.com/ko/sagemaker/), [Vertex AI](https://cloud.google.com/vertex-ai), [AzureML](https://azure.microsoft.com/ko-kr/services/machine-learning/))를 먼저 활용해보세요. 이런 서비스들은 MLOps에 필요한 대부분의 기능을 제공합니다. [Neptune](https://neptune.ai/), [Weights & Biases](https://wandb.ai/), [Valohai](https://valohai.com/), [Algorhmia](https://algorithmia.com/), [cnvrg](https://cnvrg.io/)와 같이 MLOps 통합 솔루션을 제공하는 회사들도 많이 있습니다.
3. _필요한 툴을 찾아보세요._ 특정 클라우드 환경으로의 종속을 피하고 온프레미스 환경에 MLOps 인프라를 구축해야 한다면 위에 나열된 MLOps 도구 목록을 참조하여 도움이 될만한 도구를 찾아보세요. MLOps 도구는 현재 가장 빠르게 트렌드가 변화하고 있는 분야중 하나이므로 적용을 검토하는 시점에서 인터넷 검색은 필수 입니다.
4. _배보다 배꼽이 커지지 않게 주의하세요._ MLOps 툴을 사용하는 하는 것은 ML 워크플로우를 자동화 하거나 최대한 편하게 만드는 것이 목적입니다. 무분별한 MLOps 툴 도입은 학습 리소스 낭비와 ML 시스템 복잡도만 높인다는 것을 잊지 마세요. 시험 운영 기간 동안 기존 대비 큰 개선점이 없다면 안쓰는 것이 정답입니다.
5. _메인 툴을 정하세요._ [하이브리드 클라우드](https://www.vmware.com/kr/topics/glossary/content/hybrid-cloud.html) 환경에는 온프레미스 환경에 설치된 MLOps 툴과 공용 클라우드의 ML 서비스간에 겹치는 기능이 많습니다. 더구나 요즘 MLOps 툴들은 통합 솔루션을 지향하고 있어 위에 나열된 도구들 중에도 중복된 기능을 가진 것이 많습니다. 비용, 확장성, 고가용성, 관리용이성, 개발편의성 등을 고려하여 중복되는 기능은 어떤 도구를 메인으로 사용할지 결정해야 합니다.

[![ML Lanscape](https://about.mlreef.com/images/blog/ml_landscape.png)](https://about.mlreef.com/blog/global-mlops-and-ml-tools-landscape)
_<center>선택 장애를 일으키는 수많은 MLOps 도구들...AI/ML 분야의 또다른 겨울이 오지 않는 이상 줄어들것 같지는 않습니다.</center>_

## 더 읽을거리

- [MLOps 소개](https://taehun.github.io/mlops/2022/04/19/introduction-mlops.html): MLOps 컴포넌트와 핵심 요소들에 대응되는 툴은?
- [MLOPS Tools & Platforms Landscape: In-Depth Guide for 2022](https://research.aimultiple.com/mlops-tools/): 비슷한 내용의 영문 기사 입니다.
- [Build Your MLOps Tool Stack](https://neptune.ai/mlops-tool-stack): [Neptune](https://neptune.ai/)의 MLOps 툴과 관련된 기사 모음집 입니다. [Neptune](https://neptune.ai/)에서 작성한 글이므로 모든 항목에서 추천 도구로 Neptune이 포함되는 것만 감안해서 참조하세요.
