---
layout: post
title: "MLOps 소개"
categories: mlops
---

## MLOps 정의

**MLOps (Machine Learning + Operation)** 를 한마디로 정의하면 *머신러닝 어플리케이션에 DevOps 원칙을 적용 한 것*이라고 할 수 있습니다. (by [MLOps Sig](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2020/MLOpsRoadmap2020.md)) MLOps에서는 머신러닝 모델을 학습하고 추론하기 위한 워크플로우 및 시스템을 개발하고 운영합니다. 다른 의미로 MLOps는 *프로덕션 환경에서 머신러닝 모델을 안정적이고 효율적으로 배포 및 유지 관리하는 것을 목표로 한 일종의 관행*으로 볼 수 있습니다. (by [Breuel](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f))

DevOps가 개발/QA/운영의 교차점이라면, MLOps는 머신러닝/데이터 엔지니어링/DevOps의 교차점 입니다. 이러한 관점에서 MLOps는 *머신러닝, DevOps 및 데이터 엔지니어링을 결합한 일련의 방식으로 프로덕션 환경에서 ML 시스템을 안정적이고 효율적으로 배포 및 유지 관리하는 것*이라고 할 수 있습니다.

![DevOpsMLOps](https://github.com/Taehun/taehun.github.io/blob/master/imgs/devops_mlops.png?raw=true)

- 그림 1. DevOps와 MLOps

사전적 정의를 떠나서 MLOps를 실제 프로젝트에 적용해야하는 팀이나 MLOps 엔지니어와 같은 담당자 관점에서 MLOps는 **_End-to-End 머신러닝 워크플로우를 자동화 하는 것_** 이라고 할 수 있습니다. 데이터 라벨링 작업과 같은 [HITL (Human-in-the-loop)](https://en.wikipedia.org/wiki/Human-in-the-loop)로 사람이 반드시 관여해야하는 워크플로우는 모든 과정을 100% 자동화를 하는 것은 불가능 합니다. 이러한 HITL 프로세스도 _능동적 학습 (Active learning), 자동 라벨링 (Auto labeling), 준지도 학습 (Semi-supervised learning)_ 등의 기법들을 적용하여 할수 있는한 최대한 자동화를 해야 합니다.

## End-to-End 머신러닝 워크플로우

![ML Workflow](https://github.com/Taehun/taehun.github.io/blob/master/imgs/ml-workflow.png?raw=true)

- 그림 2. End-to-End 머신러닝 워크플로우

<그림 2>는 머신러닝 프로젝트의 End-to-End 워크플로우를 도식화한 것 입니다:

1. **데이터 추출**&nbsp;&nbsp;
   데이터 레이크와 같은 데이터 소스에서 관련된 데이터를 추출 합니다.
2. **데이터 탐색**&nbsp;&nbsp;
   데이터에 대한 이해를 위해 데이터 분석(EDA)을 수행합니다.
3. **데이터 가공**&nbsp;&nbsp;
   모델 학습에 필요한 데이터셋을 만들기 위해 특정 스키마로 데이터를 가공 합니다.
4. **데이터 검증**&nbsp;&nbsp;
   가공된 데이터에 유효하지 않는 데이터가 포함되어 있진 않은지 검증 합니다.
5. **모델 개발**&nbsp;&nbsp;
   머신 러닝 모델을 준비 합니다. 실제 모델 개발은 1~4 과정과 함께 이루어 집니다.
6. **모델 학습**&nbsp;&nbsp;
   준비된 데이터셋으로 모델을 학습 합니다.
7. **모델 평가**&nbsp;&nbsp;
   테스트 데이터셋에서 학습된 모델을 평가합니다.
8. **모델 배포**&nbsp;&nbsp;
   학습된 모델을 배포 포맷으로 변환하여 모델 레지스트리에 배포 합니다.
9. **모델 서빙**&nbsp;&nbsp;
   모델 레지스트리에 있는 모델을 로드하여 모델 추론 REST API 서비스를 배포 합니다.
10. **A/B 테스트**&nbsp;&nbsp;
    새로 배포된 모델 추론 서비스가 실제로 개선이 되었는지 판단하기 위해 A/B 테스트를 수행합니다.
11. **로깅**&nbsp;&nbsp;
    배포된 추론 서비스의 요청 및 응답 과정을 로깅 합니다.
12. **모니터링**&nbsp;&nbsp;
    로그를 바탕으로 모델의 추론 성능을 모니터링 합니다.

이 워크플로우는 머신러닝 모델을 사용하는 웹 앱이나 모바일 앱의 백엔드에 배포하여 사용하는 예시 입니다. (장치에 직접 모델을 배포하여 추론하는 *온 디바이스(On-Device) AI*는 과정은 비슷하지만 각 단계마다 수행하는 방식이 상이합니다. 다음에 기회가 있으면 온 디바이스 AI MLOps에 대해 다루어 보도록 하겠습니다.)

<그림 1>의 MLOps 벤 다이어그램에서 MLOps는 머신러닝, 데이터 엔지니어링, DevOps의 교차점이라고 하였습니다. <그림 2>의 머신러닝 워크플로우에서 초록색은 _데이터 엔지니어링_, 노란색은 _머신러닝_, 파란색은 *DevOps*에서 수행하는 단계들 입니다. *MLOps 엔지니어*가 End-to-End 머신러닝 워크플로우를 혼자서 모두 다 할 수도 있겠지만, 가장 현실성 있는 시나리오는 데이터 엔지니어링은 _데이터 엔지니어_, 머신러닝은 _데이터 과확자_, DevOps는 *DevOps 엔지니어*로 구성된 팀으로서 머신러닝 워크플로우를 수행하는 것 입니다.

앞서 MLOps를 실제 머신러닝 프로젝트에 적용하는 것은 End-to-End 머신러닝 워크플로우를 자동화 하는 것이라고 하였습니다. 각 분야별 담당자로 구성된 머신러닝 팀이라면 이런 목적을 달성하기 위해 모든 구성원이 MLOps를 이해하고 하나의 자동화된 End-to-End 머신러닝 시스템을 만들어 나가야 합니다. 머신러닝 워크플로우를 자동화하는 대표적인 구현체가 머신러닝 파이프라인(이하 ML 파이프라인) 입니다.

<img src="https://v0-6.kubeflow.org/docs/images/pipelines-xgboost-graph.png" alt="drawing" width="600"/>

- 그림 3. Kubeflow Pipeline

MLOps에는 ML 파이프라인을 포함하여 머신러닝 모델을 제품화하는데 필요한 다양한 구성 요소들이 있습니다.

## MLOps 구성 요소 및 핵심 요소

구글의 [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) 문서에서 정의한 MLOps 구성 요소들 입니다.

- 실험(Experimentation)
- 데이터 처리(Data processing)
- 모델 학습(Model training)
- 모델 평가(Model evaluation)
- 모델 서빙(Model serving)
- 온라인 실험(Online experimentation)
- 모델 모니터링(Model monitoring)
- ML 파이프라인(ML pipelines)
- 모델 레지스트리(Model registry)
- 데이터셋과 피처 스토어(Dataset and feature repository)
- ML 메타데이터와 아티팩트 추적(ML metadata and artifact tracking)

위 구성 요소에 대한 내용은 이미 다른곳에서 많이 요약되어 있어서 생략하겠습니다. 제가 지금까지 MLOps를 하면서 느꼈던 제 기준에서 MLOps의 핵심 요소들은 다음과 같습니다.

- 데이터 관리
- 실험 관리
- 인프라 관리
- 모델 관리
- ML 파이프라인

### 데이터 관리

> _“Data is food for AI" by Andrew Ng_

MLOps는 작년(2021년)부터 업계 종사자들에게 널리 알려지기 시작 했습니다. MLOps 대중화의 시발점이 된 것은 AI 업계의 대부이신 Andrew Ng 교수님의 ["MLOps: From Model-centric to Data-centric AI"](https://www.youtube.com/watch?v=06-AZXmwHjo&t=1s) 온라인 세미나 였습니다. 세미나 제목에서 유추 할 수 있듯이 머신러닝 프로젝트에서 데이터의 중요성을 강조하는 세미나 입니다.

![Code+Data](https://github.com/Taehun/taehun.github.io/blob/master/imgs/software-vs-mlmodel.png?raw=true)

- 그림 4. 소프트웨어와 머신러닝 모델에서 코드와 데이터

머신러닝 모델은 코드 뿐만아니라 데이터로부터 만들어 집니다. 즉, Data-centric AI 세미나의 내용처럼 데이터의 품질이 머신러닝 모델의 품질을 좌지우지 합니다. 머신러닝 팀은 모델 학습에 사용되는 데이터의 품질을 최상으로 유지하도록 노력해야 합니다.

데이터의 품질 관리과 더불어 머신러닝 실험 재현성을 위해 데이터셋 버전 관리도 필요합니다. 과거 모델 학습에 사용된 데이터를 현재에도 그대로 불러와서 사용하여 동일하게 재현 할 수 있어야 합니다. [DVC](https://dvc.org/)와 같은 툴로 데이터셋 버전을 관리를 하거나 원본 데이터가 읽기 전용이라면 학습에 사용된 데이터의 기간을 실험 정보에 추가하는 방법으로 데이터셋 버전을 관리 할 수 있습니다. 고정된 데이터셋으로 하는 머신러닝 실험과 달리 프로덕션 환경에서 데이터는 지속적으로 변경되고 추가된다는 점을 잊지 마세요. (오프라인 데이터 Vs. 온라인 데이터)

> 아무리 강조해도 지나치지 않습니다.  
> **Data, Data, Data, ...**

### 실험 관리

머신러닝 모델 재현성을 위해서는 앞서 언급한 데이터뿐만 아니라 하이퍼 파라메터, 네트워크, 체크포인트등의 메타 정보들이 필요합니다. 모델 학습 과정을 하나의 실험으로 관리하고, 학습된 모델이 어떤 실험으로 부터 생성되었는지 관리하는 매커니즘이 있어야 합니다.

머신러닝 모델 개발에도 실험 관리는 중요 합니다. 원하는 머신러닝 모델을 만들기 위해서는 이전 실험이 어떻게 실행되었는지 추적이 되어야 다른 알고리즘을 사용해보거나, 하이퍼 파라메터 값을 조절하여 다음 실험에 적용 할 수 있습니다. AutoML을 활용해 모델 개발의 많은 부분을 자동화 하더라도, 머신러닝 실험 관리는 MLOps에 반드시 필요합니다.

### 인프라 관리

많은 연산을 요구하는 딥러닝 모델을 학습하거나 서빙할때는 GPU나 NPU와 같은 딥러닝 연산에 특화된 별도의 컴퓨팅 리소스가 필요합니다. 최근 대부분의 백엔드 서비스에 사용하는 쿠버네티스 환경에서 딥러닝 전용 리소스를 추가하고 관리하는 것은 많은 어려움이 따릅니다. 기본적으로 쿠버네티스에서 GPU 리소스는 CPU나 메모리처럼 잘게 쪼개서 사용할 수 없으므로 리소스 낭비가 발생하게 됩니다. 최근에는 이런 문제점을 해결하고자 NVIDIA의 [MIG (Multi-Instance GPU)](https://www.nvidia.com/ko-kr/technologies/multi-instance-gpu/)나 소프트웨어 레벨에서 논리적 GPU로 분할하려는 시도들이 있습니다.

프로덕션 환경에서 머신러닝 모델 학습에 사용되는 데이터는 일반적으로 빅 데이터 입니다. 모델 학습시에는 효율적인 빅 데이터 처리를 위해 높은 처리량을 지원해야 합니다. 마이크로서비스 형태로 모델 서빙시에는 기존의 마이크로서비스와 동일하게 고가용성과 확장성을 보장해야 합니다. 효율적인 저장공간 관리, 컴퓨팅 리소스 관리, 모델 서빙 마이크로서비스의 고가용성 및 확장성 지원 등의 고수준의 DevOps 스킬이 필요합니다.

### 모델 관리

모델 관리는 중앙집중화된 모델 레지스트리에 모델을 배포하여 모델 버전 관리를 하는 것을 의미 합니다. 모델 배포시에는 재현성을 위해 어떤 실험에 의해 (혹은 어떤 ML 파이프라인 실행에 의해) 모델이 생성되고 배포 되었는지 기록되어야 합니다. 여러 버전의 모델 중에 현재 서비스 중인 모델의 버전이 어떤 것인지 알 수 있어야 합니다.

딥러닝 모델은 학습에 사용되는 체크포인트와 추론 서비스에 사용되는 서빙 모델로 구분됩니다. 서빙 모델은 학습에 사용하지 못하므로 재학습을 위해 학습된 모델의 체크포인트도 같이 관리되어야 합니다.

> **서빙 모델**  
> 딥러닝 모델은 학습이 완료되면 체크포인트 파일이 생성됩니다. 체크포인트 파일을 그대로 배포하기에는 느리고 무겁습니다. 양자화, 프루닝, 클러스터링등의 기법을 적용하여 모델 최적화를 한 뒤, 서빙용으로 모델 포맷을 변환 합니다. 서빙 모델은 추론시 GPU, NPU, DSP, TPU 등의 전용 하드웨어에서 하드웨어 레벨 가속화를 지원 합니다. 대표적인 서빙 모델로는 ONNX, TFLite, TensorRT 등이 있습니다.

### ML 파이프라인

<img src="https://cloud.google.com/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg" alt="drawing" width="600"/>

- 그림 5. ML 파이프라인 CI/CD (출처> [MLOps: 머신러닝의 지속적 배포 및 자동화 파이프라인 ](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning))

구글의 [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) 문서와 Valohai의 [Practical MLOps](https://valohai.com/assets/files/practical-mlops-ebook.pdf) 문서에는 배포 단위를 모델이 아닌 ML 파이프라인 되어야 한다고 주장하고 있습니다.

ML 파이프라인 단위 배포의 주요 장점은 온라인 데이터에 대한 대응이 자동화 되어 있다는 점 입니다. 프로덕션 데이터는 시시각각 변화하므로 배포된 모델의 [데이터 드리프트](https://docs.microsoft.com/ko-kr/azure/machine-learning/how-to-monitor-datasets?tabs=python#what-is-data-drift) 발생시 재학습이 필요합니다. MLOps의 주요 기능 중 하나인 CT (Continuous Training, 지속적 학습)가 이에 해당 됩니다.

모든 머신러닝 모델이 재학습이 필요한 것은 아닙니다. 재학습이 필요 없는 머신러닝 모델도 분명 존재할 것 입니다. 데이터 특성상 데이터 드리프트가 거의 발생하지 않아 매우 드물게 재학습을 하는 모델도 있을 것 입니다. 이런 모델들은 자동화된 지속적 학습이 아닌 수동으로 다시 학습하는 것만으로 충분할지도 모릅니다. 마찬가지로 파이프라인 단위 배포가 아닌 모델 단위 배포로도 충분할 것 입니다. 이런 모델들은 ML 파이프라인이 필요가 없을까요? 아니요. 굳이 지속적 학습을 하지 않더라도 ML 파이프라인은 있어야 합니다. MLOps는 _End-to-End 머신러닝 워크플로우를 자동화_ 하는 것이기 때문 입니다. 데이터가 변하지 않더라도 AI 분야 기술이 발전되면서 새로운 머신러닝 알고리즘을 프로덕트에 적용하는 것도 ML 파이프라인이 있으면 편리합니다.

## MLOps Vs. DataOps Vs. AIOps

요즘 MLOps와 더불어 많은 xOps들이 난립하고 있습니다. Ops 앞의 접두사가 달라지더라도 대부분의 xOps의 기본 원칙은 동일합니다. "x를 효율적으로 하기 위해 DevOps 원칙을 적용한 것" 입니다. DataOps는 데이터 관리와 처리에 DevOps 원칙을 적용하여 효율적으로 한 것 입니다. 많은 회사에서 데이터 엔지니어가 DataOps를 담당하고 있습니다. MLOps 정의에서 알려드린 것처럼 MLOps는 DataOps를 포함합니다.

MLOps와 가장 혼동되는 것이 AIOps 입니다. AIOps와 MLOps는 단어상으로는 유사하지만 의미하는바는 완전히 다릅니다. AIOps는 IT 운영의 자동화 및 관리를 위해 운영 데이터에 AI 기술을 적용하는 것을 의미합니다. 로그 분석, 애플리케이션 모니터링, 서비스 데스크, 인시던트 대응 등 IT 운영을 AI로 확장한 것을 AIOps라고 볼 수 있습니다. '머신러닝'이 'AI'의 일부라고 해도 MLOps와 AIOps는 전혀 다른 것을 의미하니 유의하시기 바랍니다.

## 요약

- MLOps란 머신러닝 어플리케이션에 DevOps 원칙을 적용 한 것이다. (개념적 정의)
- MLOps란 End-to-End 머신러닝 워크플로우를 자동화 하는 것이다. (실무적 정의)
- MLOps는 머신러닝, 데이터 엔지니어링, DevOps의 상호 작용이다.
- End-to-End 머신러닝 워크플로우: 데이터 추출 -> 데이터 탐색 -> 데이터 가공 -> 데이터 검증 -> 모델 개발 -> 모델 학습 -> 모델 평가 -> 모델 배포 -> 모델 서빙 -> A/B 테스트 -> 로깅 -> 모니터링
- MLOps 구성 요소: 실험, 데이터 처리, 모델 학습, 모델 평가, 모델 서빙, 온라인 실험, 모델 모니터링, ML 파이프라인, 모델 레지스트리, 데이터셋과 피처 스토어, ML 메타데이터와 아티팩트 추적
- MLOps 핵심 요소: 데이터 관리, 실험 관리, 인프라 관리, 모델 관리, ML 파이프라인
- MLOps = DevOps + DataOps + ML, MLOps != AIOps
