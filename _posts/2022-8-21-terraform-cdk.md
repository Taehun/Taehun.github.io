---
layout: post
title: Python으로 인프라 생성하기 - CDKTF
categories: devops
tags: [cloud, infra, devops]
toc: true
comments: true
excerpt_separator: <!--more-->
---

DevOps 성숙도가 높은 기업이나 팀이라면 _IaC (Infrastructure as Code)_ 를 도입하여 인프라를 코드로 관리 하고 있을 것 입니다. 인프라 배포 자동화, 인프라 일관성 유지, 인프라 가시성 확보, 온디멘드 인프라등의 IaC의 많은 장점 들로 인해 이제 IaC는 새로운 기술이 아닌 인프라 관리의 기본이 되어 가고 있습니다. 대표적인 IaC 툴로는 프로비저닝은 [Terraform](https://www.terraform.io/), 설정은 [Ansible](https://www.ansible.com/)을 많이 사용하고 있습니다. 기타 클라우드 벤더 종속적인 [AWS CloudFormation](https://aws.amazon.com/ko/cloudformation/), [AWS CDK](https://aws.amazon.com/ko/cdk/), [ARM (Azure Resource Manager)](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/overview)등의 다양한 IaC 툴들이 있습니다.

그 중 대표적인 IaC 툴인 Terraform은 많은 자료들이 있지만, _HCL (HashiCorp Configuration Language)_ 이라는 전용 언어를 익혀야 하는 약간의 진입 장벽이 있습니다. 직관적이고 가독성이 높은 쉬운 언어이지만, 새로운 언어를 배운다는 것은 언제나 심리적 허들이 생기기 마련입니다. HCL을 사용하지 않고, 익숙한 Python과 같은 언어로 인프라 코드를 작성할 수 있는 **CDKTF (CDK for Terraform)** 를 소개해 드리겠습니다.

<!--more-->


<img src="https://github.com/Taehun/taehun.github.io/blob/main/imgs/cdktf.png?raw=true" height="640" width="800"/>
* CDKTF 개요 (_출처> [https://www.terraform.io/cdktf](https://www.terraform.io/cdktf)_ )

## CDKTF CLI 설치

CDKTF를 사용하시려면 `cdktf` 라는 커맨드라인 툴이 필요 합니다. `cdktf` 커맨드라인 툴 설치 방법은 다음과 같습니다. `terraform` CLI가 사전 설치 되어 있어야 합니다. (참고> [Install Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli))

* brew로 설치

```bash
$ brew install cdktf
```

* npm으로 설치

```bash
$ npm install --global cdktf-cli@latest
```

설치 확인
```bash
$ cdktf help
cdktf

명령:
  cdktf init                Create a new cdktf project from a template.
  cdktf get                 Generate CDK Constructs for Terraform providers and modules.
  cdktf convert             Converts a single file of HCL configuration to CDK for Terraform. Takes the file to be converted on stdin.
  cdktf deploy [stacks...]  Deploy the given stacks                                                                            [별칭: apply]
  cdktf destroy [stacks..]  Destroy the given stacks
  cdktf diff [stack]        Perform a diff (terraform plan) for the given stack                                                 [별칭: plan]
  cdktf list                List stacks in app.
  cdktf login               Retrieves an API token to connect to Terraform Cloud or Terraform Enterprise.
  cdktf synth               Synthesizes Terraform code for the given app in a directory.                                  [별칭: synthesize]
  cdktf watch [stacks..]    [experimental] Watch for file changes and automatically trigger a deploy
  cdktf output [stacks..]   Prints the output of stacks                                                                      [별칭: outputs]
  cdktf debug               Get debug information about the current project and environment
  cdktf provider            A set of subcommands that facilitates provider management
  cdktf completion          generate completion script

옵션:
      --version                   버전 표시                                                                                         [불리언]
      --disable-plugin-cache-env  Dont set TF_PLUGIN_CACHE_DIR automatically. This is useful when the plugin cache is configured
                                  differently. Supported using the env CDKTF_DISABLE_PLUGIN_CACHE_ENV.              [불리언] [기본값: false]
      --log-level                 Which log level should be written. Only supported via setting the env CDKTF_LOG_LEVEL             [문자열]
  -h, --help                      도움말 표시                                                                                       [불리언]

Options can be specified via environment variables with the "CDKTF_" prefix (e.g. "CDKTF_OUTPUT")
```

## CDKTF 프로젝트 초기화

CDKTF 프로젝트 폴더를 생성하고 초기화 합니다. 

```bash
$ mkdir cdktf-example
$ cd cdktf-example
$ cdktf init --template=python --local
```

Python 코드로 인프라를 정의할 것이므로 `--template` 옵션은 `python`으로 설정 하였습니다. `--local` 옵션은 `.tfstate` 파일을 로컬에서 관리한다는 설정이므로 실행시 경고 메세지를 띄웁니다. 실제 프로젝트 환경에는 `--local` 옵션을 생략하여 [Terraform Cloud](https://cloud.hashicorp.com/products/terraform)과 같은 곳에서 `.tfstate` 파일을 중앙 집중식으로 관리하시기 바랍니다.

## Google provider 설치

Terraform은 플러그인 형태로 사용할 수 있는 몇가지 사전 빌드된 provider를 제공합니다. AWS provider 예제는 많이 있으니, 여기서는 GCP (Google Cloud Platform) 인프라 생성 예제를 위해 Google provider를 추가 하겠습니다. Terraform으로 GCP 인프라 생성 및 삭제를 위해서는 필요한 API 활성화 및 서비스 계정 `.json` 키 파일이 필요 합니다. [Terraform 시작하기](https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=ko) 문서와 [서비스 계정 생성 및 관리](https://cloud.google.com/iam/docs/creating-managing-service-accounts?hl=ko) 문서를 참고하시기 바랍니다.

```bash
$ cdktf provider add "google@~>4.0"
```

Provider를 설치하면 `imports` 라는 폴더에 설치한 provider가 생성 됩니다. 

```bash
$ tree imports
imports
└── google
    ├── __init__.py
    ├── _jsii
    │   ├── __init__.py
    │   └── hashicorp_google@0.0.0.jsii.tgz
    └── py.typed
```

## 인프라 코드 작성

`main.py` 파일을 열어 아래와 같이 코드를 수정 합니다. 이 예제는 `f1-micro` 머신 유형의 VM을 하나 생성하는 예제 입니다.

```python
#!/usr/bin/env python
from cdktf import App, TerraformOutput, TerraformStack
from cdktf_cdktf_provider_google import ComputeInstance, ComputeNetwork, GoogleProvider
from constructs import Construct


class MyStack(TerraformStack):  # Provider와 생성할 모든 리소스를 정의하는 코드가 포함된 새 스택
    def __init__(self, scope: Construct, ns: str):
        super().__init__(scope, ns)

        # 서비스 계정 키 파일을 로드 합니다.
        with open("credentials.json", "r") as fp:
            credentials = fp.read()

        # Google provider를 초기화 합니다.
        GoogleProvider(
            self,
            "Google",
            region="asia-northeast3",  # 사용할 region (서울)
            zone="asia-northeast3-a",  # 사용할 zone
            project="YOUR_GCP_PROJECT_ID",  # GCP 프로젝트 ID
            credentials=credentials,  # 서비스 계정 키
        )

        # 생성할 VPC 네트워크를 정의합니다.
        network = ComputeNetwork(self, "Network", name="cdktf-network")

        # 생성할 VM 인스턴스를 정의합니다.
        instance = ComputeInstance(
            self,
            "ComputeInstance",
            name="sample-vm",
            machine_type="f1-micro",  # 머신 유형
            boot_disk={
                "initialize_params": {
                    "image": "debian-11-bullseye-v20220719",  # 운영체제 이미지
                },
            },
            network_interface=[{"network": network.name}],  # 새로 생성한 "cdktf-network"
            tags=["web", "dev"], 
            depends_on=[network],
        )

        # 실행 완료후 출력할 메세지를 설정합니다. 단순히 VM의 인스턴스 이름 ("sample-vm")을 출력 합니다.
        TerraformOutput(
            self,
            "instance_name",
            value=instance.name,
        )


# CDKTF 어플리케이션을 생성하고, app.synth()를 호출하여 Terraform 구성을 생성합니다.
app = App()
stack = MyStack(app, "gcp_instance")
app.synth()
```

## 인프라 프로비저닝

안프라 코드 작성이 완료 되었으므로, 이제 인프라 코드를 배포하여 실제로 인프라를 생성할 것 입니다. `cdktf deploy` 커맨드로 인프라 코드를 배포하여 정의된 인프라를 생성 할 수 있습니다. 커맨드를 실행하면 생성하거나 삭제될 인프라에 대한 정보가 출력되고, _Approve_ , _Dismiss_ , _Stop_ 세 가지 메뉴에서 하나를 선택해야 합니다. _Approve_ 를 선택하면 인프라 코드의 변경점이 실제 인프라에 적용 됩니다. 

```bash
$ cdftf deploy

(......)

Please review the diff output above for gcp_instance
❯ Approve  Applies the changes outlined in the plan.
  Dismiss
  Stop
```

[GCP 콘솔](https://console.cloud.google.com)에 접속하여 `Compute Engine` -> `VM 인스턴스` 메뉴를 클릭하면 아래와 같이 새로운 VM이 생성되어 있는 것을 확인 하실 수 있습니다.

![CDFTF-VM](https://github.com/Taehun/taehun.github.io/blob/main/imgs/cdktf-vm.png?raw=true)

## 인프라 정리하기

CDKTF로 생성한 인프라는 `cdktf destroy` 커맨드로 삭제 할 수 있습니다. 커맨드 실행후 _Approve_ 메뉴를 선택하면, CDKTF로 생성된 인프라를 삭제합니다.

```bash
$ cdktf destroy

(......)

Please review the diff output above for aws_instance
❯ Approve  Applies the changes outlined in the plan.
  Dismiss
  Stop

(......)

            Destroy complete! Resources: 2 destroyed.
```

## 결론

Terraform 사용 경험이 있는 분에게는 CDKTF를 아직 실 프로젝트에 적용하기에는 부족함을 많이 느끼실 것 입니다. CDKTF는 Terraform HCL을 대체하기 보다는 익숙한 프로그래밍 언어로 인프라 코드를 작성할 수 있도록 도와주는 일종의 래퍼 모듈 입니다. 아마 HCL 코드가 복잡하고 배우기 어려웠으면 CDKTF 사용자가 더 많아지고 프로젝트도 좀 더 활성화 되었을지도 모르겠네요. CDKTF는 아직 Terraform과 HCL에 대해 문외한이고 익숙한 Python 코드로 인프라를 생성해보고 싶으신 분들께 추천합니다.

그럼, 이미 Terraform HCL을 잘쓰고 있는 팀이나 엔지니어에게는 전혀 쓸모 없는 것인가?는 좀 더 고민해 봐야 할 부분입니다. [4 Use Cases for the Terraform CDK](https://medium.com/codex/4-use-cases-for-the-terraform-cdk-5864630d147e)라는 기사에서 아래와 같은 CDKTF의 4가지 사용 예를 정리하였습니다:

- 동적 자원 속성 (Dynamic Resource Attributes)
- 동적 모듈 구성 (Dynamic Module Composition)
- 외부 구성 통합 (External Configuration Integration)
- 프로비젼 후 단계 (Post Provision Steps)

좀 더 완성도가 높아지고, 빅테크 기업에서 CDKTF 적용 예들이 하나 둘 씩 생기기 시작하면 인기 있는 툴이 될 것 같네요. (~~그러기엔 HCL이 너무 쉬운걸?~~)

> 이 글은 [Terraform CDKTF 공식 문서](https://learn.hashicorp.com/collections/terraform/cdktf)와 [CDKTF 예제](https://github.com/hashicorp/terraform-cdk/tree/main/examples) 코드를 참고 하여 작성하였습니다.

