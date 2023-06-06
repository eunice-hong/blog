---
layout: post
title: "미세 조정: Open AI 문서 한국어로 읽기"
date:   2023-06-06 16:16:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- open-ai

category: book
author: eunice-hong
description: Open AI 문서 미세 조정(Fine-tuning)을 한국어로 읽어봅니다. 오역, 의역이 있을 수 있습니다.

---

> 내 입맛에 맞게 모델을 변경하는 방법에 대해 알아봅니다.

### 들어가기에 앞서

미세 조정 기능을 사용하면 API를 통해 제공되는 모델을 보다 효과적으로 활용할 수 있습니다:

1. 프롬프트 설계를 하는 것보다 결과물의 **품질**이 높다.
2. 프롬프트에 맞추기 보다 **더 많은 예제를 훈련**시킬 수 있다.
3. 프롬프트 길이를 줄여, **토큰을 절감**할 수 있다.
4. 요청 **대기 시간을 단축**시킬 수 있다.

GPT-3는 개방형 인터넷에서 방대한 양의 텍스트에 대해 사전 훈련을 받았습니다.
몇 가지 예시가 포함된 프롬프트를 제공하면 수행 내용을 직관적으로 파악하고 그럴듯한 결과물을 생성할 수 있습니다.
이것을 "few-shot learning(이하 퓨샷 학습)"이라고 합니다.

미세 조정은 보다 많은 예제를 훈련하여 결과물의 품질을 높입니다.
프롬프트 작성에 용을 쓰는 것 보다 훨씬 더 많은 예제를 훈련할 수 있습니다.
모델을 미세 조정한 후에는 더 이상 프롬프트에 예제를 추가할 필요가 없습니다.
이를 통해 비용을 절감하고 대기 시간을 단축할 수 있습니다.

최종 단계에서, 미세 조정는 아래와 같은 순서로 이뤄집니다:

1. 훈련 데이터 준비 및 업로드
2. 미세 조정된 새 모델 훈련
3. 미세 조정된 모델 사용

미세 조정된 모델 훈련 및 사용 비용이 청구되는 방법에 대해 자세히 알아보려면 OpenAI의 [가격 페이지][open_ai_pricing]를 방문하십시오.

### 미세 조정 가능한 모델이란?

현재 `davinci`, `curie`, `babbage` 및 `ada`의 기본 모델만 미세 조정 기능을 사용할 수 있습니다.
이러한 모델은 훈련 후 별도의 지시사항이 없는 오리지널 모델입니다(예: text-davinci-003).
또한 처음부터 시작할 필요 없이 미세 조정된 모델을 계속해서 미세 조정하여 추가 데이터를 추가할 수도 있습니다.

### 설치하기

OpenAI CLI(명령줄 인터페이스)를 사용하는 것이 좋습니다. 설치하려면 다음을 실행합니다.

```bash
pip install --upgrade openai
```

다음 지침은 버전 0.9.4 이상에서 사용할 수 있습니다. 또한 OpenAI CLI에는 python 3이 필요합니다.)

셸 초기화 스크립트(예: .bashrc, zshrc 등)에 다음 행을 추가하거나 미세 조정 명령 전에 명령줄에서 실행하여 OPENAI_API_KEY 환경 변수를 설정합니다:

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

### 훈련 데이터 준비 및 업로드

> 💡"훈련 데이터"란 GPT-3에게 어떤 것을 말해야하는 지 알려주는 교재입니다.

데이터는 [JSONL][jsonl_official] 문서여야 합니다.
여기서 각 줄은 훈련 예제에 해당하는 프롬프트-결과 쌍으로 이뤄져 있습니다.
[CLI 데이터 준비 도구](#cli-데이터-준비-도구)를 사용하여 데이터를 이 파일 형식으로 쉽게 변환할 수 있습니다.

```jsonlines
{
  "prompt": "<프롬프트>",
  "completion": "<이상적으로 생성된 텍스트 예시>"
}
{
  "prompt": "<프롬프트>",
  "completion": "<이상적으로 생성된 텍스트 예시>"
}
{
  "prompt": "<프롬프트>",
  "completion": "<이상적으로 생성된 텍스트 예시>"
}
...
```

미세 조정을 위한 프롬프트-결과 쌍을 설계하는 것은
기본 모델(davinci, curie, babbage, ada)에서 사용할 프롬프트를 설계하는 것과 다릅니다.
특히, 기본 모델에 대한 프롬프트는 종종 미세 조정을 위해 여러 예제로 구성(퓨샷 학습)되지만,
미세 조정 훈련 예제는 일반적으로 단일 입력 예제와 해당 입력과 관련된 출력으로 구성되며,
자세한 지침을 제공하거나 동일한 프롬프트에 여러 예제를 포함할 필요가 없습니다.

다양한 작업에 대한 훈련 데이터를 준비하는 방법에 대한 자세한 지침은 [데이터 준비 모범 사례](#일반-모범-사례)를 참조하세요.

예시가 많을수록 좋습니다.
적어도 200개의 예를 갖추는 것을 추천합니다.
연구 결과, 일반적으로 데이터 세트 크기를 두 배로 늘릴 때마다 모델 품질이 선형적으로 증가합니다.

#### CLI 데이터 준비 도구

OpenAI는 데이터를 검증, 제안하며 포맷을 변경할 수 있는 툴을 개발했습니다.

```bash
openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

이 도구는 프롬프트와 완료 열/키를 포함해야 하는 다른 형식을 사용할 수 있습니다.
**CSV**, **TSV**, **XLSX**, **JSON** 또는 **JSONL** 파일을 전달하면 제안된 변경 프로세스를 안내한 후
출력을 미세 조정할 수 있는 JSONL 파일에 저장합니다.

### 미세 조정 모델 생성하기

다음은 [위의 지침](#훈련-데이터-준비-및-업로드)에 따라 훈련 데이터를 이미 준비했다고 가정합니다.

OpenAI CLI를 사용하여 미세 조정 작업을 시작합니다:

```bash
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
```

여기서 `BASE_MODEL`은 시작할 기본 모델(ada, babbage, curie 또는 davinci)의 이름입니다.
[접미사 매개 변수](#모델명-개인화)를 사용하여 미세 조정된 모델의 이름을 사용자 지정할 수 있습니다.

위 명령을 실행하면 아래와 같은 작업들이 수행됩니다.

1. [파일 API][open_ai_files]를 사용하여 파일을 업로드합니다. (또는 이미 업로드된 파일 사용)
2. 미세 조정 작업을 만듭니다.
3. 작업이 완료될 때까지 이벤트를 스트리밍(이 작업은 몇 분이 소요되는 경우가 많지만 대기열에 작업이 많거나 데이터셋이 큰 경우 몇 시간이 걸릴 수 있음)합니다.

모든 미세 조정 작업은 기본 모델에서 시작되며 기본 모델은 curie입니다.
모델 선택은 모델의 성능과 미세 조정된 모델 실행 비용에 모두 영향을 미칩니다.
모형은 ada, babbage, curie 또는 davinci 중 하나일 수 있습니다.
세부 요금에 대한 자세한 내용은 OpenAI의 [가격 페이지][open_ai_pricing]를 참조하세요.

미세 조정 작업을 시작한 후 완료하는 데 시간이 걸릴 수 있습니다.
귀하의 작업은 당사 시스템의 다른 작업 뒤에 대기될 수 있으며,
모델 및 데이터 세트 크기에 따라 모델을 훈련하는 데 몇 분 또는 몇 시간이 걸릴 수 있습니다.
이벤트 스트림이 모종의 이유로 중단된 경우 다음을 실행하여 다시 시작할 수 있습니다:

```bash
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
```

작업이 완료되면 미세 조정된 모델의 이름이 표시됩니다.

미세 조정 작업을 생성할 수 있을 뿐만 아니라 기존 작업을 나열하거나 작업 상태를 검색하거나 작업을 취소할 수도 있습니다.

```bash
# 생성된 모든 미세 조정 나열
openai api fine_tunes.list

# 미세 조정 상태를 검색합니다. 
# 결과 객체는 작업 상태(보류, 실행 중, 성공 또는 실패 중 하나) 및 기타 정보를 포함합니다.
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# 미세 조정 작업 취소
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>
```

### 미세 조정된 모델 사용하기

작업이 성공하면 `fine_tuned_model` 필드에 모델 이름이 입력됩니다.
이제 이 모델을 [Completions API][open_ai_completions_api]의 매개 변수로 지정하고 
[Playground][open_ai_playground]를 사용하여 요청할 수 있습니다.

작업이 처음 완료된 후 모델이 요청을 처리할 준비가 되는 데 몇 분 정도 걸릴 수 있습니다.
모델에 대한 완료 요청이 시간 초과되면 모델이 계속 로드되고 있기 때문일 수 있습니다.
이 경우 몇 분 후에 다시 시도하십시오.

모델 이름을 완료 요청의 모델 매개 변수로 전달하여 요청을 시작할 수 있습니다:

Open AI CLI:

```bash
openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>
```

cURL:

```bash
curl https://api.openai.com/v1/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": YOUR_PROMPT, "model": FINE_TUNED_MODEL}'
```

Python:

```python
import openai

openai.Completion.create(
    model=FINE_TUNED_MODEL,
    prompt=YOUR_PROMPT)
```

Node.js:

```javascript
const response = await openai.createCompletion({
  model: FINE_TUNED_MODEL
  prompt: YOUR_PROMPT,
});
```

### 미세 조정된 모델 삭제하기

미세 조정된 모델을 삭제하려면, "소유자" 권한이 필요합니다.

Open AI CLI:

```bash
openai api models.delete -i <FINE_TUNED_MODEL>
```

cURL:

```bash
curl -X "DELETE" https://api.openai.com/v1/models/<FINE_TUNED_MODEL> \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Python:

```python
import openai

openai.Model.delete(FINE_TUNED_MODEL)
```

## 데이터 준비하기

### 데이터 규격화

### 일반 모범 사례

### 구체적인 가이드라인

#### 범주화

##### 케이스 스터디 1.

##### 케이스 스터디 1.

##### 케이스 스터디 1.

#### 조건부 생성

##### 케이스 스터디 1.

##### 케이스 스터디 1.

##### 케이스 스터디 1.

## 고급 사용법

### 모델명 개인화

### 미세 조정 모델 분석하기

#### 분류별 메트릭

#### 유효화

#### 하이퍼 파라미터

### 미세 조정된 모델에서 미세 조정 계속

## 비중 과 편향

미세 조정을 가중치 & 바이어스와 동기화하여 실험, 모델 및 데이터 세트를 추적할 수 있습니다.

시작하려면 가중치 & 바이어스 계정과 유료 OpenAI 요금제가 필요합니다. 최신 버전의 openai와 wandb를 사용하고 있는지 확인하려면 다음을 실행합니다:

```bash
pip install --upgrade openai wandb
```

가중치 & 바이어스와 미세 조정을 동기화하려면 다음을 실행합니다:

```bash
openai wandb sync
```

이 통합에 대한 자세한 내용은 가중치 & 바이어스 설명서를 참조하십시오.

## 예시 코드 - 주피터 노트북

### 범주화

### 질의 응답

[open_ai_pricing]: https://openai.com/pricing

[jsonl_official]: https://jsonlines.org/

[open_ai_completions_api]: https://platform.openai.com/docs/api-reference/completions/create

[open_ai_playground]: https://platform.openai.com/playground

[open_ai_files]: https://platform.openai.com/docs/api-reference/files