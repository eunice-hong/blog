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

다양한 작업에 대한 훈련 데이터를 준비하는 방법에 대한 자세한 지침은 [데이터 준비 모범 사례](#모범-사례)를 참조하세요.

예시가 많을수록 좋습니다.
적어도 200개의 예를 갖추는 것을 추천합니다.
연구 결과, 일반적으로 데이터셋 크기를 두 배로 늘릴 때마다 모델 품질이 선형적으로 증가합니다.

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
모델은 ada, babbage, curie 또는 davinci 중 하나일 수 있습니다.
세부 요금에 대한 자세한 내용은 OpenAI의 [가격 페이지][open_ai_pricing]를 참조하세요.

미세 조정 작업을 시작한 후 완료하는 데 시간이 걸릴 수 있습니다.
귀하의 작업은 당사 시스템의 다른 작업 뒤에 대기될 수 있으며,
모델 및 데이터셋 크기에 따라 모델을 훈련하는 데 몇 분 또는 몇 시간이 걸릴 수 있습니다.
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

미세 조정은 사용 사례에 맞는 새로운 모델을 만드는 강력한 기술입니다.
모델을 미세 조정하기 전에 아래의 [모범 사례](#모범-사례) 및 
[사용 사례에 대한 상황별 지침](#상황별-지침)을 읽는 것이 좋습니다.

### 데이터 형식 지정

모델을 미세 조정하려면
각각 단일 입력(이하 `프롬프트`)과 관련 출력(이하 `완료`)으로 구성된 일련의 훈련 예제가 필요합니다.
기본 모델에는 단일 프롬프트에 구체적인 지시사항이나 여러 예제를 입력하지만, 
미세 조정에 사용되는 훈련 예제는 이와 현저히 다릅니다.

* 각 `프롬프트`는 프롬프트가 끝나고, `완료`가 시작되는 부분에 고정 구분 기호를 넣어 모델에 알려야 합니다.
  일반적으로 잘 작동하는 간단한 구분 기호는 `\n\n##\n\n`입니다.
  `프롬프트` 내 다른 곳에 해당 구분 기호를 사용하면 안됩니다.
* 모든 값은 [형태소 분석(tokenization)][open_ai_tokenizer] 되기 때문에, 
  각 `완료`는 공백으로 시작해야 합니다. 
  형태소 분석은 각 단어 앞에 있는 공백을 기준으로 단어를 분리합니다.
* 각 `완료`는 `완료`가 종료될 때 모델에 알리기 위해 정지 시퀀스 고정값으로 끝나야 합니다.
  정지 시퀀스는 `\n`, `###` 또는 `완료`에 나타나지 않는 다른 형태소를 사용할 수 있습니다.
* 모델의 추론을 돕기 위해, 훈련 데이터셋를 만들 때와 동일한 방식으로 `프롬프트` 형식을 지정해야 합니다.
  동일한 구분 기호를 사용해야하고, `완료`를 올바르게 자르기 위해 동일한 정지 시퀀스를 지정합니다.

### 모범 사례

미세 조정은 고품질 예제가 많을수록 성능이 향상됩니다.
고품질 프롬프트를 사용하는 것보다 더 좋은 성능을 내기 위해 미세조정을 하려면 어떻게 해야할까요?
기본 모델로 모델을 미세 조정하려면 인간 전문가가 이상적으로 검토한 
수백 개의 고품질 예제를 제공해야 합니다. 
고품질 예제를 사용한다면, 예제 수가 두 배로 증가할 때마다 성능이 선형적으로 증가시킬 수 있습니다.
예제 수를 늘리는 것이 일반적으로 성능을 향상시키는 가장 좋은 방법입니다.

분류기(Classifier)는 시작하기 가장 쉬운 모델입니다. 
분류 작업의 경우, Open AI 기본 모델 중 ada 를 권합니다.
ada는 다른 모델들에 비해 성능은 살짝 뒤쳐질지 모르지만, 
일반적으로 한 번 미세 조정되면 훨씬 빠르고 저렴합니다.

프롬프트를 백지에서부터 작성하는 대신 기존 데이터셋로 미세 조정한다면,
가능한 한 수작업으로 데이터 내에 공격적이거나 부정확한 내용이 있는지 검토하세요.
특히, 데이터셋의 랜덤 샘플이 큰 경우 가능한 한 많이 검토하십시오.

### 상황별 지침

미세 조정은 다양한 문제를 해결할 수 있으며, 최적의 사용 방법은 사용 상황에 따라 달라질 수 있습니다.
다음은 미세 조정을 위한 가장 일반적인 사용 사례와 해당 지침입니다.

* [분류](#분류)
  1. [광고 속 정보 오류 찾기](#사례-연구-광고-속-정보-오류-찾기)
  2. [감정 분석하기](#사례-연구-감정-분석하기)
  3. [이메일 카테고리 분류하기](#사례-연구-이메일-카테고리-분류하기)
* [조건부 생성](#조건부-생성)
  1. [위키피디아를 기반으로 매력적인 광고 작성하기](#사례-연구-위키피디아를-기반으로-매력적인-광고-작성하기)
  2. [개체 추출하기](#사례-연구-개체-추출하기)
  3. [고객 지원 챗봇 만들기](#사례-연구-고객-지원-챗봇-만들기)
  4. [제품 상세정보를 기반으로 한 제품 설명 작성하기](#사례-연구-제품-상세정보를-기반으로-한-제품-설명-작성하기)

#### 분류

분류 문제에서 `프롬프트`의 각 입력은 미리 정의된 클래스 중 하나로 분류되어야 합니다.
이러한 유형의 문제에 대해서는 다음을 권장합니다:

* 프롬프트 끝에 구분 기호(예: `\n\n###\n\n`)를 사용합니다.
* 최종적으로 모델에 요청할 때도 이 구분 기호를 추가해야 합니다.
* 단일 토큰에 매핑할 클래스를 선택합니다.
* 분류에는 첫 번째 토큰만 필요하므로 추론 시 `max_token=1`을 지정합니다.
* 구분 기호를 포함하여 `프롬프트` + `완료`의 토큰 개수가 2,048개를 초과하지 않는지 확인합니다.
* 클래스당 최소 100개의 예제를 목표로 합니다.
* 클래스 로그 확률을 얻으려면 모델을 사용할 때 `logprob=5`(5개의 클래스에 대해)를 지정할 수 있습니다.
* 미세 조정에 사용되는 데이터셋가 모델이 사용될 대상과 구조 및 작업 유형이 유사한지 확인합니다.

##### 사례 연구: 광고 속 정보 오류 찾기

웹 사이트의 광고에 제품과 회사가 올바르게 언급되어 있는지 확인하려고 합니다.
즉, 모델이 조작을 하고 있지 않은지 확인하려고 합니다.
잘못된 광고를 걸러내는 분류기를 미세 조정할 수 있습니다.

데이터셋은 다음과 같은 형식을 사용할 수 있습니다:

```jsonlines
{"prompt":"Company: BHFF insurance\nProduct: allround insurance\nAd:One stop shop for all your insurance needs!\nSupported:", "completion":" yes"}
{"prompt":"Company: Loft conversion specialists\nProduct: -\nAd:Straight teeth in weeks!\nSupported:", "completion":" no"}
```

위의 예에서는 회사 이름, 제품 및 관련 광고가 포함된 구조화된 입력을 사용했습니다.
구분 기호로 `\nSupported:`를 사용하여 `프롬프트`와 `완료`를 명확하게 구분했습니다.
충분한 수예를 들어, 분리막은 프롬프트 또는 완료 내에 나타나지 않는 한(통상 0.4% 미만) 큰 차이가 없다.

이 사용 사례를 위해 우리는 ada 모델을 미세 조정했습니다.
분류 작업이기 때문에 성능이 더 큰 모델들보다 더 빠르고 저렴한 모델을 사용하는 것이 좋습니다.

이제 `완료` 요청을 보내, 모델에게 문의할 수 있습니다.

```bash
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "Company: Reliable accountants Ltd\nProduct: Personal Tax help\nAd:Best advice in town!\nSupported:",
    "max_tokens": 1,
    "model": "YOUR_FINE_TUNED_MODEL_NAME"
  }'
```

`예` 또는 `아니오` 중 하나를 반환합니다.

##### 사례 연구: 감정 분석하기

특정 트윗의 긍/부정 정도를 등급으로 매기고 싶다고 해봅시다.
데이터셋은 다음과 같이 보일 수 있습니다:

```jsonlines
{"prompt":"새 아이폰이 생겨 넘 즐거워요! ->", "completion":" positive"}
{"prompt":"@lakers 3일 연속 밤에 실망 https://t.co/38EFe43 ->", "completion":" negative"}
```

모델이 미세 조정되면 `완료` 요청에 `logprob=2`를 설정하여
첫 번째 `완료` 토큰에 대한 로그 확률을 다시 가져올 수 있습니다.
긍정적인 계층에 대한 확률이 높을수록 상대적인 정서도 높아집니다.

이제 `완료` 요청을 하여 모델을 쿼리할 수 있습니다.

```bash
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "https://t.co/f93xEd2 제 최신 블로그 게시물을 공유하게 되어 기뻐요! ->",
    "max_tokens": 1,
    "model": "YOUR_FINE_TUNED_MODEL_NAME"
  }'
```

위 명령어는 아래와 같은 결과를 반환합니다.

```jsonlines
{
  "id": "cmpl-COMPLETION_ID",
  "object": "text_completion",
  "created": 1589498378,
  "model": "YOUR_FINE_TUNED_MODEL_NAME",
  "choices": [
    {
      "logprobs": {
        "text_offset": [
          19
        ],
        "token_logprobs": [
          -0.03597255
        ],
        "tokens": [
          " positive"
        ],
        "top_logprobs": [
          {
            " negative": -4.9785037,
            " positive": -0.03597255
          }
        ]
      },
      "text": " positive",
      "index": 0,
      "finish_reason": "length"
    }
  ]
}
```

##### 사례 연구: 이메일 카테고리 분류하기

이메일 수신 시, 미리 정의된 많은 카테고리 중 하나로 분류하려고 합니다.
분류하려는 카테고리 수가 많다면, 카테고리를 숫자로 변환하는 것이 좋습니다.
최대 500개까지 카테고리를 숫자로 변환하여 사용할 수 있습니다.
형태소 분석을 고려하여, 숫자 앞에 공백을 추가하면 성능에 약간 도움이 될 수 있습니다.
훈련 데이터를 다음과 같이 구성할 수 있습니다:

```jsonlines
{"prompt":"Subject: <email_subject>\nFrom:<customer_name>\nDate:<date>\nContent:<email_body>\n\n###\n\n", "completion":" <numerical_category>"}
```

예시 데이터는 다음과 같습니다:

```jsonlines
{"prompt":"Subject: Update my address\nFrom:Joe Doe\nTo:support@ourcompany.com\nDate:2021-06-03\nContent:Hi,\nI would like to update my billing address to match my delivery address.\n\nPlease let me know once done.\n\nThanks,\nJoe\n\n###\n\n", "completion":" 4"}
```

위의 예에서 우리는 2,043개 토큰으로 제한된 수신 이메일을 입력으로 사용했습니다.
(이를 통해 4개의 토큰 구분자와 1개의 토큰 완료가 가능하며, 합계는 2,048입니다.)
구분자로 `\n\n##\n`을 사용하여 이메일에서 `###`이 발생하지 않도록 제거했습니다.

#### 조건부 생성

조건부 생성은 어떤 종류의 입력이 주어지면 콘텐츠가 생성되어야 하는 문제이다.
여기에는 패러프레이징, 요약, 개체 추출, 주어진 사양의 제품 설명 쓰기, 챗봇 등이 포함된다.
이러한 유형의 문제에는 다음을 권장합니다:

* 프롬프트 끝에 구분 기호(예: \n\n###\n\n)를 사용합니다.
  최종적으로 모델에 요청할 때도 이 구분 기호를 추가해야 합니다.
* 완료의 끝에 END와 같은 종료 토큰 사용.
* 추론 중에 종료 토큰을 정지 시퀀스로 추가해야 합니다(예: stop=["END"]).
* 최소 500개의 예제를 목표로 합니다.
* 프롬프트 + 완료가 구분 기호를 포함하여 2048개의 토큰을 초과하지 않는지 확인합니다.
* 예제가 고품질인지, 원하는 형식을 따르는지 확인합니다.
* 미세 조정에 사용되는 데이터셋가 모델이 사용될 대상과 구조 및 작업 유형이 매우 유사한지 확인합니다.
* 낮은 학습률과 1-2 에포크만 사용하는 것이 이러한 사용 사례에 더 효과적인 경향이 있습니다.

##### 사례 연구: 위키피디아를 기반으로 매력적인 광고 작성하기

이것은 생성형 사용 사례이므로 제공한 표본이 최상의 품질인지 확인하려고 합니다.
미세 조정된 모델이 주어진 예제의 스타일(및 실수)을 모방하려고 하기 때문입니다.
좋은 출발점은 약 500개의 예시입니다.
샘플 데이터셋은 다음과 같습니다:

```jsonlines
{"prompt":"<Product Name>\n<Wikipedia description>\n\n###\n\n", "completion":" <engaging ad> END"}
```

예시 데이터는 다음과 같습니다:

```jsonlines
{"prompt":"Samsung Galaxy Feel\nThe Samsung Galaxy Feel is an Android smartphone developed by Samsung Electronics exclusively for the Japanese market. The phone was released in June 2017 and was sold by NTT Docomo. It runs on Android 7.0 (Nougat), has a 4.7 inch display, and a 3000 mAh battery.\nSoftware\nSamsung Galaxy Feel runs on Android 7.0 (Nougat), but can be later updated to Android 8.0 (Oreo).\nHardware\nSamsung Galaxy Feel has a 4.7 inch Super AMOLED HD display, 16 MP back facing and 5 MP front facing cameras. It has a 3000 mAh battery, a 1.6 GHz Octa-Core ARM Cortex-A53 CPU, and an ARM Mali-T830 MP1 700 MHz GPU. It comes with 32GB of internal storage, expandable to 256GB via microSD. Aside from its software and hardware specifications, Samsung also introduced a unique a hole in the phone's shell to accommodate the Japanese perceived penchant for personalizing their mobile phones. The Galaxy Feel's battery was also touted as a major selling point since the market favors handsets with longer battery life. The device is also waterproof and supports 1seg digital broadcasts using an antenna that is sold separately.\n\n###\n\n", "completion":"Looking for a smartphone that can do it all? Look no further than Samsung Galaxy Feel! With a slim and sleek design, our latest smartphone features high-quality picture and video capabilities, as well as an award winning battery life. END"}
```

여기서는 위키백과 기사에 여러 문단과 제목이 포함되어 있기 때문에 다중 줄 구분 기호를 사용했습니다.
또한 모델이 `완료`의 종료 시점을 알 수 있도록, 간단한 `END` 를 토큰으로 사용했습니다.


##### 사례 연구: 개체 추출하기

이것은 언어 변환 작업과 유사합니다. 
성능을 향상시키려면 추출된 여러 개체를 알파벳 순으로 정렬하거나 
원래 텍스트에 나타나는 순서와 동일하게 정렬하는 것이 좋습니다. 
이는 모델이 순서대로 생성해야 하는 모든 개체를 추적하는 데 도움이 됩니다. 
데이터셋은 다음과 같이 표시될 수 있습니다:

```jsonlines
{"prompt":"<any text, for example news article>\n\n###\n\n", "completion":" <list of entities, separated by a newline> END"}
```
예시 데이터는 다음과 같습니다:


```jsonlines
{"prompt":"Portugal will be removed from the UK's green travel list from Tuesday, amid rising coronavirus cases and concern over a \"Nepal mutation of the so-called Indian variant\". It will join the amber list, meaning holidaymakers should not visit and returnees must isolate for 10 days...\n\n###\n\n", "completion":" Portugal\nUK\nNepal mutation\nIndian variant END"}
```

텍스트에 여러 줄이 포함될 가능성이 높기 때문에 다중 줄 구분 기호를 사용하는 것이 좋습니다.
뉴스 기사, 위키피디아 페이지, 트윗, 법률 문서 등과 같은 유형의 입력값 `프롬프트`에서 
개체를 추출해야하는 경우 가장 이상적으로 사용할 수 있습니다.


##### 사례 연구: 고객 지원 챗봇 만들기

챗봇은 다음 항목들을 잘 파악해야한다.
1. 상세 대화 내용(주문 세부 정보 등)
2. 전체 대화 내용
3. 최근 메시지 내용

이 사용 사례에서는 같은 대화 내역으로 다양한 대답을 생성해야합니다.
매 답변마다 아주 조금씩 맥락이 달라야 합니다.
이 사용 사례의 모델은 다양한 유형의 고객 문제를 처리하기 위해 사용될 것입니다. 
따라서 수천 가지의 예제가 필요합니다.
챗봇 대화 품질을 위해, 대화 샘플을 면밀히 검토하는 것이 좋습니다. 
별도의 "텍스트 변환 미세 조정 모델"을 통해 요약 텍스트를 추가하여 입력 값으로 사용해볼 수 있습니다.

데이터셋은 다음과 같이 표시할 수 있습니다:

```jsonlines
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent:", "completion":" <response2>\n"}
{"prompt":"Summary: <summary of the interaction so far>\n\nSpecific information:<for example order details in natural language>\n\n###\n\nCustomer: <message1>\nAgent: <response1>\nCustomer: <message2>\nAgent: <response2>\nCustomer: <message3>\nAgent:", "completion":" <response3>\n"}
```

여기서는 의도적으로 서로 다른 유형의 입력 정보를 분리했지만
`프롬프트`와 `완료` 사이에 동일한 형식의 고객-직원 대화 형식을 유지했습니다.
모든 `완료`는 직원에 의해서만 이루어져야 하며, 모델이 추론할 때 `\n`을 정지 시퀀스로 사용할 수 있습니다.

##### 사례 연구: 제품 상세정보를 기반으로 한 제품 설명 작성하기

이 사용 사례에서는 입력 데이터를 자연어로 변환하는 것이 중요합니다.
자연어 변환이 잘 될 수록, 원하는 결과물을 얻을 가능성이 높아집니다.

예를 들어 다음과 같은 형식을 사용할 수 있습니다:

```jsonlines
{"prompt":"Item=handbag, Color=army_green, price=$99, size=S->", "completion":" This stylish small green handbag will add a unique touch to your look, without costing you a fortune."}
```

다음과 같이 잘 작동하지 않음:

```jsonlines
{"prompt":"Item is a handbag. Colour is army green. Price is midrange. Size is small.->", "completion":" This stylish small green handbag will add a unique touch to your look, without costing you a fortune."}
```

결과물 품질을 높이려면, `완료`의 내용이 제공된 설명에 맞게 작성되었는지 확인하세요.
외부 콘텐츠를 자주 참조하시나요?
자동화 툴을 사용해 외부 콘텐츠들을 추가한다면 생산성을 향상시킬 수 있습니다.
이미지 기반 설명 글을 생성하고 싶나요?
이미지 속 원문 설명을 추출하는 알고리즘을 사용하면 도움을 받을 수 있습니다.
`완료`가 한 문장 밖에 되지 않기 때문에, 정지 시퀀스로 `.`를 모델이 추론하는 데에 사용할 수 있습니다.

## 고급 사용법

### 모델명 개인화

접미사 매개 변수를 사용하여 최대 40자의 접미사를 미세 조정된 모델 이름에 추가할 수 있습니다.

OpenAI CLI:
```bash
openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"
```
결과 이름은 다음과 같습니다:

```bash
ada:ft-your-org:custom-model-name-2022-02-15-04-21-04
```

### 미세 조정 모델 분석하기

작업이 완료되면 각 작업에 결과 파일을 첨부합니다. 
이 결과 파일 ID는 미세 조정을 검색할 때와 미세 조정의 이벤트를 볼 때 나열됩니다. 
다음 명령어로 파일을 다운로드할 수 있습니다:

OpenAI CLI:
```bash
openai api fine_tunes.results -i <YOUR_FINE_TUNE_JOB_ID>

```

CURL:
```bash
curl https://api.openai.com/v1/files/$RESULTS_FILE_ID/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" > results.csv
```

`_results.csv` 파일에는 각 훈련 단계에 대한 행이 포함되어 있습니다. 
여기서 단계란 데이터 배치에 대한 하나의 전진 및 후진 패스를 의미합니다. 
각 행에는 단계 번호 외에도 해당 단계에 대한 내용들이 포함됩니다.
아래 필드는 훈련 단계에 대한 정보를 제공합니다:

* `elapsed_token`: 모델이 지금까지 확인한 토큰 수(반복 포함)
* `elapsed_examples`: 지금까지 모델이 확인한 예제의 수(반복 포함). 
  여기서 한 예제는 배치의 한 요소입니다. 
  예를 들어 `batch_size = 4`인 경우 각 단계는 `elapsed_examples`를 4만큼 늘립니다.
* `training_loss`: 훈련 배치에서의 손실
* `training_sequence_accuracy`: 모델의 예측 토큰과 실제 `완료` 토큰의 일치율을 나타냅니다. 
  예를 들어, `batch_size`가 3이고, 데이터에 `완료` [1, 2], [0, 5], [4, 2]가 포함되어 있고,
  모델의 예측 `완료`에 [1, 1], [0, 5], [4, 2]가 포함되어 있으면 이 정확도는 2/3 = 0.67이 됩니다
* `training_module_accuracy`: 모델에 의해 올바르게 예측된 훈련 배치의 토큰 백분율입니다.
  예를 들어, `batch_size`가 3인 경우 데이터에 완성도 [1, 2], [0, 5], [4, 2]가 포함되어 있고 
  모델이 예측한 [1, 1], [0, 5], [4, 2]가 포함되어 있으면 이 정확도는 5/6 = 0.83이 됩니다

#### 분류 작업 관련 메트릭

또한 정확도 및 가중 F1 점수와 같은 추가 분류별 메트릭을 결과 파일에 생성하는 옵션을 제공한다. 
이러한 메트릭은 전체 유효성 검사 세트에 대해 정기적으로 계산되고 미세 조정이 완료될 때 계산됩니다. 
결과 파일에 추가 열로 표시됩니다.

이를 활성화하려면 매개 변수 `--compute_classification_metrics`를 설정합니다. 
또한 유효성 검사 파일을 제공하고 다중 클래스 분류의 경우 
`classification_n_classes` 매개 변수를 설정하거나 
이진 분류의 경우 `classification_positive_class`를 설정해야 합니다.

```bash
# For multiclass classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes <N_CLASSES>

# For binary classification
openai api fine_tunes.create \
  -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_OR_PATH> \
  -m <MODEL> \
  --compute_classification_metrics \
  --classification_n_classes 2 \
  --classification_positive_class <POSITIVE_CLASS_FROM_DATASET>
```

`--compute_classification_metrics`를 설정하면 다음 메트릭이 결과 파일에 표시됩니다:

**다중 클래스 분류용**

* **classification/accuracy**: 정확도
* **classification/weighted_f1_score**: 가중 F-1 점수

**이진 분류용**

다음 메트릭은 0.5의 분류 임계값을 기반으로 합니다.
(예: 확률이 0.5보다 높을 때, 예제는 양의 클래스에 속하는 것으로 분류됩니다.)

* **classification/accuracy**
* **classification/precision**
* **classification/recall**
* **classification/f{beta}**
* **classification/auroc** - AUROC
* **classification/auprc** - AUPRC

위에서 설명한 것처럼 단일 토큰으로 토큰화하는 클래스에 대해 텍스트 레이블을 사용한다고 가정합니다. 
이러한 조건이 충족되지 않으면, 여러분이 얻는 숫자가 틀릴 가능성이 높습니다.

#### 유효화
유효성 검사를 위해 일부 데이터를 예약할 수 있습니다. 
유효성 검사 파일의 형식은 열차 파일과 정확히 동일하며, 
열차와 유효성 검사 데이터는 상호 배타적이어야 합니다.

미세 조정 작업을 생성할 때 유효성 검사 파일을 포함하는 경우 
생성된 결과 파일에는 훈련 중 주기적인 간격으로 미세 조정된 모델이 
유효성 검사 데이터에 대해 얼마나 잘 수행되는지에 대한 평가가 포함됩니다.

OpenAI CLI:
```bash
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> \
  -v <VALIDATION_FILE_ID_OR_PATH> \
  -m <MODEL>
```

유효성 검사 파일을 제공한 경우 훈련 시간 동안 
유효성 검사 데이터 배치에 대한 메트릭을 주기적으로 계산합니다. 
결과 파일에 다음과 같은 추가 메트릭이 표시됩니다:

* `validation_loss`: 유효성 검사 배치의 손실
* `validation_sequence_accuracy`: 모델의 예측 토큰이 실제 완료 토큰과 
  정확히 일치하는 유효성 검사 배치의 완료 비율입니다. 
  예를 들어, batch_size가 3인 경우 데이터에 완성도 [1, 2], [0, 5], [4, 2]가 포함되어 있고
  모델이 예측한 [1, 1], [0, 5], [4, 2]가 포함되어 있으면 이 정확도는 2/3 = 0.67이 됩니다.
* `validation_token_accuracy`: 모델에 의해 올바르게 예측된 유효성 검사 배치의 토큰 백분율입니다. 
  예를 들어, batch_size가 3인 경우 데이터에 완성도 [1, 2], [0, 5], [4, 2]가 포함되어 있고 
*   모델이 예측한 [1, 1], [0, 5], [4, 2]가 포함되어 있으면 이 정확도는 5/6 = 0.83이 됩니다

#### 하이퍼 파라미터
다양한 사용 사례에서 잘 작동하는 기본 하이퍼 파라미터를 선택했습니다. 
필요한 매개 변수는 훈련 파일뿐입니다.

즉, 미세 조정에 사용되는 하이퍼 파라미터를 조정하면 
종종 더 높은 품질의 출력을 생성하는 모델로 이어질 수 있다. 
특히 다음을 구성할 수 있습니다:

* `model`: 미세 조정할 기본 모델의 이름입니다. 
  "ada", "babbage", "curie", "davinci" 중에서 선택할 수 있습니다. 
  모델에 대한 자세한 내용은 [모델 설명서][open_ai_models]를 참조하십시오.
* `n_sys` - 기본값은 4입니다. 모델을 훈련할 에포크 수입니다. 
  이 때, 에포크란 훈련 데이터셋을 통해 이뤄지는 하나의 전체 사이클을 말합니다.
* `batch_size` - 기본값은 훈련 데이터 셋의 예제 수의 0.2% 이하입니다. 
  훈련 데이터 셋의 최대값은 256 입니다.
  배치 크기는 단일 전진 및 후진 패스를 훈련하는 데 사용되는 훈련 예제의 수입니다.
  일반적으로 배치 크기가 클수록 데이터 세트가 더 잘 작동한다는 것을 발견했습니다.
* `learning_rate_batchier` - 최종 `batch_size`에 따라, 기본값은 0.05, 0.1 또는 0.2입니다.
  미세 조정 학습 속도는 사전 훈련에 사용되는 원래 학습 속도에 승수를 곱한 것과 같습니다.
  최상의 결과를 얻기 위해 0.02 - 0.2 범위 내의 값으로 실험하는 것이 좋습니다.
  실험 결과, 우리는 학습 속도가 클수록 종종 배치 크기가 클수록 성능이 더 좋다고 알려져 있습니다.
* `compute_classification_metrics` - 기본값은 False입니다. 
  True로 설정할 경우, 매 에포크 끝에 설정된 검증 데이터셋에 대한 분류별 메트릭을 계산합니다.
  분류별 메트릭으로는 정확도, F-1 점수 등이 있습니다.
  분류 작업을 위한 미세 조정에 사용할 때 유용합니다.

이러한 추가 하이퍼 파라미터를 구성하려면, OpenAI CLI의 명령어 플래그를 통해 아래와 같이 전달합니다:

```bash
openai api fine_tunes.create \
  -t file-JD89ePi5KMsB3Tayeli5ovfW \
  -m ada \
  --n_epochs 1
```

### 미세 조정된 모델을 더 미세 조정하기

필요에 맞춰 모델을 이미 미세 조정했으며 통합하려는 추가 훈련 데이터가 있는 경우 
모델에서 미세 조정을 계속할 수 있습니다. 
이를 통해 처음부터 다시 훈련할 필요 없이 모든 훈련 데이터에서 학습한 모델이 생성됩니다.

이렇게 하려면 새 미세 조정 작업을 생성할 때 미세 조정된 모델 이름을 전달합니다(예: `-m curie:ft-<org>-<date>`). 
다른 훈련 매개 변수를 변경할 필요는 없지만, 
새 훈련 데이터가 이전 훈련 데이터보다 훨씬 작을 경우 `learning_rate_multiplier`를 
2배에서 4배로 줄이는 것이 유용할 수 있습니다.


## 가중치 & 바이어스

미세 조정을 [가중치 & 바이어스][open_ai_weight_and_bias]와 동기화하면, 
실험, 모델 및 데이터셋를 추적할 수 있습니다.

시작하려면 [가중치 & 바이어스][open_ai_weight_and_bias] 계정과 OpenAI 유료 요금제가 필요합니다. 
최신 버전의 `openai`와 `wandb`를 사용하고 있는지 확인하려면, 다음을 실행하세요:

```bash
pip install --upgrade openai wandb
```

[가중치 & 바이어스][open_ai_weight_and_bias]와 미세 조정을 동기화 시키려면 다음을 실행합니다:

```bash
openai wandb sync
```

이 통합에 대한 자세한 내용은 [가중치 & 바이어스 설명서][open_ai_weight_and_bias]를 참조하세요.

## 예시 코드 - 주피터 노트북

### 분류하기

* [**finetuning-classification.ipynb**][example_classification_notebook]

이 노트북에서는 입력 텍스트가 야구 또는 하키와 관련이 있는지 여부를 분류할 수 있는 모델을
미세 조정하는 방법을 시연합니다. 이 작업은 노트북에서 4단계로 수행됩니다:

1. 데이터 탐색을 통해 데이터 소스의 개요와 예를 볼 수 있습니다.
2. 데이터 준비는 데이터 소스를 미세 조정에 사용할 수 있는 `JSONL` 파일로 변환합니다.
3. 미세 조정은 미세 조정 작업을 시작하고 결과 모델의 성능을 설명합니다.
4. 모델을 사용하여 예측을 얻기 위해 미세 조정된 모델에 대한 요청을 하는 것을 시연합니다.

### 질의 응답

* [**olm-1-collect-data.ipynb**][example_qa_notebook_1]
* [**olm-2-create-qa.ipynb**][example_qa_notebook_2]
* [**olm-3-train-qa.ipynb**][example_qa_notebook_3]

이 프로젝트의 아이디어는 제공된 텍스트의 몇 단락을 기반으로 질문 답변 모델을 만드는 것입니다.
기본 GPT-3 모델은 단락 내에 답변이 포함되어 있을 때 질문에 대한 답변을 잘 수행하지만,
답변이 포함되어 있지 않을 경우 기본 모델은 어쨌든 답변을 위해 최선을 다하려는 경향이 있으며,
종종 거짓 답변으로 이어집니다.

이를 위한 충분한 맥락이 있는 경우에만 질문에 대답하는 모델을 만들기 위해
먼저 텍스트 단락을 기반으로 질문과 답변의 데이터 세트를 만듭니다.
답변이 있을 때만 모델이 답변하도록 훈련하기 위해 질문이 맥락과 일치하지 않는 적대적 사례도 추가합니다.
이러한 경우 모델이 "질문에 답변하기에 충분한 컨텍스트 없음"을 출력하도록 요청합니다.

이 작업은 세 가지 노트북에서 수행합니다:

1. [첫 번째 노트북][example_qa_notebook_1]은 GPT-3가 사전 교육 중에 보지 못한
   최근 데이터를 수집하는 데 중점을 둡니다. 우리는 올림픽 게임 2020(실제 2021년 여름에 개최됨)이라는 주제를
   선정하여 713개의 독특한 페이지를 다운로드했습니다. 우리는 개별 섹션별로 데이터 세트를 구성했는데,
   이는 질문을 묻고 대답하는 컨텍스트 역할을 할 것입니다.
2. [두 번째 노트북][example_qa_notebook_2]은 위키백과 섹션을 기반으로 몇 가지 질문을 하고
   해당 섹션을 기반으로 질문에 답하기 위해 다빈치 명령을 사용할 것입니다.
3. [세 번째 노트북][example_qa_notebook_3]은 컨텍스트, 질문 및 답변 쌍의 데이터 세트를 활용하여
   질문이 해당 컨텍스트에서 생성되지 않은 적대적 질문 및 컨텍스트 쌍을 추가로 생성합니다.
   이 경우 모델은 "질문에 답변하기에 충분한 컨텍스트가 없습니다"라는 메시지를 표시합니다.
   우리는 또한 문맥을 기반으로 질문에 대답할 수 있는지 여부를 예측하는 판별자 모델을 훈련할 것입니다.

[open_ai_pricing]: https://openai.com/pricing

[jsonl_official]: https://jsonlines.org/

[open_ai_completions_api]: https://platform.openai.com/docs/api-reference/completions/create

[open_ai_playground]: https://platform.openai.com/playground

[open_ai_files]: https://platform.openai.com/docs/api-reference/files

[open_ai_tokenizer]: https://platform.openai.com/tokenizer

[open_ai_models]: https://platform.openai.com/docs/models

[open_ai_weight_and_bias]: https://docs.wandb.ai/guides/integrations/openai

[example_classification_notebook]: https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb

[example_qa_notebook_1]: https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb

[example_qa_notebook_2]: https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-2-create-qa.ipynb

[example_qa_notebook_3]: https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-3-train-qa.ipynb
