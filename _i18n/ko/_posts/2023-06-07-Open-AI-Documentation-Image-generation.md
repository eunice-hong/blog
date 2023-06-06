---
layout: post
title: "이미지 생성: Open AI 문서 한국어로 읽기"
date:   2023-06-07 01:39:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:

- open-ai

category: book
author: eunice-hong
description: Open AI 문서 이미지 생성(Image generation)을 한국어로 읽어봅니다. 오역, 의역이 있을 수 있습니다.

---

> DALL·E 모델로 이미지를 생성하거나 조작하는 방법에 대해 알아봅니다.

## 들어가기에 앞서

Images API는 이미지와 상호 작용하는 세 가지 방법을 제공합니다:

1. 텍스트 프롬프트를 기반으로 이미지 생성하기
2. 텍스트 프롬프트를 기반으로 기존 이미지 편집하기
3. 기존 이미지의 변형 만들기

이 가이드에서는 유용한 코드 샘플과 함께 이 세 가지 API 끝점을 사용하는 기본 사항을 다룹니다.
그들의 활동을 보려면 [DALL·E 미리보기 앱][dalle_preview_app]을 확인하세요.

> 🧪이미지 API는 베타 버전입니다.
> 이 기간 동안 API와 모델은 사용자의 피드백을 기반으로 진화할 것입니다.
> 모든 사용자가 편안하게 시제품을 제작할 수 있도록 기본 속도 제한은 분당 50개 이미지입니다.
> 요금 제한에 대한 자세한 내용은
> [요금 제한 안내서](https://platform.openai.com/docs/guides/rate-limits)를 참조하십시오.

## 사용방식

### 생성

이미지 생성 끝점을 사용하면 텍스트 프롬프트가 지정된 원본 이미지를 생성할 수 있습니다.
생성된 이미지의 크기는 256x256, 512x512 또는 1024x1024 픽셀일 수 있습니다.
크기가 작을수록 생성 속도가 빠릅니다.
[n 매개 변수](https://platform.openai.com/docs/api-reference/images/create#images/create-n)를
사용하여 한 번에 1-10개의 이미지를 요청할 수 있습니다.

```python
response = openai.Image.create(
    prompt="a white siamese cat",
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
```

자세한 설명이 수록 사용자나 최종 사용자가 원하는 결과를 얻을 가능성이 높아집니다.
[DALL·E 미리보기 앱][dalle_preview_app]에서 예제를 탐색하여 더 많은 영감을 얻을 수 있습니다.
다음은 간단한 예입니다:

| 프롬프트                                                                                             | 생성 결과                                   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------|
| a white siamese cat                                                                              | ![생성 결과 예시 이미지 1][generation_example_1] |
| a close up, studio photographic portrait of a white siamese cat that looks curious, backlit ears | ![생성 결과 예시 이미지 2][generation_example_2] |

각 이미지는 매개 변수 [`response_format`][open_ai_docs_response_format]를 사용하여 URL 또는 Base64 데이터로 반환할 수 있습니다.
URL은 한 시간 후에 만료됩니다.

### 편집

이미지 편집 끝점을 사용하면 마스크를 업로드하여 이미지를 편집하고 확장할 수 있습니다. 
마스크의 투명한 영역은 이미지를 편집해야 하는 위치를 나타내며, 
프롬프트는 지워진 영역뿐만 아니라 전체 새 이미지를 설명해야 합니다. 
이 엔드포인트는 [DALL·E 미리보기 앱][dalle_preview_app]의 편집기와 같은 경험을 가능하게 할 수 있다.

```python
response = openai.Image.create_edit(
    image=open("sunlit_lounge.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="A sunlit indoor lounge area with a pool containing a flamingo",
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
```

| 이미지                           | 마스킹                           | 결과                           |
|-------------------------------|-------------------------------|------------------------------|
| ![편집 입력값 이미지][edit_example_1] | ![편집 마스킹 이미지][edit_example_2] | ![편집 결과 이미지][edit_example_3] |

업로드된 이미지와 마스크는 모두 4MB 미만의 정사각형 PNG 이미지여야 하며 서로 동일한 치수를 가져야 합니다.
마스크의 투명하지 않은 영역은 출력을 생성할 때 사용되지 않으므로
위의 예와 같이 원본 이미지와 일치할 필요가 없습니다.

### 변형

이미지 변경 엔드포인트를 사용하여 지정된 이미지의 변형을 생성할 수 있습니다.

```python
response = openai.Image.create_variation(
    image=open("corgi_and_cat_paw.png", "rb"),
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
```

| 이미지                                    | 변형 결과                                   |
|----------------------------------------|-----------------------------------------|
| ![변형 입력 예시 이미지][variant_example_input] | ![변형 결과 예시 이미지][variant_example_result] |

### 내용 조정

[콘텐츠 정책](https://labs.openai.com/policies/content-policy)에 따라 프롬프트 및 이미지가 필터링되며,
프롬프트 또는 이미지에 플래그가 지정되면 오류가 반환됩니다.
잘못된 긍정 또는 관련 문제에 대한 피드백이 있으면 [헬프 센터](https://help.openai.com/)를 통해 문의하십시오.

## 프로그래밍 언어별 팁

### Python

#### 인메모리 이미지 데이터 사용하기

위 가이드의 Python 예제는 디스크에서 이미지 데이터를 읽기 위해 `open` 함수를 사용합니다.
경우에 따라 이미지 데이터가 메모리에 저장될 수도 있습니다.
아래 코드는 `BytesIO` 객체에 저장된 이미지 데이터를 사용하는 API 호출의 예입니다.

```python
from io import BytesIO

# This is the BytesIO object that contains your image data
byte_stream: BytesIO = [your image data]
byte_array = byte_stream.getvalue()
response = openai.Image.create_variation(
    image=byte_array,
    n=1,
    size="1024x1024"
)
```

#### 이미지 데이터 다루기

API로 전달하기 전에 이미지에 대한 작업을 수행하는 것이 도움이 될 수 있습니다.
다음은 `PIL`을 사용하여 이미지 크기를 조정하는 예입니다:

```python
from io import BytesIO
from PIL import Image

# Read the image file from disk and resize it
image = Image.open("image.png")
width, height = 256, 256
image = image.resize((width, height))

# Convert the image to a BytesIO object
byte_stream = BytesIO()
image.save(byte_stream, format='PNG')
byte_array = byte_stream.getvalue()

response = openai.Image.create_variation(
    image=byte_array,
    n=1,
    size="1024x1024"
)
```

#### 에러 처리

API 요청은 잘못된 입력, 속도 제한 또는 기타 문제로 인해 잠재적으로 오류를 반환할 수 있습니다.
`try...except`문을 통해 처리할 수 있고, 오류에 대한 세부 정보는 `e.error`에서 확인할 수 있습니다:

```python
try:
    openai.Image.create_variation(
        open("image.png", "rb"),
        n=1,
        size="1024x1024"
    )
    print(response['data'][0]['url'])
except openai.error.OpenAIError as e:
    print(e.http_status)
    print(e.error)
```

### Node.js

#### 인메모리 이미지 데이터 사용하기

위 가이드의 Node.js 예제에서는 `fs` 모듈을 사용하여 디스크에서 이미지 데이터를 읽습니다.
경우에 따라 이미지 데이터가 메모리에 대신 저장될 수 있습니다.
다음은 Node.js `Buffer` 개체에 저장된 이미지 데이터를 사용하는 API 호출의 예입니다:

```javascript
// This is the Buffer object that contains your image data
const buffer = [your image data];
// Set a `name` that ends with .png so that the API knows it's a PNG image
buffer.name = "image.png";
const response = await openai.createImageVariation(
  buffer,
  1,
  "1024x1024"
);
```

#### TypeScript 사용하기

TypeScript를 사용하는 경우 이미지 파일 인수와 함께 몇 가지 문제가 발생할 수 있습니다.
다음은 인수를 명시적으로 캐스팅하여 유형 불일치를 해결하는 예입니다:

```typescript
// Cast the ReadStream to `any` to appease the TypeScript compiler
const response = await openai.createImageVariation(
  fs.createReadStream("image.png") as any,
  1,
  "1024x1024"
);
```

다음은 메모리 내 이미지 데이터에 대한 유사한 예입니다:

```typescript
// This is the Buffer object that contains your image data
const buffer: Buffer = [your image data];
// Cast the buffer to `any` so that we can set the `name` property
const file: any = buffer;
// Set a `name` that ends with .png so that the API knows it's a PNG image
file.name = "image.png";
const response = await openai.createImageVariation(
  file,
  1,
  "1024x1024"
);
```

#### 에러 처리

API 요청은 잘못된 입력, 속도 제한 또는 기타 문제로 인해 잠재적으로 오류를 반환할 수 있습니다.
이러한 오류는 `try...catch` 문으로 처리할 수 있으며
오류 세부 정보는 `error.response` 또는 `error.message`에서 찾을 수 있습니다:

```javascript
try {
  const response = await openai.createImageVariation(
    fs.createReadStream("image.png"),
    1,
    "1024x1024"
  );
  console.log(response.data.data[0].url);
} catch (error) {
  if (error.response) {
    console.log(error.response.status);
    console.log(error.response.data);
  } else {
    console.log(error.message);
  }
}
```

[dalle_preview_app]: https://labs.openai.com/

[generation_example_1]: https://cdn.openai.com/API/images/guides/image_generation_simple.webp

[generation_example_2]: https://cdn.openai.com/API/images/guides/image_generation_detailed.webp

[edit_example_1]: https://cdn.openai.com/API/images/guides/image_edit_original.webp

[edit_example_2]: https://cdn.openai.com/API/images/guides/image_edit_mask.webp

[edit_example_3]: https://cdn.openai.com/API/images/guides/image_edit_output.webp

[variant_example_input]: https://cdn.openai.com/API/images/guides/image_variation_original.webp

[variant_example_output]: https://cdn.openai.com/API/images/guides/image_variation_output.webp

[open_ai_docs_response_format]: https://platform.openai.com/docs/api-reference/images/create#images/create-response_format
