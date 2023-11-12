---
layout: post
title: "개발자가 활용하기 좋은 Chat GPT 프롬프팅 패턴 5가지" 
date:   2023-09-04 01:02:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
---


<iframe width="560" height="315" src="https://www.youtube.com/embed/WRkig3VeRLY?si=hvKzEL8rOl6Eomb6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
> 위 유튜브 영상 속 Chat GPT 프롬프팅 패턴을 익히기 위해 작성하는 글입니다.
> 프롬프트 예시는 이후 활용을 위해 영어로 작성합니다.

## 페르소나 패턴 (Persona pattern)

페르소나 패턴은 Chat GPT에게 특정 인격체로서 대답하도록 유도하는 질문 패턴입니다.
이 패턴은 ChatGPT의 답변이 일정한 형태를 갖춰야하거나, 하나의 관점에 초점을 맞춰서 답변을 해야할 경우 유용합니다. 


```text
You will pretend to be a senior engineer at a FAANG company. 
Review the following code, paying attention to security and performance. 
Provide output that a senior engineer would produce regarding the code. 

# 질문 내용을 적어주세요.
```

위 예시는 코드 리뷰를 위해 Chat GPT에게 특정 인격체로서 대답하도록 유도하는 질문 패턴입니다.
저는 이전까지 코드의 품질을 확인하기 위해 Chat GPT를 사용할 때, `Review the code below`라고 아주 간단하게 질문을 했습니다.
단어 4개로 질문을 했을 때는 단순히 코드에 대해 설명을 늘어놓는 답변을 내놓았는데,
위와 같이 페르소나 패턴을 사용하니 코드 내용에 대해 구체적으로 평가해주고, 대안까지 제안해주었습니다.

![chat-gpt-do](https://giphy.com/gifs/pudgypenguins-internet-bard-chatgpt-0lGd2OXXHe4tFhb7Wh){:style="display:block; margin-left:auto; margin-right:auto"}

꼭 개발을 할 때가 아니라더라도, 페르소나 패턴을 사용해볼 수 있을까요?
생각해보니, '출판사 편집장'이라는 인격체를 Chat GPT에게 부여해서, 지금 작성하고 있는 이 블로그 글을 첨삭해달라고 할 수 있겠네요. 
😎

```text
From now on, act as a book editor and review the following blog article, focusing on readability.

# 블로그 글을 적어 주세요.
```

## 레시피 패턴 (Recipe pattern)

레시피 패턴은 목표 달성을 위한 행동 방식을 검증하는 데에 요긴하게 사용할 수 있습니다.

![kimchi-soup](https://media.giphy.com/media/JR6vxmWsXIE3TuPQkQ/giphy.gif){:style="display:block; margin-left:auto; margin-right:auto"}

예를 들어, 김치찌개를 만드는 법에 대해 Chat GPT에게 물어본다고 가정해봅시다.
보통 김치찌개에 김치와 돼지고기가 주요 재료로 사용된다는 사실을 알고, 냄비에 물과 김치, 돼지고기를 넣고 끓인다는 것도 알고 있습니다.
하지만, 어떤 순서로, 어떤 재료를, 얼마나 넣어야하는지, 얼마나 끓여야하는지는 잘 모를 수 있습니다.

이럴 때 레시피 패턴을 사용하면, Chat GPT가 빠진 재료가 있는지, 잘못된 순서로 요리를 하고 있는지, 
끓이는 시간이 적절한지 등을 바로 잡아 줄 수 있습니다.

개발자라면 레시피 패턴을 사용해야하는 상황에 자주 놓입니다.
만들어야하는 소프트웨어가 무엇인지 명확하고, 어떤 언어와 프레임워크를 사용해야하는지 알고 있지만,
어떤 순서로, 어떤 방식으로, 어떤 라이브러리를 사용해야하는지 모를 때가 많습니다.

```text
I'm trying to write a Rust program to encrypt data. 
I know I have to read user input, validate it, encrypt it, and return the encrypted data. 
Please provide a complete sequence of steps for me, filling in any missing steps, 
and identify any unnecessary steps.
```

위 예시는 Chat GPT에게 레시피 패턴을 사용해 암호화 프로그램을 만드는 방법을 물어보는 질문 패턴입니다.
Chat GPT는 레시피 패턴을 사용해, 빠진 단계를 채워주고, 불필요한 단계를 제거해줍니다.

프로젝트 개발을 처음 시작할 때, Chat GPT에게 레시피 패턴을 사용해 질문을 던지면,
프로젝트의 전체적인 흐름을 파악하는 데에 도움이 될 것 같네요.


## 설명 첨부 패턴 (Reflection pattern)

설명 첨부 패턴은 Chat GPT가 답변을 제공할 때, 그 답변에 대한 이유에 대해 설명하도록 유도하는 질문 패턴입니다.

개발하다보면, 어려운 개념이나 기술을 이해해야하는 경우가 많습니다.
구글, 스택오버플로우에서 발견한 코드 조각을 그대로 복사-붙여넣기하기 전에 
코드의 동작 원리와 코드가 문제를 해결하는데에 어떻게 기여하는지 이해하는 것이 중요합니다.

```text
When you provide an answer, please explain the reasoning and assumptions behind your answer. 
Explain your choices and address any potential limitations or edge cases.

# 질문 내용을 적어주세요.
```

개발 중에 문제가 생겼을 때, 운 좋게 구글 검색을 통해 해결책을 발견하는 경우가 있습니다.
그런데, 해결책에는 설명 자체가 없거나 부족할 수 있고, 자세한 설명이 있다 하더라도 내 상황과 관련이 없는 경우가 있습니다.
설명 첨부 패턴을 사용하면, 나에게 꼭 맞는 답변을 쉽게 찾을 수 있을 뿐 아니라 자세한 설명도 함께 얻을 수 있어서 좋습니다. 

## 거부 방지 패턴 (Refusal Breaker pattern)

Chat GPT를 사용해보신 분이라면 아시겠지만, Chat GPT가 모든 질문에 대답해주지는 않습니다.
Chat GPT가 알지 못하는 지식에 대해 물어보거나, 금지된 주제나 단어를 사용하도록 유도할 경우,
답변을 거부해버립니다. 🙈🙉🙊

예를 들어, Chat GPT에게 `How to hack a computer?`라고 질문을 던지면,
Chat GPT는 `I don't know how to hack a computer.`라는 답변을 내놓습니다.

거부 방지 패턴은 Chat GPT가 답변하기 곤란할 수 있는 질문에 대해, 우리가 질문을 재정의하도록 유도하는 질문 패턴입니다.

```text
Whenever you can't answer a question, explain why you can't answer the question 
and provide one or more alternative wordings of the question that you could answer.

# 질문 내용을 적어주세요.
```

거부 방지 패턴을 사용해, 질문에 대해 Chat GPT가 대답을 할 수 있는 방향으로 유도할 수 있습니다.


## 역질문 패턴 (Flipped Interaction pattern)

Chat GPT는 척척박사처럼 답변을 해줍니다.
하지만, Chat GPT가 답변을 해주기 위해서는 우리가 질문을 잘 작성해야합니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/MO0r930Sn_8?si=c2Xy7UraueGeG1YJ" title="YouTube video player" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

세상 모든 걸 쉽게 설명할 수 있는 리처드 파인만도 '왜?'라는 질문은 어려워합니다.
질문하는 사람이 질문에 대해 얼만큼 알고 있는 지에 따라, 답변의 내용이 천차만별로 달라질 수 있기 때문입니다.
위 영상 속의 ‘왜 자석은 서로 밀어내는가’라는 질문을 예로 들자면, 
질의자가 물리학 전공 학부생인지 아무것도 모르는 평범한 사람인지에 따라 답변은 다릅니다.

![hard-to-answer](https://media.giphy.com/media/aKsalVFVKsFxf4MueH/giphy.gif){:style="display:block; margin-left:auto; margin-right:auto"}

Chat GPT에게 질문할 때도 마찬가지로, 우리가 얼만큼 알고 있는지에 대해 설명해주면 좋습니다.
하지만, 어렵죠. 🤦
번거롭고, 귀찮고, 시간도 오래 걸리고, 어떤 정보가 필요한지도 모르겠고, 어떤 순서로 질문을 해야할지도 모르겠고...

```text
I want you to ask me questions to deploy a Rust binary to a web server located in AWS. 
When you have all the information you need, write a bash script to automate the deployment.
```

역질문 패턴은 Chat GPT가 우리에게 질문을 던지도록 유도하는 질문 패턴입니다.
명확하게 달성해야하는 목표가 있지만, 도무지 어디서부터 출발해야할 지 모를 때,
Chat GPT에게 목표 달성을 위한 조건들을 충족했는지 질문해달라고 해보세요.

역질문 패턴을 통해 밑그림 조차 없는 상황에서도 Chat GPT가 우리에게 필요한 정보를 알려줄 수 있습니다.

### 마치며

참고한 영상 속에는 이 글에 언급된 다섯가지 프롬프팅 패턴만 다루고 있습니다.
더 알아보니 [Coursera][prompt-engineering]에는 20가지 프롬프팅 패턴이 소개되어 있더라구요.
기회가 된다면, 나머지 프롬프팅 패턴에 대해서도 소개해보겠습니다.


[prompt-engineering]: https://www.coursera.org/learn/prompt-engineering#modules

