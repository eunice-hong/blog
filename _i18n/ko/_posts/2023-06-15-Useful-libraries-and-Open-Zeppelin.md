---
layout: post
title: "관련 라이브러리와 제플린 사용하기: Solidity 10일 챌린지 - 06일차"
date:   2023-06-15 23:01:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- 10-days-of-solidity
category: book
author: eunice-hong
description: 보안과 오버플로우 오류를 방지를 위해 사용 할 수 있는 스마트 계약 라이브러리에 대해 알아봅니다.
  그 중 오픈 제플린를 사용하여 맞춤형 공급 장치가 있는 ERC-20 토큰을 만드는 방법을 안내합니다.
languages: ["ko"]
---

프로젝트에 스마트 계약 라이브러리를 사용하는 데는 다양한 이점이 있습니다. 
우선, 직접 작성하는 것보다 시스템에 포함시킬 수 있는 즉시 사용할 수 있는 건물 조각을 제공하여 시간을 절약합니다. 
가장 큰 이점은 보안입니다. 
오픈 소스인 스마트 계약 라이브러리는 종종 보안에 대한 체크를 받습니다. 
너무 많은 프로젝트가 이를 의존하기 때문에 커뮤니티가 지속적으로 검증할 수 밖에 없는 구조입니다. 
응용 프로그램 코드의 오류는 재사용 가능한 계약 라이브러리보다 훨씬 더 일반적입니다. 
일부 라이브러리에 대해서도 보안을 추가하기 위해 외부 검사가 수행됩니다.

테스트 가능성이 높고, 재사용 가능하며 보안이 철저한 라이브러리가 점점 더 중요해집니다. 
이 글에서는 오픈 제플린을 통해 맞춤형 공급 장치를 사용하여 ERC-20 토큰을 만들 것입니다. 
널리 사용되는 이더리움 라이브러리와 효율적인 계약 작성을 가능하게 하는 방법에 대해 자세히 알아 봅시다!

# SafeMath

Solidity의 산술 연산은 오버플로우를 기반으로 합니다. 
오버플로우는 에러를 일으켜 문제가 되기 십상입니다. 고급 프로그래밍 언어에서 흔히 있는 일이기 때문에 다들 으레 그렇게 생각합니다.  
어떤 동작에 오버플로우가 발생하면, [SafeMath][safe_math]는 트랜잭션을 돌이켜서 상태를 복구할 수 있도록 도와줍니다.

불확실한 함수를 쓰는 것보다 이 라이브러리를 사용하는 것이 클래스 전체에 생길 수 있는 문제를 예방할 수 있게 해줍니다. 
다시 말해, SafeMath를 사용하는 것이 매우 권장됩니다.

# DS Math

Solidity에서 가장 일반적인 수치 프리미티브 유형의 경우 [DS-Math](https://dappsys.readthedocs.io/en/latest/ds_math.html) 는 산술 함수를 제공합니다. 
정수 오버플로우의 위험 없이 안전하게 uint 값을 더하고, 빼고, 곱하고, 나눌 수 있습니다. 
최소와 최대 두 개의 숫자도 찾을 수 있습니다.

이 패키지에는 두 가지 새로운 상위 수준 숫자 개념인 와드(18 소수점)와 광선(27 소수점)에 대한 산술 함수도 포함되어 있습니다. 
고정 소수점은 이로 표시됩니다. 
이 라이브러리는 (기본 계약 대신) 언어로 제공되지 않는 산술 함수에 대한 오버플로 검사를 제공합니다. 
이 라이브러리를 사용하여 계약을 오버플로로부터 보호하는 것이 좋습니다. 
치명적일 수 있습니다!

# Open Zeppelin

OpenZepelin은 Zepelin 팀이 만든 안전한 스마트 계약을 개발하기 위한 라이브러리입니다. 
ERC20 및 ERC721 구현(필요에 따라 그대로 사용하거나 수정할 수 있음)은 물론 맞춤형 계약 및 보다 복잡한 분산 시스템을 만들기 위한 Solidity 구성 요소를 포함합니다. 
안전한 블록체인 애플리케이션을 구성하기 위한 황금 표준으로 환영받고 있습니다. 
[OpenZepelin](https://docs.openzeppelin.com/contracts/3.x/erc20) 은 분산형 애플리케이션을 개발, 자동화 및 실행하기 위한 보안 기술을 만듭니다. 
또한 시스템 및 제품에 대한 보안 평가를 수행하여 상위 기업을 보호합니다. 
Solidity로 작성된 이더리움 시스템을 위한 모듈식, 안전하고 효율적인 스마트 계약 라이브러리를 제공합니다. 
OpenZepelin은 업계에서 가장 인기 있는 라이브러리일 뿐만 아니라 감사 코드를 재사용하여 공격 표면을 줄입니다. 
이제 맞춤형 공급 장치를 사용하여 ERC-20 토큰을 만드는 작업으로 넘어갑니다.

# Using Open Zeppelin Libraries

이제 여러 Open Zepelin 라이브러리를 사용하여 스마트 계약에서 유용한 사용 사례를 확인할 수 있습니다. 
경험적으로 계약을 선언하기 전에 **import** 문장을 사용하여 해당 라이브러리를 계약으로 가져와야 합니다. 
**ERC20.sol**을(리믹스 IDE에서) 참조 목적으로만 가져올 것입니다. 
라이브러리를 상속할 때 **is**라는 용어는 다음과 같이 사용됩니다:

```solidity
//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;
import "github/OpenZeppelin/openzeppelin-contracts/contracts/token/ERC20/ERC20.sol";
contract myERC20 is ERC20 {}
```

# 맞춤형 공급 메커니즘으로 ERC-20 토큰 생성하기

맞춤형 공급 메커니즘으로 ERC-20 토큰을 만드는 방법을 알아봅시다. 
이 튜토리얼에서는 언급된 목적에 맞게 오픈제플린을 사용하는 방법 두가지를 소개하겠습니다. 
공급 없이 ERC-20 토큰을 발행하는 것은 실용적이지 않으므로, 공급이 어떻게 만들어지는 지 정의해야합니다.

### Fixing Supply

계약을 배포하는 계정에 처음 할당된 고정 공급 5,000개의 토큰을 원한다고 가정해봅시다.
계약 v1의 경우 다음 코드를 입력합니다:

```solidity
contract ERC20FixedSupply is ERC20 {
	constructor() public {
		totalSupply += 5000;
		balances[msg.sender] += 5000;
	}
}
```

이 접근 방식은 권장되지 않을 뿐만 아니라 계약 v2에서 금지되어 있습니다.
TotalSupply 및 balances은 비밀 ERC20 구현 세부 사항이므로 직접 작성할 수 없습니다.
대신 동일한 작업을 수행하는 내부 _mint 함수가 있습니다:

```solidity
contract ERC20FixedSupply is ERC20 {
    constructor() public {
        _mint(msg.sender, 5000);
    }
}
```
계약 확장은 이런 식으로 상태가 캡슐화(encapsulated)될 때 더 안전합니다.

### 채굴자 인센티브 제도 (Incentivizing Miners)

내부 함수 **_mint** 는 ERC20 확장을 작성하는 데 중요한 구성 요소입니다. ERC20 확장은 공급 메커니즘을 구현하는데에 사용됩니다.

우리가 사용할 접근 방식은 이더리움 블록을 생성하는 채굴자를 위한 토큰 인센티브 시스템입니다.
현재 블록의 채굴자 주소는 글로벌 변수 블록에서 확인할 수 있습니다.
누군가 우리의 코인에서 **mintMinerReward()** 방법을 실행하면 이 주소로 토큰 보상을 전송합니다:

```solidity
contract ERC20WithMinerReward is ERC20 {
    function mintMinerReward() public {
        _mint(block.coinbase, 5000);
    }
}
```

### Modularizing the Mechanism

ERC20Mintable은 이미 Contracts에 통합되어 있는 공급 메커니즘 중 하나로, 
계정 그룹에게 Minter 역할을 할당하여 _mint의 외부 버전인 민트 함수를 호출할 수 있는 일반적인 기법입니다.

이는 외부에 보유된 계정(즉, 암호 키 쌍을 가진 누군가)이 생산할 공급량을 선택하고 누구에게 분배해야 하는지 선택하는 
중앙 집중식 민팅에 사용될 수 있습니다. 
예를 들어, 전통적인 자산 지원 스테이블 코인은 이 기술에 완벽하게 합리적인 사용 사례입니다.

한편, Minter 역할이 있는 계정은 외부에서 소유할 필요가 없으며, 신뢰할 수 없는 프로세스를 구현하는 스마트 계약이 될 수 있으며, 
앞 절과 동일한 행위를 구현할 수 있습니다.

```solidity
contract MinerRewardMinter {
    ERC20Mintable _token;

    constructor(ERC20Mintable token) public {
        _token = token;
    }

    function mintMinerReward() public {
        _token.mint(block.coinbase, 5000);
    }
}
```

이로 인해 서로 다른 계약에 역할을 할당하여 여러 공급 시스템을 간단하게 결합할 수 있으며, 동적으로 수행할 수 있습니다.

### 보상 자동화를 위한 배포 시스템

_mint 이외에도 _transfer와 같은 다른 내부 ERC20 연산을 활용하거나 향상시킬 수 있습니다. 
이 방법은 토큰 전송을 용이하게 하고 ERC20에 의해 활용되기 때문에 자동으로 기능을 활성화하는 데 사용될 수 있습니다. 
이는 함수 `ERC20Mintable` 에서는 허용하지 않는 것입니다.

이를 활용하여 이전 공급 시스템 외에도 블록체인에 포함된 모든 토큰 전송에 대한 광부 보상을 주조할 수 있습니다:

```solidity
contract ERC20WithAutoMinerReward is ERC20 {
    function _mintMinerReward() internal {
        _mint(block.coinbase, 5000);
    }

    function _transfer(address from, address to, uint256 value) internal {
        _mintMinerReward();
        super._transfer(from, to, value);
    }
}
```

`super`를 호출하기 전에 채굴자 보상을 먼저 민트하기 위해 `_transfer`를 재정의하는 법을 이해하셨나요? 
이전 버전을 수행하기 위해 전송합니다. 
이 마지막 단계는 ERC20 전송의 원래 의미를 유지하는 데 중요합니다.

[safe_math]: https://docs.openzeppelin.com/contracts/3.x/utilities#math