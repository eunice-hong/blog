---
layout: post
title: "ERC-20 컨트랙 직접 생성하기: Solidity 10일 챌린지 - 05일차"
date:   2023-06-14 23:01:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- 10-days-of-solidity
category: book
author: eunice-hong
description: ERC-20 토큰 표준에 대해 설명하고 이를 사용하여 토큰의 코드를 작성하는 방법을 안내합니다.  블록체인과 암호화폐의 중요성이 증가함에 따라 사람들은 자신만의 토큰을 만들고 싶어합니다. ERC-20은 가장 인기 있는 교환 가능한 토큰 표준으로, 토큰의 규칙과 규정을 정의합니다. ERC-20 토큰을 생성하기 위해 Solidity 언어를 사용하여 스마트 계약을 작성해보고, 토큰의 이름, 기호, 소수점 단위 할당, 토큰의 소유권 전송 및 허용, 잔액 조회, 토큰 전송 등의 기능을 구현해봅니다.
languages: ["ko"]
---

# 5일차: 간단한 ERC-20 계약 작성

앞서 논의한 바와 같이 블록체인은 비트코인과 이더리움과 같은 암호화폐 시스템에서 거래 기록을 안전하고 분산적으로 유지하는 데 중요한 기능을 하는 것으로 잘 알려져 있습니다. 이 암호화폐 열풍이 지금 당장 전 세계를 장악하면서, 사람들은 그들만의 (흔히 거래 가능한) 토큰을 만드는 방법을 알고 싶어 합니다. 스마트 계약은 엔지니어가 원하는 것과 방법을 코딩할 수 있는 자유를 허용하기 때문에, 교환 가능한 토큰에 대한 표준이 필수적이다. 오늘 우리는 가장 인기 있는 교환 가능한 토큰의 표준인 ERC-20에 대해 이야기하고 이 계약을 사용하여 당신의 첫 번째 토큰의 코드를 작성할 것입니다.

## ERC-20 표준은 무엇입니까?

토큰은 바우처, IOU, 심지어 실제 사물을 포함하여 이더리움 시스템에서 광범위한 디지털 자산을 나타냅니다. 이더리움 토큰은 본질적으로 이더리움 네트워크에서 실행되는 스마트 계약이다. ERC-20은 대다수 스마트 계약의 트렌드 표준으로 등장한 그러한 토큰 중 하나이다. 그것은 모든 ETH 기반 토큰이 따라야 하는 다양한 규칙과 규정을 열거합니다. 여기에는 토큰 소유권을 이전할 수 있는 방법과 트랜잭션을 승인하는 방법이 포함됩니다.

결과적으로, 이 토큰은 모든 종류의 개발자들이 미래의 토큰이 이더리움 시스템 내에서 어떻게 행동할지 정확하게 예측할 수 있게 합니다. 토큰이 규칙을 충족하는 한 새로운 토큰이 게시될 때마다 새로운 프로젝트를 수행할 필요가 없다는 것을 알고 개발자들이 작업을 계속할 수 있기 때문에 프로세스가 단순화됩니다. 다양한 이더리움 토큰의 호환성을 보장하기 때문에 이러한 규정 준수도 필요합니다. 이제 토큰 표준에 관한 기본 사항을 몇 가지 살펴보았으니 ERC-20 계약을 작성해 보겠습니다.

## ERC-20 스마트 계약 작성 시작하기

이제 ERC-20에 대한 몇 가지 기본 사항과 토큰 생성에서 ERC-20이 수행하는 중요한 역할에 대해 설명했으므로, ERC-20을 생성하는 과정으로 넘어갑시다. 이것은 간단한 ERC-20 토큰을 작성하는 방법을 배울 수 있는 쉬운 튜토리얼입니다. 전체 코드를 바로 확인하고 싶은 분은 [여기](https://github.com/ismailvohra/10-days-of-solidity/blob/main/day5.md#final-code)에서 확인하세요. 시작합시다!

- 우리는 `VOH 코인`이라는  우리의 토큰을 만들 것입니다.

```solidity
// SPDX-License-Identifier: GPL-3.0

pragma solidity ^0.8.9;

contract VOHCoinERC20{} 
```

- 코드의 첫 줄에서, 우리는 코드가 작성된 Solidity 버전과 함께 라이센스 식별자를 정의합니다. 위의 코드에서 Solidity 버전은 0.8.0 이상임을 알 수 있습니다. 그 아래를 보시면, `contract` 키워드를 사용하여 계약을 선언하고 VOHCoINERC20이라는 이름을 붙였습니다.
- 이 단계가 끝나면 계약 내에서 양도 및 승인 이벤트를 선언할 예정입니다:

```solidity
Transfer(address indexed from, address indexed to, uint tokens);

eventApproval(address indexed tokenOwner, address indexed spender, uint tokens);
```

#### 토큰의 이름, 기호 및 십진수 할당

- 이벤트를 선언한 후 토큰에 대한 기호, 이름 및 (십진수/소수점) 할당으로 넘어갑니다:

```solidity
string public constant name = "VOH Coin";
string public constant symbol = "VOHN";
uint8 public constant decimals = 18;
```
- 이 경우 우리의 토큰 이름은 "VOH Coin"이고 기호는 "VOHN"입니다. 소수는 각각 18로 설정된다.

#### 매핑 선언

- 이제 위의 단계를 마쳤으므로 두 매핑을 선언합니다:

```solidity
mapping(address => uint256) balances;

mapping(address => mapping (address => uint256)) allowed;
```
- Solidity에서 mapping이란 비교 가능한 키-값 쌍을 의미합니다. `balances`의 키는 `address`이고 값은 `uint256`(부호가 없는 256비트 정수)입니다.
- Solidity 문서에 따르면, `address` 은 산술 연산을 지원하지 않는 160비트 값입니다. 외부 계정에 속하는 키 쌍의 공용 절반의 해시나 계약 `address`를 저장하는 데 사용할 수 있습니다.
- `balances`는 키 `address`를 `uint256 int`에 매핑합니다:

| ADDRESS | UINT256 |
|---------|---------|
| 0x01    | 23      |
| 0x02    | 10      |
| 0x03    | 2       |

#### 액세스 가능한 토큰의 총 수량 정의

- 다음 코드는 계약에서 액세스할 수 있는 토큰 수를 선언하는 데 사용됩니다.

```solidity
uint256 totalSupply_;
```

#### 순 공급 및 균형의 가치 정의

- 클래스 생성 시에 호출되는 생성자에 대해 배워봅시다. 스마트 계약의 경우, 우리는 계약이 원하는 네트워크에 배치되었을 때, 생성자를 호출합니다.

```solidity
constructor(uint256 total) {
    totalSupply_ = total;
    balances[msg.sender] = totalSupply_;
}
```

- 위의 코드에서, 우리는 우리가 계약에서 원하는 총 토큰 수(총)로 생성자를 부른다. 총계는 totalSupply_와 같고 배포 `address`의 잔액은 총 토큰과 같습니다. 현재 실행 중인 계약 함수의 이더리움 계정은 `msg.sender` 변수에 저장됩니다.

#### 소유자 잔액 획득

- `balanceOf` 함수를 사용해보겠습니다:

```solidity
function balanceOf(address tokenOwner) public view returns (uint) {
  return balances[tokenOwner];
}
```

- `tokenOwner`는 위 함수에서 사용되는 인수로, 토큰 소유자의 주소입니다. 이 계약에서 토큰의 잔액을 환불할 때 사용합니다. 결과적으로 이 함수는 `balances`에서 토큰 소유자 `address`를 사용해, 잔액을 검색합니다.

#### 원하는 계정으로 토큰 전송

- 우리는 이전 방법으로 작업할 것입니다:

```solidity
function transfer(address receiver, uint numTokens) public returns (bool) {
  require(numTokens <= balances[msg.sender]);
  balances[msg.sender] -= numTokens;
  balances[receiver] += numTokens;
  emit Transfer(msg.sender, receiver, numTokens);
  return true;
}
```

이 함수는 다음 인수로 구성됩니다:

- `receiver`: 토큰을 받을 계정 `address`
- `numTokens`: 수신자 계정으로 보낼 토큰의 양(Num)토큰
- 함수 안에서 배포자의 `address` 잔액에 따라 수신자에게 제공될 토큰의 양이 충분한지 확인합니다.
- 그런 다음 배포자의 계정에서 `numToken`만큼 차감되고, 해당 분은 수신자 계정에 가산됩니다. 전송 이벤트가 실행되고, 부울 값 `true`가 반환되며 마무리됩니다.

#### 토큰 전송 승인

- `approve` 함수를 사용해보겠습니다.

```solidity
function approve(address delegate, uint numTokens) public returns(bool) {

    allowed[msg.sender][delegate]= numTokens;
    
    emit Approval(msg.sender, delegate, numTokens);
    
    return true;

}
```

- `delegate`와 `numTokens`은 함수의 인자입니다.
- `delegate`는 배포자가 보낼 수 있는 토큰 수를 설정할 `address`입니다.
- `numTokens`는 배포자가 `delegate`에게 보낼 수 있는 토큰 수를 의미합니다.
- 함수 내부적으로는 `allowed` 매핑에서 `delegate`을 이용해, 토큰 수를 저장합니다. 그런 다음 Approval 이벤트가 전송되고 true가 반환되며 마무리됩니다.

#### 원하는 계정의 허용 상태 획득

- `allowance` 함수를 사용해보겠습니다.

```solidity
function allowance(address owner, address delegate) public view returns(uint){
    return allowed[owner][delegate];
}
```
이 함수는 다음 인수로 구성됩니다:

- `owner`와 `delegate`. `owner`는 `delegate`에 있는 수신자에게 전송 가능한 토큰 수를 반환하는 `address`입니다.

#### 한 계정에서 다른 계정으로 토큰 전송

- 토큰 전송 로직은 다음과 같이 작성합니다:

```solidity
function transferFrom(address owner, address buyer, uint numTokens) public returns (bool) {
    require(numTokens <= balances[owner]);
    require(numTokens <= allowed[owner][msg.sender]);
    balances[owner] -= numTokens;
    allowed[owner][msg.sender] -= numTokens;
    balances[buyer] += numTokens;
    emit Transfer(owner, buyer, numTokens);
    return true;
}
```

함수 **transferFrom**는 **owner**, **buyer***, ***numTokens**이라는 인수가 있습니다.

- `owner`는 토큰 출금 대상 계좌의 주소를 나타내고, `buyer`는 토큰 입금 대상 계좌의 주소를 나타냅니다. `numTokens`은 `owner`가 `buyer`에게 전송할 토큰의 수를 나타냅니다.
- 먼저 함수 내부에서 `owner`의 잔액이 충분한지, `owner`가 `buyer`에게 해당 수량의 토큰을 전송할 권한이 있는지 확인합니다.
 the transfer is made by removing the number of tokens from the owner's balance and the authorized balance to complete the transfer. 
- 이후, `owner` 잔액에서 토큰 수를 차감하고, 인가된 잔액을 제거해 토큰이 전송됩니다. `buyer`의 잔액은 구매한 토큰 수만큼 증가합니다. `Transfer` 이벤트가 발생하고 `true` 값이 반환됩니다.

## 최종 코드

계약의 전체 코드는 다음과 같습니다:

```solidity
// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.9;

contract VOHCoinERC20 {

    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);

    string public constant name = "VOH Coin";
    string public constant symbol = "VOHN";
    uint8 public constant decimals = 18;

    mapping(address => uint256) balances;

    mapping(address => mapping (address => uint256)) allowed;

    uint256 totalSupply_;

    constructor(uint256 total) {
      totalSupply_ = total;
      balances[msg.sender] = totalSupply_;
    }

    function totalSupply() public view returns (uint256) {
      return totalSupply_;
    }

    function balanceOf(address tokenOwner) public view returns (uint) {
        return balances[tokenOwner];
    }

    function transfer(address receiver, uint numTokens) public returns (bool) {
             
        require(numTokens <= balances[msg.sender]);
        balances[msg.sender] -= numTokens;
        balances[receiver] += numTokens;
        emit Transfer(msg.sender, receiver, numTokens);
        return true;
    }

    function approve(address delegate, uint numTokens) public returns (bool) {
        allowed[msg.sender][delegate] = numTokens;
        emit Approval(msg.sender, delegate, numTokens);
        return true;
    }

    function allowance(address owner, address delegate) public view returns (uint) {
        return allowed[owner][delegate];
    }

    function transferFrom(address owner, address buyer, uint numTokens) public returns (bool) {
        require(numTokens <= balances[owner]);
        require(numTokens <= allowed[owner][msg.sender]);

        balances[owner] -= numTokens;
        allowed[owner][msg.sender] -= numTokens;
        balances[buyer] += numTokens;
        emit Transfer(owner, buyer, numTokens);
        return true;
    }
}
```

첫 번째 ERC-20 토큰이 준비되었습니다. 리믹스를 통해 로컬 가상 머신에 구현하여 계약을 활용할 수 있습니다([4일차][10-days-of-solidity-day-4]에 다루었습니다).

## 마치며

오늘, 우리는 이더리움 네트워크에서 당신만의 ERC-20 토큰을 만드는 과정을 배웠습니다. ERC-20 토큰은 교환 가능한 토큰 표준이며, 이 시리즈의 후반부에서 교환 불가능한 토큰(NFT)에 대해 다룰 것입니다. 행운을 빌어요, 언제나처럼, 해피 코딩!


[10-days-of-solidity]: {{ site.baseurl_root }}/Writing-Hello-World-Contract-using-remix