---
layout: post
title: "Solidity 모범 사례: Solidity 10일 챌린지 - 10일차"
date:   2023-06-19 23:01:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- 10-days-of-solidity
category: book
author: eunice-hong
description: Solidity 모범 사례(Best Practices) 모음
languages: ["ko"]
---

부실하게 설계되고 작성된 코드는 장기적으로 기술 부채가 되어 당신을 괴롭힐 것 입니다. 
본인이 작성한 코드가 본인 스스로나 다른 동료 개발자들을 계속해서 불안하게 만들 수 있습니다! 그 누구도 바라지 않는 상황이죠.

기술 부채는 일관성이 없고 잘못된 개발 관행이 앱 개발에 침투했을 때 점점 더 불어납니다. 
모두가 알다시피 한번 코드에 기술 부채가 발생하면 되돌릴 수 있는 가능성은 낮습니다. 
또한 기술 부채로 인해 앱 구동이 불안정해지기도 합니다. 
따라서, 처음부터 모범 사례를 보며 앞으로 아름다운 코드 베이스를 구축할 수 있도록 하는 것이 좋습니다!

바로 시작해볼까요?

# 개인정보 데이터 (Data Privacy)

요즘은 개인 정보 보호에 대해 많은 사람들이 관심을 갖습니다. 
GDPR, CCPA, LGPD, PIPL 등 개인 정보 보호 규정등을 보면 알 수 있죠. 
누군가 계약에서 데이터를 훔치거나 블록체인 자체에 PII(개인 식별 가능 정보)를 추가했다고 상상해 보세요.
너무 끔찍합니다.

따라서, 우리는 개인정보를 다룰 때 특히 주의를 기울여야합니다. 
Private 변수도 예외는 아닙니다. 
우리는 개인정보를 다루는 부분은 철저하게 암호화하거나 체인 외 다른 곳에 저장해야합니다.

# 반복문 페이지 설정하기 (Loop Pagination)

계약 내부의 함수를 실행하기 위해서는 계산 복잡도에 따라 일정량의 가스를 사용해야합니다. 
안타깝게도 이더리움 블록당 가스 양에는 임계값이 있습니다. 
더 정확히 말하자면, 단일 블록 내에서 발생하는 모든 거래에 사용되는 가스의 합이 이더리움 지정 가스 한도보다 적어야합니다.

코드 최적화가 되어있지 않거나 대규모 배열을 처리하거나 반복문을 실행하는 도중에 
트랜잭션이 블록 가스 제한을 초과했음을 의미하는 DoS(서비스 거부: Denial-of-Service) 오류가 발생할 가능성이 높습니다.

페이지네이션을 해두면, 여러 블록에 걸쳐 거래가 반복되거나 분산 처리되는 동안 오류가 발생하는 일을 막을 수 있습니다.

```solidity
contract DummyDatastore {
   uint256[] private _items;
    
    function getItemAtIndex(uint256 index) public view returns (uint256) {
        return _items[index];
    }

    function countItems() public view returns (uint256) {
        return _items.length;
    }

    ...
}

contract DummyDatastorePaginatedWrapper { 
    function getItemAtIndexPaginated(
        DummyDatastore store, 
        uint256 startIndex, 
        uint256 offset) public view returns(uint256[] memory) { 
        
        uint256[] memory result; 
        require(startIndex + offset 
        <= store.countItems(), "Read index out of bounds"); 

        for (uint256 i = 0; i < offset; ++i) { 
            result[i] = store.getItemAtIndex(i + startIndex); 
        } 
        
        return result; 
    } 
}
```
# 메소드 시그니처 (Method Signature)

계약 함수에 매개 변수(parameter)를 설정할 때, 매개 변수의 사용 목적을 분명히 해야합니다. 
**require**문을 사용하여, 해당 매개변수가 필요할 경우에만 사용하도록 유효성 확인을 하세요.

# 써드파티 사용하기 (Third-Party Calls)

모든 계약이 안전한 것은 아닙니다. 
개발자들은 계약이 안전하게 진행되고, 각종 위협에 대응할 수 있도록 만들 책임이 있습니다. 
종종 타사 화폐를 사용하면서, 유효성 검사를 하지 않고 곧바로 결과를 직접 사용하려는 경우에 오류가 발생하기도 합니다. 
다시 말해, 외부 통화로 반환된 모든 값들은 계약에 사용하기 전에 유효성을 반드시 확인해야합니다. 
잘 처리해두지 않으면 Dos 상태에 빠집니다.

```solidity
(bool success, ) = dummyAddress.call.value(55)("");
if(!success) {
    // Add your logic here
}
```

# 이벤트 (Events)

계약에서는 이벤트를 적극적으로 사용하여 데이터 관련 상태 변화를 기록해야 합니다. 
이벤트를 통해 기록해두면, 우리는 계약의 상태 변화를 모니터링할 수 있고, 상태 변화 추적또한 쉽게 할 수 있습니다. 

# 안전하게 타입 변경하기 (Safe Casting)

큰 단위의 데이터를 작은 단위의 타입으로 변경(type casting)하다보면, 뜻하지 않게 데이터가 변형되기도 합니다. 
계산 중에 데이터가 변형된다면, 그 계약은 신뢰할 수 없겠죠.

이런 엣지 케이스를 위해, Solidity 라이브러리인 오픈 제플린을 사용하는 것이 좋습니다.

```solidity
import "@openzeppelin/contracts/utils/SafeCast.sol";

contract DummyContract {
    function safeCastToUint64(uint256 input) public pure returns (uint64) {
        return SafeCast.toUint64(input);
    }
}
```

# Checks-Effects-Interactions 패턴

체크-효과-상호작용(Checks-Effects-Interactions)은 계약의 예상 실행을 보장하는 디자인 패턴입니다. 
이 패턴을 사용하면, 계약 상태 업데이트가 완료되기 전까지 써드파티가 실행되거나 외부에서 호출하는 것을 막을 수 있습니다. 

```solidity
contract DummyContract {
    …

    function withdrawAmount() public {
        uint amountToWithdraw = balances[msg.sender];
        balances[msg.sender] = 0;

        (bool success, ) = msg.sender.call.value(amountToWithdraw)("");

        // The user's balance is already 0, 
        // so future invocations won't withdraw anything
        require(success);
    }
    …
}
```

# 재진입 취약성 확인하기 (Watch Out for Reentrancy Vulnerability)

Uniswap V1에서 발견된 유명한 재진입 취약성(Reentrancy Vulnerability)에 대해 아시나요? 
간단히 말하자면, 주로 ERC777의 전/후 전송 훅(transfer hooks) 때문에 발생하는 취약성입니다.

오픈 제플린의 [RentryGuard][rentry_guard]를 사용하여 계약의 재진입 취약성을 방지할 수 있습니다. 
이 취약성의 작동 방식에 대한 자세한 내용은 [Assignment][assignment] 섹션을 참조하십시오.

# `tx.origin` 지양하기 (Say No to tx.origin)

Solidity 에는 글로벌 변수 `tx.origin`이 있습니다. 
이 변수는 거래를 보낸 계정의 주소를 유지하는 역할을 합니다. 
승인을 위해 이 변수에 의존하는 계약은 불안정하며 피싱 공격에 쉽게 타격을 받을 수 있습니다.

이에 대한 대안으로는 `msg.sender`가 있습니다. 
계정 주소를 유지하는 대신 기능이나 거래를 호출하거나 시작한 주소를 저장합니다.

```solidity
contract DummyContract {
    address owner;

    function DummyContract() public {
        owner = msg.sender;
    }

    function sendTo(address receiver, uint amount) public {
        require(msg.sender == owner);
        receiver.transfer(amount);
    }
}
```

# 공용 기능에 대한 액세스 제어

계약의 모든 함수를 공개해야한다는 룰은 없습니다!
일단은 외부 접근을 최대한 제한하는 것을 권장합니다.
함수 작성도 신중해야겠지만, 함수의 접근 제한자를 결정할 때 특히 신경을 써야합니다.
함수 접근을 제한하게 되면, 외부 사용자가 공개 함수에 대해 파악하기 어렵습니다.
이에 따라, 모든 함수를 공개하는 것보다는 공개 함수에 대한 정보를 상세하게 설명하는 것이 좋습니다.

# 선호 인터페이스 유형

만약 당신이 당신의 계약 안에서 다른 계약을 부르고 싶다면, 당신은 아마도 그 계약에 대한 정보를 어딘가에 보관해야 할 것입니다. 
개발자들은 보통 자신의 계약 내부에 타 계약의 주소를 저장합니다. 이것은 잘못된 관행입니다. 
왜냐하면, 이러한 방식은 컴파일-타임 체크를 사용하지 않기 때문입니다.

컴파일-타임에도 확인할 수 있도록 인터페이스 유형 또는 계약 유형 자체를 사용하는 것이 좋습니다.

```solidity
contract DummyContract {
    OtherContract otherContract;

    function callOtherContract() public view returns (bool){
        bool answer = otherContract.check(69,true);
        return answer;
    }
}

contract OtherContract {
    function check(uint32 x, bool y) public pure returns (bool r) {
        r = x > 32 || y;
    }
}
```

# 불필요한 코드 제거(Get Rid of Dead Code)

불필요한 코드는 혼란을 만들어냅니다. 
코드내에 상주하면서, 아무런 도움이 되지 않는데도 유지 보수가 필요합니다. 
현재 Solidity 컴파일러는 이 불필요한 코드에 대해 오류나 경고를 보내지 않지만 개발자로서 주의해야 합니다.

또한 코드가 의도한 대로 작동하는지 확인하기 위해 단위 테스트를 추가하세요.
보통 올바른 코드를 작성하고 있다고 생각하지만, 불필요한 코드가 있을 수 있습니다.

다음 코드가 계약의 함수 중 하나에 추가되었다고 가정해봅시다.

```solidity
msg.sender.call.value(address(this).balance);
```

이제 육안으로만 확인할 경우, 위 코드가 잘못됐다는 것을 확인하지 않고 넘어가기 쉽습니다. 
위에 있는 코드 한 줄은 기본적으로 함수 참조이기 때문에 불필요한 코드입니다. 
함수가 호출되지 않습니다. 올바른 코드는 다음과 같습니다:

```solidity
msg.sender.call.value(address(this).balance)("");
```

적절한 테스트가 준비되어있지 않는 한 놓치기 쉬운 부분입니다.

[rentry_guard]: https://docs.openzeppelin.com/contracts/2.x/api/utils#ReentrancyGuard

[assignment]: https://github.com/ismailvohra/10-days-of-solidity/blob/main/day10.md#assignment