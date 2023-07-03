---
layout: post
title: "Hardhat으로 단위 테스트 작성하기: Solidity 10일 챌린지 - 09일차"
date:   2023-06-18 23:01:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- 10-days-of-solidity
category: book
author: eunice-hong
description: 이더리움과 관련 용어 소개
---
2015년 5월, 에어버스 A400M은 스페인의 도시인 세비야 근처에서 추락했습니다. 
조사 결과, 소프트웨어의 취약점으로 인해 필수 엔진 데이터가 삭제되었고, 비행 도중 엔진이 고장난 것으로 밝혀졌습니다. 
이 에어버스 A400M 화물기에 탑승했던 승무원 6명 중 4명이 사망했습니다.

우리는 개발자로서 약간의 소홀함으로도 심각한 사고와 
사람의 죽음을 초래할 수 있는 분야에 있다는 것을 기억해야합니다.
우리가 개발하는 소프트웨어는 철저한 테스트를 거쳐야 하며, 
부족하거나 과도하게 실행되지 않고 예상대로 정확히 작동하는지 확인해야 합니다.

일반적으로 비즈니스 환경이 확장 가능한 소프트웨어를 요구하기 때문에
사소한 변경 후에도 모든 기능이 예상대로 작동하도록 보장해야합니다. 
따라서 단위 테스트는 반드시 필요합니다. 
마찬가지로 블록체인 생태계에서 단위 테스트는 스마트 계약의 무결성을 확립하는 데도 중요한 역할을 합니다.

말이 필요없겠죠?
이제 Hardhat 생태계에서 Solidity 계약 단위 테스트 작성 방법에 대해 자세히 알아보겠습니다.

# 하드햇이 뭐죠?

개정 목적을 위해, Hardhat은 전문가들을 위해 특별히 구축된 빠르고 유연하며 확장 가능한 이더리움 개발 환경입니다. 
다른 잘 알려진 프레임워크와 마찬가지로 
개발자들이 계약을 신속하게 구축, 확장 및 테스트할 수 있도록 최소 개발 설정을 갖추고 있습니다.

하드햇은 개발자들에게 실제 블록체인의 실제 경험을 제공하고 
개발 주기를 빠르게 혁신할 수 있는 자신감을 주는 다양한 지원 운영과 함께 지역 개발 체인이 내장되어 있습니다.

어제 Hardhat 설정에 대해 다루었으며, 오늘은 7일차부터 ERC-721 계약에 대해 Hardhat을 통해 단위 테스트를 작성하는 데 중점을 둘 예정입니다.

# ERC-721 스마트 컨트랙트

다음은 [7일차 ERC-721의 전체 계약 내용][day-7-contract]입니다:

```solidity
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

contract HelloNFT is ERC721("HelloNFT", "HNFT") {
    uint tokenId;
    mapping(address=>tokenMetaData[]) public ownershipRecord;

    struct tokenMetaData {
        uint tokenId;
        uint timeStamp;
        string tokenURI;
    }

    function mintToken(address recipient, string memory url) public {
        _safeMint(recipient, tokenId);
        ownershipRecord[recipient].push(tokenMetaData(tokenId, block.timestamp, url));
        tokenId = tokenId + 1;
    }
}
```

이 계약서는 단지 동전의 한 면일 뿐입니다. 함수 `mintToken` 가 예상대로 작동한다고 어떻게 보증하죠?
맞아요, 당신이 정확히 맞췄어요!

다음 단계로 넘어가 단위 테스트를 작성해 보겠습니다.

# Hardhat 계약을 위한 쓰기 단위 테스트

우리가 만든 HelloNFT 계약은 주어진 URL에 따라 토큰을 민팅하기 위한 함수를 공개했습니다.
단위 테스트를 작성하기 전에 함수 `mintToken` 를 테스트 하기 위한 시나리오를 생각해봅시다.
아래는 함수 동작을 보장하기 위해 생각해보아야하는 지점들입니다. 필요한 경우의 수를 자유롭게 추가해보세요.

1. 수취인의 주소가 유효해야 합니다.
2. 토큰 식별자는 민트가 성공할 때마다 증가해야 합니다.
3. 민트 토큰 메타데이터에는 우리가 제공한 올바른 URL 리소스(이번 경우에는 `tokenURI`)가 포함되어야 합니다.
4. URL 리소스는 수신자 주소에만 할당해야 합니다.

In a project that is created using Hardhat CLI, there is a separate directory that entails all the unit tests. This directory is named test in the Hardhat boilerplate code structure and resides at the project's root.

Let's create a new JavaScript file in the **test** folder and call it **hello-nft-test.js**.
We will write tests for some of the example scenarios mentioned above.


Hardhat CLI를 사용하여 생성된 프로젝트에는 모든 장치 테스트를 수반하는 별도의 디렉터리가 있습니다. 
Hardhat 보일러플레이트에서 `/test` 라는 이름으로 사용하고, 프로젝트 루트에 위치합니다.

**test** 폴더에 새 JavaScript 파일을 **hello-nft-test.js**를 만들어봅시다. 
위에서 언급한 몇 가지 예제 시나리오에 대한 테스트를 작성해보겠습니다.

!https://user-images.githubusercontent.com/34173058/169704158-f58d5ee7-c9c6-44f2-b275-fe17db799da7.png

hello-nft-test.js에서 테스트 프로세스 전체에 필요한 모듈을 가져오는 것부터 시작해봅시다.

```jsx
const { expect } = require("chai");
const { ethers } = require("hardhat");
```

위에 언급했던 시나리오들의 다양한 케이스들을 테스트 해봅시다.

이제 위에서 언급한 각 시나리오에 대해 몇 가지 긍정적인 사례와 부정적인 사례를 테스트해 보겠습니다.

### 1. 수취인의 주소가 유효해야 합니다

이 시나리오의 경우, 수취인 주소 유효성에 따라 함수 `mintToken`이 제대로 작동하는지 확인해야합니다.

**참고: 일부 데모 계정/주소는 개발 목적으로 제공됩니다. 데모 계정 목록을 보려면 `npx hardhat accounts`을 실행하십시오.**

동일한 hello-nft-test.js 파일에서 다음과 같이 테스트 스위트를 등록합니다:

```solidity
describe("HelloNFT Mint", function () {});
```

테스트 스위트를 등록했다면, 주소의 유효성을 확인하는 아래 두 가지 테스트를 추가합니다:

```jsx
`it("Should not throw BAD ADDRESS CHECKSUM error if correct recipient address is provided", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       try {
           await helloNFT.mintToken("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "https://www.google.com");
          
           // If reached here, it means that code did not throw any error so we can simply pass the test
           expect(true).to.equal(true);
       } catch (err) {
           // If any error is thrown by code, in order for this test to pass,
           // we must check that error is not "bad error checksum"
           expect(
               'bad address checksum (argument="address", value="0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", code=INVALID_ARGUMENT, version=address/5.6.0)'
               ).to.equal(err.message);
           }
       }
   );`

`it("Should throw BAD ADDRESS CHECKSUM error if correct recipient address is provided", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       try {
           await helloNFT.mintToken("0x0000a0000a0aa00000aaa0aaa00a0a0a0a0a0000", "https://www.google.com");
          
           // If reached here, it means that code did not throw any error so we can simply pass the test
           expect(true).to.equal(true);
       } catch (err) {
           // If any error is thrown by code, in order for this test to pass,
           // we must check that error is not "bad error checksum"
           expect(
               'bad address checksum (argument="address", value="0x0000a0000a0aa00000aaa0aaa00a0a0a0a0a0000", code=INVALID_ARGUMENT, version=address/5.6.0)'
               ).to.equal(err.message);
           }
       }
   );`
```

`npx hardhat test`를 실행하여 테스트가 실행 가능한지 확인합니다.

```bash
HelloNFT Mint
    ✔ Should not throw BAD ADDRESS CHECKSUM error if correct recipient address is provided (1044ms)
    ✔ Should throw BAD ADDRESS CHECKSUM error if correct recipient address is provided (96ms)

  2 passing (1s)
```

### 2. 토큰 식별자는 민트가 성공할 때마다 증가해야 합니다.

이 시나리오의 장치 테스트는 각 성공적인 민트 작업 후 tokenId가 정확하게 증가하는지 확인해야 합니다.

테스트는 다음과 같습니다:

```jsx
it("Should check if tokenId is correctly increment after minting", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       await helloNFT.mintToken("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "https://www.google.com");
       const ownershipRecord1 = await helloNFT.ownershipRecord("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", 0);
       expect(ownershipRecord1[0]).to.equal(0)
      
       await helloNFT.mintToken("0xdD2FD4581271e230360230F9337D5c0430Bf44C0", "https://www.google.com");
       const ownershipRecord2 = await helloNFT.ownershipRecord("0xdD2FD4581271e230360230F9337D5c0430Bf44C0", 0);
       expect(ownershipRecord2[0]).to.equal(1)
});
```

다시 한번, `npx hardhat test`을 실행하면:

```bash
HelloNFT Mint
    ✔ Should check if tokenId is correctly increment after minting (1163ms)

  1 passing (1s)
```
이 글에서는 Hardhat 환경에서 Solidity 스마트 계약에 대한 단위 테스트를 작성하는 방법에 대해 살펴봤습니다. 
Hardhat을 사용하면 개발 체인에서 계약을 쉽게 구축, 컴파일 및 테스트할 수 있으며, 배송 및 배포 과정에서 자신감을 얻을 수 있습니다.

전체적으로 테스트 파일 hello-nft-test.js는 다음과 같습니다.

```solidity
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("HelloNFT Mint", function () {
   it("Should not throw BAD ADDRESS CHECKSUM error if correct recipient address is provided", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       try {
           await helloNFT.mintToken("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "https://www.google.com");
          
           // If reached here, it means that code did not throw any error so we can simply pass the test
           expect(true).to.equal(true);
       } catch (err) {
           // If any error is thrown by code, in order for this test to pass,
           // we must check that error is not "bad error checksum"
           expect(
               'bad address checksum (argument="address", value="0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", code=INVALID_ARGUMENT, version=address/5.6.0)'
               ).to.equal(err.message);
           }
       }
   );

   it("Should throw BAD ADDRESS CHECKSUM error if correct recipient address is provided", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       try {
           await helloNFT.mintToken("0x0000a0000a0aa00000aaa0aaa00a0a0a0a0a0000", "https://www.google.com");
          
           // If reached here, it means that code did not throw any error so we can simply pass the test
           expect(true).to.equal(true);
       } catch (err) {
           // If any error is thrown by code, in order for this test to pass,
           // we must check that error is not "bad error checksum"
           expect(
               'bad address checksum (argument="address", value="0x0000a0000a0aa00000aaa0aaa00a0a0a0a0a0000", code=INVALID_ARGUMENT, version=address/5.6.0)'
               ).to.equal(err.message);
           }
       }
   );

   it("Should check if tokenId is correctly increment after minting", async function () {
       const HelloNFT = await ethers.getContractFactory("HelloNFT");
       const helloNFT = await HelloNFT.deploy();
       await helloNFT.deployed();
      
       await helloNFT.mintToken("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "https://www.google.com");
       const ownershipRecord1 = await helloNFT.ownershipRecord("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", 0);
       expect(ownershipRecord1[0]).to.equal(0)
      
       await helloNFT.mintToken("0xdD2FD4581271e230360230F9337D5c0430Bf44C0", "https://www.google.com");
       const ownershipRecord2 = await helloNFT.ownershipRecord("0xdD2FD4581271e230360230F9337D5c0430Bf44C0", 0);
       expect(ownershipRecord2[0]).to.equal(1)
   });
   }
);
```

[day-7-contract]: {{ site.baseurl_root }}/Creating-your-own-Simple-ERC-721-Contract