---
layout: post
title: "ERC-721 컨트랙 직접 생성하기: Solidity 10일 챌린지 - 07일차"
date:   2023-06-16 23:01:00 +0900
image: /assets/images/eunice-hong-opengraph.jpg
headerImage: false
tag:
- 10-days-of-solidity
category: book
author: eunice-hong
description: ERC-721 계약을 작성하는 방법과 NFT의 중요성에 대해 알아봅시다.
---

NFT는 이제 세계를 휩쓸고 있습니다!
NFT는 아이템과 서비스를 수익화함으로써 우리 사회의 모든 측면을 바꾸고 있죠.
크립토 대중을 통해 NFT가 창출한 엄청난 양의 매출은 디지털 예술과 수집품의 세계를 완전히 바꾸어 놓았습니다. 
디지털 토큰의 인기가 치솟으면서 다양한 토큰 표준이 존재로 진화했습니다. 
ERC-20 토큰에 대해서는 이미 논의한 바 있고, 이번 튜토리얼에서는 ERC-721 계약을 작성해보도록 하겠습니다. 
ERC-721 표준에 대한 자세한 내용과 스마트 계약을 작성하는 방법에 대해 알아보려면 집중해서 읽어주세요.!

## ERC-721 표준은 무엇입니까?

ERC-721은 NFT 마니아를 촉발시킨 전설적인 표준으로 여겨집니다.
이 표준은 다양한 획기적인 기능 덕분에 크게 성공할 수 있었습니다.
ERC-721를 통해 계정 간에 NFT 전송이 편리해졌고, 다른 통화와 교환도 할 수 있게 되었습니다. 
사유재산의 일부 소유권, 수천만 달러에 팔리는 디지털 아트 카피 원본의 블록체인 소유권, 
엘리트 클럽의 공공 회원이 된 독특한 아바타 등이 그 예시입니다.

일반적으로 ERC-20과 혼동되기 쉽습니다.
ERC-20는 계약을 만들 때, 고유 ID와 URI 또는 Uniform Resource Locator 뿐만 아니라 이름이나 기호와 같은 기본 메타데이터가 필요합니다. 
ERC-721은 다른 ERC와는 다르게 고유 ID로 인해 복제할 수 없다는 특징이 있습니다. 
ERC-721 표준은 스마트 계약 내에서 비-신탁(non-fiduciary) 토큰을 더 쉽게 보유하고 전송할 수 있도록 했습니다. 
ERC-721과 관련된 기본 사항을 몇 가지 복습했으므로, 이제 ERC-721과의 계약 구축으로 넘어갑시다.

## 오픈 제플린을 사용하여 간단한 ERC-721 계약 체결하기

다음 튜토리얼에서는 계약서 작성, 컴파일, 배포를 위한 기본 IDE로 [Remix][remix] 을 사용합니다.

- Remix로 이동하여 "HelloNFT.sol"이라는 이름의 새 파일을 만듭니다. 여기서 스마트 계약서를 작성합니다.
- 코드를 초기화하기 전에 기본 라이센스와 함께 pragma를 사용하여 계약에 대한 Solidity 컴파일러 버전을 정의해야 합니다:

```solidity
// SPDX-License-Identifier:UNLICENSED

pragma solidity ^0.8.9;
```

- Solidity 컴파일러 버전을 선언한 후, 우리는 공식 소스에서 적절한 Open Zeppelin 라이브러리로 ERC721을 가져옵니다.
- ERC721 라이브러리에는 ERC-721 준수로 간주되는 계약에 필요한 기본 정의가 포함되어 있습니다.

```solidity
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
```

- 계약 이름은 Hellonft라고 짓고, 기호를 "HNFT" 로 정의하고 ERC-721 준수하는 것이라고 선언합니다.

```solidity
contract Hellonft is ERC721("HelloNft", "HNFT") {
```

- 그 다음 tokenId(각 토큰에 대한 고유 식별자)를 0으로 초기화합니다.

```solidity
uint tokenId;
```

- 주소를 키로 하고 `tokenMetaData`를 값으로 하는 매핑을 만듭니다.

```solidity
mapping(address=>tokenMetaData[]) public ownershipRecord;
```
- `struct tokenMetaData`는 각 토큰의 `tokenId`, `timestamp`, `tokenURI`를 한 곳에 모아 
  `ownershipRecord`의 주소와 완전히 연결되도록 설계되었습니다.

```solidity
struct tokenMetaData {
	uint tokenId;
	uint timeStamp;
	string tokenURI;
}
```
We mint the sample image in the .png format.
- 이제 우리는 우리의 mintToken 함수를 작성해봅시다. 민팅(Minting)을 통해 다양한 자산을 토큰화할 수 있습니다. 
- 다음 코드로 샘플 이미지를 `.png` 형식으로 민팅합니다.

```solidity
function mintToken(address recipient, string memory url) public {
```

- 이제 함수 `_safeMint` 를 사용하여 ERC721 계약에서 가져온 민팅 프로세스를 시작하고 
  `tokenMetaData`를 수신기에 푸시하여 자체 `ownershipRecord`를 유지합니다. 
  우리는 계약에서 토큰을 민팅하기 위해 URI와 동일한 .png 파일을 하드 코딩하여 상황을 단순화했습니다.

```solidity
_safeMint(recipient, tokenId);
        ownershipRecord[recipient].push(tokenMetaData(tokenId, block.timestamp, url));
        tokenId = tokenId + 1;
    }
}
```

- 이제 Remix를 포함한 모든 테스트 네트워크에서 이 계약을 구축할 수 있습니다. 전체 계약은 다음과 같습니다:

```solidity
// SPDX-License-Identifier:UNLICENSED

pragma solidity ^0.8.9;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

contract Hellonft is ERC721("HelloNft", "HNFT") {
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
 
- 함수 mintToken 를 통해 당신의 첫 NFT를 토큰화했습니다!

## 정리하며...

이 튜토리얼에서는 Open Zeppelin을 사용하여 NFT ERC-721 계약을 구축하는 과정을 살펴보았습니다. 
또한 가장 일반적인 토큰 표준의 기초와 본질, 인기 증가 이유에 대해 자세히 설명했습니다. 
이 글이 첫 번째 NFT 스마트 계약을 만드는 데 도움이 되었기를 바랍니다!

[remix]: https://remix.ethereum.org/