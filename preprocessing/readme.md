
## NLP 데이터셋 파이프라인 ##
```
1. 원본 텍스트 파일 수집    
2. 정제 (cleaning)      ← wikiextractor 같은 도구로 처리
3. 문장/문서 단위로 분할    ← 선택적
4. 토크나이저 학습         ← 아래 '코크나이저 학습' 샘플 코드로 처리
5. 토큰 → 정수 ID 변환
6. PyTorch Dataset
7. DataLoader
```
### 1. 텍스트 파일 수집 ###
```
# 한국어 덤프 다운로드 (약 1GB)
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2

# 영어 덤프 (20 GB 넘음, 시간 오래 걸림)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# 압축 해제는 따로 안 해도 됨 (wikiextractor 가 .bz2 직접 읽음)

# wikiextractor 로 plain text 추출
pip install wikiextractor
python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2 \
    --output extracted_ko \
    --bytes 100M \
    --processes 4
```

### 2. 정제 ###
꼭 해야 할 것
* HTML 태그 제거 — `<p>, <div>, <br>` 등
* HTML 엔티티 디코딩 — `&nbsp; → , & → &`
* URL 제거 `(또는 <URL> 토큰으로 치환)`
* 제어 문자, 이상한 공백 통일

해도 되고 안 해도 되는 것 (선택)
* 특수기호 제거
* 대소문자 통일
* 숫자 정규화 (123 → <NUM>)
* 이모지 처리
* 중복 문서 제거

절대 하지 말아야 할 것
* 조사/어미 제거 (한국어가 영어처럼 보이게 됨)
* 띄어쓰기 임의 변경
* 문장부호 전부 제거 (정보 손실)

```
import re
import html

def clean_text(text):
    # 1. HTML 엔티티 디코딩: &amp; → &, &nbsp; → 공백
    text = html.unescape(text)
    
    # 2. HTML 태그 제거: <p>, <div>, <br> 등
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. 위키 마크업 정리 (위키 원문이면)
    text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', text)   # [[link|text]] → text
    text = re.sub(r'\{\{[^}]+\}\}', '', text)                    # {{template}} 제거
    text = re.sub(r"'''([^']+)'''", r'\1', text)                 # '''bold''' → bold
    text = re.sub(r"''([^']+)''", r'\1', text)                   # ''italic'' → italic
    
    # 4. URL 제거 (또는 토큰화)
    text = re.sub(r'https?://\S+', '', text)
    
    # 5. 여러 공백/줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

HTML 많이 섞인 진짜 웹 크롤링 데이터면 정규식보다 BeautifulSoup 이 좋다.
```
from bs4 import BeautifulSoup

def clean_html(text):
    return BeautifulSoup(text, 'html.parser').get_text()
```

### 3. 문장/문서 단위로 분할 (선택적) ###
* 청크 기반 학습 (GPT 스타일, nanoGPT 등)

전체 텍스트를 하나의 긴 토큰 스트림으로 보고, block_size (예: 512 토큰) 씩 잘라서 학습.
```
전체 토큰: [23, 45, 67, 12, 89, 34, 56, 78, ...  수백만 개]
              └── 512개 ──┘└── 512개 ──┘└── 다음 ...
                 청크 0       청크 1       청크 2
```
한 청크 안에 여러 문장이 섞여 있어도 상관없다. 이 경우엔 3번 (문장 분할) 완전히 skip 해도 된다.

* 문장 단위 학습 (BERT 사전학습, Seq2Seq 등)
이 경우 문장 단위 분할이 필요하다.  
```
BERT 의 NSP (Next Sentence Prediction) — 두 문장이 연속인지 맞히기. 문장 경계를 알아야 함
Seq2Seq 번역 — 문장 하나가 입력, 문장 하나가 출력. 문장 경계 필수
문장 분류, 감성 분석 — 문장이 샘플 단위
```

### 4. 토크나이저 학습 ~ DataLoader ###
```
import os
import sentencepiece as spm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

RAW_TEXT = 'wiki_clean.txt'
SP_MODEL = 'ko_en_bpe.model'
TOKENS_BIN = 'tokens.bin'

# === 1단계: 토크나이저 (없으면 학습) ===
if not os.path.exists(SP_MODEL):
    print('Training tokenizer...')
    spm.SentencePieceTrainer.train(
        input=RAW_TEXT,
        model_prefix='ko_en_bpe',
        vocab_size=32000,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)

# === 2단계: 바이너리 토큰 파일 (없으면 생성) ===
if not os.path.exists(TOKENS_BIN):
    print('Encoding text to tokens.bin...')
    with open(RAW_TEXT, encoding='utf-8') as f, open(TOKENS_BIN, 'wb') as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            np.array(ids, dtype=np.uint16).tofile(out)

# === 3단계: Dataset & DataLoader (매번) ===
class LMDataset(Dataset):
    def __init__(self, tokens_path, block_size=512):
        self.data = np.memmap(tokens_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start:start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

dataset = LMDataset(TOKENS_BIN, block_size=512)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# 이제 학습 루프
for x, y in loader:
    print(x.shape, y.shape)   # (32, 512), (32, 512)
    break
```

## 부연설명 ##

### 대규모 언어 모델 사전학습 (nanoGPT, 대형 LLM) ###
대규모 언어모델 사전학습의 경우 코퍼스를 미리 토크나이즈하여 데이터셋을 만들어 놓는다.  
왜냐하면:
* 토큰 수가 엄청나게 많음 (수십억~수조 토큰). 매번 토크나이징하면 너무 느림
* 시퀀스가 고정 길이 청크 — 문서 경계 무시하고 토큰 스트림으로 잘라 씀. 이 방식은 int16 배열이 최고 효율
* Disk I/O 가 병목 — memmap 이 가장 빠른 읽기 방법
* 동일 데이터를 여러 epoch 학습 — 토크나이징 비용 한 번만 지불
* 예: nanoGPT, GPT-2 학습, Llama 사전학습 등.



### (즉석 또는 HF Datasets) 가 유리한 경우 ###
파인튜닝, 지도학습, 가변 길이 태스크에 해당하는 것으로 배치단위로 토크나이징 한다.
이 경우 다음과 같으 경우에 해당한다.

* 데이터 크기가 작음 (~수십만 샘플) — 토크나이징이 병목 아님
* 샘플마다 길이가 다름 — padding/truncation 을 배치 구성 시 동적으로
* 메타데이터 유지 필요 — label, metadata 등이 같이 따라다녀야 함
* 실험 다양성 — tokenizer 바꿔가며 비교 실험 쉬움 (원본 텍스트 보존)
* 예: 감성 분류 fine-tuning, Q&A, 번역 파인튜닝, RLHF.


