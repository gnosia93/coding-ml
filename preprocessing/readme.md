
### 큰 한영 혼용 텍스트 → NLP 데이터셋 의 표준 파이프라인 ###
```
1. 원본 텍스트 파일 수집    
2. 정제 (cleaning)      ← wikiextractor 같은 도구로 처리. 선택적
3. 문장/문서 단위로 분할    ← 선택적
4. 토크나이저 학습         ← 아래 '코크나이저 학습' 샘플 코드로 처리
5. 토큰 → 정수 ID 변환
6. PyTorch Dataset
7. DataLoader
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

### 4. 토크나이저 학습 ###
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
