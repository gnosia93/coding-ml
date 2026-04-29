
### 큰 한영 혼용 텍스트 → NLP 데이터셋 의 표준 파이프라인 ###
```
1. 원본 텍스트 파일 수집    ← 이미 하신 일 (wiki 덤프 받음)
2. 정제 (cleaning)      ← wikiextractor 같은 도구로 처리. 선택적
3. 문장/문서 단위로 분할    ← 선택적
4. 토크나이저 학습         ← 아래 '코크나이저 학습' 샘플 코드로 처리
5. 토큰 → 정수 ID 변환
6. PyTorch Dataset
7. DataLoader
```

### 2. 정제 ###
꼭 해야 할 것
* HTML 태그 제거 — <p>, <div>, <br> 등
* HTML 엔티티 디코딩 — &nbsp; → , & → &
* URL 제거 (또는 <URL> 토큰으로 치환)
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


### 4. 토크나이저 학습 ###
```
import sentencepiece as spm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# === Step 1: 토크나이저 학습 (최초 1번) ===
spm.SentencePieceTrainer.train(
    input='wiki_clean.txt',
    model_prefix='ko_en_bpe',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9995,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

# === Step 2: 토크나이저 로드 ===
sp = spm.SentencePieceProcessor()
sp.load('ko_en_bpe.model')

# === Step 3: 전체 텍스트 → 토큰 파일 (최초 1번) ===
with open('wiki_clean.txt', encoding='utf-8') as f, \
     open('tokens.bin', 'wb') as out:
    for line in f:
        ids = sp.encode(line.strip(), out_type=int)
        np.array(ids, dtype=np.uint16).tofile(out)

# === Step 4: Dataset & DataLoader ===
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

dataset = LMDataset('tokens.bin', block_size=512)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# === Step 5: 학습 루프에서 사용 ===
for x, y in loader:
    # x, y: (batch, block_size)
    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    ...

```
