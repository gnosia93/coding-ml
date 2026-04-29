
### 큰 한영 혼용 텍스트 → NLP 데이터셋 의 표준 파이프라인 ###
```
1. 줄 단위 스트리밍으로 파일 읽기
2. 가벼운 정제
3. SentencePiece BPE 로 토크나이저 직접 학습 (한영 자동 처리)
4. 전체 텍스트를 토큰 ID 로 변환 후 바이너리 저장 (numpy memmap)
5. PyTorch Dataset 에서 memmap 으로 lazy 로드
6. DataLoader 로 학습 루프에 공급
```

### 샘플코드 ###
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
