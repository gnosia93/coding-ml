
### 큰 한영 혼용 텍스트 → NLP 데이터셋 의 표준 파이프라인 ###
```
1. 줄 단위 스트리밍으로 파일 읽기
2. 가벼운 정제
3. SentencePiece BPE 로 토크나이저 직접 학습 (한영 자동 처리)
4. 전체 텍스트를 토큰 ID 로 변환 후 바이너리 저장 (numpy memmap)
5. PyTorch Dataset 에서 memmap 으로 lazy 로드
6. DataLoader 로 학습 루프에 공급
```
