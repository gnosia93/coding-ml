* 원본 텍스트 파일 수집
* 정제 (cleaning) — 위키 마크업 제거, 이상한 문자 거르기
* 문장/문서 단위로 분할
* 토크나이저 학습 (sentencepiece 추천)
* 토큰 → 정수 ID 변환 (tokenization)
* PyTorch Dataset 클래스로 감싸기
* DataLoader
