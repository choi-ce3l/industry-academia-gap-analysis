# README

## 개요
이 저장소는 CNN 기사 데이터를 수집‧전처리하고 LDA 기반 토픽 모델링을 통해 산업계와 학계 간의 연구 격차를 분석하는 프로젝트입니다.

## 프로젝트 구조
| 파일/디렉터리 | 설명 |
|---------------|------|
| `CNN_crawler.py` | CNN 내부 API를 이용해 기사 메타데이터와 본문을 수집하는 크롤러 스크립트 |
| `LDA.py` | 텍스트 전처리와 LDA 토픽 모델 학습, Coherence 계산 파이프라인 |
| `Preprocessing.ipynb` | 기사 텍스트 전처리 예제를 담은 노트북 |
| `analysis_ver2*.ipynb` | 전처리된 데이터를 활용해 산업계/학계 분석을 수행한 버전 2 노트북 |
| `analysis_ver3*.ipynb` | 개선된 분석 절차가 반영된 버전 3 노트북 |

## 요구 사항
- Python 3.8 이상  
- 주요 패키지: `pandas`, `requests`, `beautifulsoup4`, `tqdm`, `nltk`, `gensim` 등  
  (필요 시 `pip install pandas requests beautifulsoup4 tqdm nltk gensim` 형태로 개별 설치)

## 환경 설정
1. 가상환경 생성  
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. 패키지 설치  
   ```bash
   pip install -r requirements.txt  # 존재하지 않을 경우 필요한 패키지를 개별 설치
   ```
3. NLTK 리소스 다운로드  
   ```python
   import nltk
   for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
       nltk.download(pkg)
   ```

## 사용 방법
### 1. CNN 기사 수집
```bash
python CNN_crawler.py -q "검색어" -s 20 -p 3 -o 결과.csv
```
- `-q`: 검색어 (기본값: `AI`)
- `-s`: 페이지당 기사 수
- `-p`: 스크래핑할 페이지 수
- `-o`: 저장할 CSV 파일명  
각 요청 사이 딜레이는 `--delay-min`, `--delay-max` 옵션으로 조절할 수 있습니다.

### 2. LDA 토픽 모델링
```bash
python LDA.py
```
- `preprocess_text`, `build_dictionary_corpus`, `train_lda_model` 등 모듈화된 함수 제공
- 파이프라인을 직접 호출하거나 노트북에서 모듈을 가져와 사용할 수 있습니다.

### 3. 노트북 실행
각 `*.ipynb` 파일을 Jupyter Notebook 또는 JupyterLab에서 열고 셀을 순서대로 실행합니다.

## 기타
- 현재 저장소에 라이선스 정보가 명시되어 있지 않으므로, 연구 목적 사용을 권장합니다.
- 개선 사항이나 버그 리포트는 이슈 트래커를 통해 제안해 주세요.

