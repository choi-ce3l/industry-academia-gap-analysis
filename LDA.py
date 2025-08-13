import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim import corpora, models
from gensim.models import CoherenceModel

# 1) 텍스트 전처리
def preprocess_text(text):
    if pd.isnull(text):
        return []
    if isinstance(text, list):
        text = ' '.join(text)
    if isinstance(text, str) and ('<' in text and '>' in text):
        text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\\\(.*?\\\)", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    pos_tags = pos_tag(tokens)
    allowed = {'NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'}
    tokens = [w for w,p in pos_tags if p in allowed]

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]


# 2) Dictionary & BoW Corpus 생성
def build_dictionary_corpus(docs, no_below=5, no_above=0.9):
    """
    docs: 리스트 of tokenized documents
    no_below: 최소 출현 문서 수
    no_above: 최대 출현 비율
    """
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, corpus_bow


# 3) TF–IDF 모델 및 변환된 Corpus
def build_tfidf_corpus(corpus_bow):
    """
    corpus_bow: BoW corpus
    returns: (tfidf_model, corpus_tfidf)
    """
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    return tfidf, corpus_tfidf


# 4) LDA 모델 학습
def train_lda_model(corpus, dictionary, num_topics=10, passes=10, random_state=42):
    """
    corpus: TF–IDF 변환된 코퍼스 또는 BoW 코퍼스
    dictionary: gensim Dictionary
    """
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )


# ——————————————
# 전체 파이프라인 호출 예시
# ——————————————
def run_lda_pipeline(
    texts,            # 원문 리스트(단일 year 또는 전체)
    num_topics=10,
    no_below=5,
    no_above=0.9,
    passes=10,
    random_state=42
):
    # 1) 전처리
    docs = [preprocess_text(t) for t in texts]

    # 2) Dictionary & BoW
    dictionary, bow_corpus = build_dictionary_corpus(docs, no_below, no_above)

    # 3) TF–IDF
    tfidf_model, tfidf_corpus = build_tfidf_corpus(bow_corpus)

    # 4) LDA (TF–IDF 기반)
    lda = train_lda_model(tfidf_corpus, dictionary,
                          num_topics=num_topics,
                          passes=passes,
                          random_state=random_state)

    # 토픽 출력
    for idx, topic in lda.print_topics():
        print(f"Topic {idx}: {topic}")
    return lda, dictionary, tfidf_model

# 5) 단일 LDA 모델 Coherence 계산
def compute_coherence_score(
    lda_model,
    texts,         # list of tokenized docs, ex) output of preprocess_text
    dictionary,    # gensim.corpora.Dictionary
    corpus_bow,    # BoW 코퍼스, list of list of (token_id, count)
    coherence='c_v'
):
    """
    lda_model : 학습된 gensim LdaModel
    texts     : 전처리 후 토큰 리스트들의 리스트
    dictionary: gensim Dictionary
    corpus_bow: 원본 BoW 코퍼스 (U_Mass 계산용)
    coherence : 'c_v', 'u_mass', 'c_uci', 'c_npmi' 중 선택
    """
    cm = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        corpus=corpus_bow,
        coherence=coherence
    )
    score = cm.get_coherence()
    print(f"Coherence ({coherence}) = {score:.4f}")
    return score


# 6) 여러 토픽 개수별 Coherence 벡터 계산
def compute_coherence_values(
    texts,
    dictionary,
    corpus_bow,
    start=2,
    limit=20,
    step=2,
    passes=10,
    random_state=42,
    coherence='c_v'
):
    """
    texts     : 전처리 후 토큰 리스트들의 리스트
    dictionary: gensim Dictionary
    corpus_bow: BoW 코퍼스
    start     : 토픽 수 탐색 시작
    limit     : 토픽 수 탐색 끝
    step      : 증분
    passes    : LDA 학습 시 passes
    """
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit+1, step):
        lda = train_lda_model(
            corpus=corpus_tfidf if False else corpus_bow,
            dictionary=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=random_state
        )
        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            corpus=corpus_bow,
            coherence=coherence
        )
        cv = cm.get_coherence()
        coherence_values.append(cv)
        model_list.append(lda)
        print(f"#Topics={num_topics}  Coherence({coherence})={cv:.4f}")

    return model_list, coherence_values