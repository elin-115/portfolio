"""코랩 제출용 노트북(기본 난이도) 생성기."""
import json
import sys

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# [개별미션 4차] 헬스케어 기술보증 데이터 분석
""")

# ───────── 준비 ─────────
md("## 0. 준비")
code("""
# 코랩에 없는 라이브러리 설치
!pip install konlpy gensim
""")
code("""
# 데이터 파일 업로드 (06_Tech_Credit_Data.xlsx 선택)
from google.colab import files
uploaded = files.upload()
""")
code("""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('06_Tech_Credit_Data.xlsx', sheet_name='기술데이터')
print(df.shape)
df.head()
""")
code("""
# 데이터 정보 확인
df.info()
""")
code("""
# 목표 변수: 진행상태 (여신승인 / 여신거부)
print(df['진행상태'].value_counts())
""")
code("""
# 형태소 분석기 불러오기
from konlpy.tag import Okt
okt = Okt()

# 예시 문장으로 명사 추출 확인
print(okt.nouns('실시간 대시보드를 적용한 감염관리 모니터링 시스템 개발'))
""")

# ───────── Q1 ─────────
md("""
---
## 1. 대출 승인 기업 중 자본금 상위 100개 기업의 기술명 최빈 단어
""")
code("""
# 자본금은 이미 숫자형(int64)이라 변환 없이 사용
print(df['자본금'].dtype)

# 대출 승인 기업 중 자본금 상위 100개 추출
top100_approved = (
    df[df['진행상태'] == '여신승인']
    .sort_values(by='자본금', ascending=False)
    .head(100)
)
top100_approved[['회사명', '기술명', '자본금']].head()
""")
code("""
# Okt로 명사 추출 (한 글자 단어는 의미가 적어서 2글자 이상만)
nouns_list = []
for text in top100_approved['기술명']:
    nouns = okt.nouns(text)
    nouns_list.extend([n for n in nouns if len(n) > 1])

# 단어 빈도수 (동점이 많아서 15개까지 확인)
word_counts = pd.Series(nouns_list).value_counts()
print('=== 전체 명사 빈도 상위 15 ===')
display(word_counts.head(15))
""")
code("""
# '개발, 기술, 제조' 같은 단어는 어느 기술명에나 붙어서 기술 내용을 알려주지 않음
# → 불용어로 빼야 하는데, 눈으로 고르지 않고 데이터로 찾기
#    기준: 주요 세부분류 20개 중 몇 개 분야에 등장하는 단어인지 세기
#          → 거의 모든 분야에 나오는 단어 = 분야와 상관없는 범용어

top20_fields = df['헬스케어 세부분류'].value_counts().head(20).index

word_field_count = {}
for field in top20_fields:
    field_words = set()            # 한 분야 안에서는 한 번만 세기
    for text in df[df['헬스케어 세부분류'] == field]['기술명']:
        field_words.update([n for n in okt.nouns(text) if len(n) > 1])
    for w in field_words:
        word_field_count[w] = word_field_count.get(w, 0) + 1

field_spread = pd.Series(word_field_count).sort_values(ascending=False)
field_spread.head(20)
""")
code("""
# 16~19개 분야에 나오는 단어와 9개 이하 단어 사이가 크게 벌어짐 → 15개 이상을 범용어로 정함
stopwords = list(field_spread[field_spread >= 15].index)

# '~을 이용한', '~기반' 처럼 문장을 잇는 표현에서 나온 단어도 추가
stopwords = stopwords + ['이용', '기반']
print('불용어:', stopwords)

word_counts_sw = pd.Series([n for n in nouns_list if n not in stopwords]).value_counts()
print('=== 불용어 제외 상위 15 ===')
display(word_counts_sw.head(15))
""")
code("""
# (참고) 수업에서 배운 BoW(CountVectorizer)로 세어도 같은 결과
from sklearn.feature_extraction.text import CountVectorizer

texts = [' '.join([n for n in okt.nouns(t) if len(n) > 1]) for t in top100_approved['기술명']]
cv = CountVectorizer(stop_words=stopwords)
bow = cv.fit_transform(texts)

bow_counts = pd.Series(bow.toarray().sum(axis=0), index=cv.get_feature_names_out())
bow_counts.sort_values(ascending=False).head(10)
""")
md("Q1_RESULT")

# ───────── Q2 ─────────
md("""
---
## 2. 업력 계산 및 대출 승인 여부에 따른 업력 차이 검정
""")
code("""
# 설립연도가 1900인 데이터는 실제 연도가 아니라 '모름'을 뜻하는 값 → 제외
print('설립연도 1900 개수:', (df['설립연도'] == 1900).sum())
df2 = df[df['설립연도'] != 1900].copy()

# 업력 = 현재 연도 - 설립연도
from datetime import datetime
this_year = datetime.now().year
df2['업력'] = this_year - df2['설립연도']

df2.groupby('진행상태')['업력'].describe()
""")
code("""
from scipy import stats

approve = df2[df2['진행상태'] == '여신승인']['업력']
reject = df2[df2['진행상태'] == '여신거부']['업력']

# 1) 정규성 검정 (p < 0.05 이면 정규분포가 아님)
print('정규성 - 승인:', stats.shapiro(approve).pvalue)
print('정규성 - 거부:', stats.shapiro(reject).pvalue)
""")
code("""
# 2) 정규분포가 아니므로 비모수 검정인 Mann-Whitney U 검정 (중앙값 비교)
#    귀무가설: 승인 기업과 거부 기업의 업력에 차이가 없다
print('승인 중앙값:', approve.median(), '/ 거부 중앙값:', reject.median())
print(stats.mannwhitneyu(approve, reject))

# (참고) 평균 비교 t-검정
print('승인 평균:', round(approve.mean(), 2), '/ 거부 평균:', round(reject.mean(), 2))
print(stats.ttest_ind(approve, reject))
""")
md("Q2_RESULT")

# ───────── Q3 ─────────
md("""
---
## 3. 평가등급과 대출 승인 여부 가설 검정 + 등급별 승인 비율
""")
code("""
# 평가등급 × 진행상태 교차표
grade_table = pd.crosstab(df['평가등급'], df['진행상태'])

# 등급별 승인 비율
grade_table['합계'] = grade_table['여신거부'] + grade_table['여신승인']
grade_table['승인비율'] = (grade_table['여신승인'] / grade_table['합계']).round(3)
grade_table.sort_values('합계', ascending=False)
""")
code("""
# 카이제곱 검정은 칸마다 개수가 충분해야 함 (기대빈도 5 이상)
# 'T3/A', 'AAA', 'CCC'처럼 10건도 안 되는 등급이 많아서, 개수가 많은 5개 등급만 사용
main_grades = ['AA', 'A', 'BBB', 'BB', 'B']
df3 = df[df['평가등급'].isin(main_grades)]
print('검정에 사용한 데이터:', len(df3), '/ 전체:', len(df))

table = pd.crosstab(df3['평가등급'], df3['진행상태'])
table = table.loc[main_grades]      # 등급 순서대로 정렬
table['승인비율'] = (table['여신승인'] / (table['여신거부'] + table['여신승인'])).round(3)
table
""")
code("""
# 카이제곱 독립성 검정
# 귀무가설: 평가등급과 대출 승인 여부는 관계가 없다(독립이다)
chi2, p, dof, expected = stats.chi2_contingency(table[['여신거부', '여신승인']])
print('카이제곱 통계량:', round(chi2, 3))
print('p-value:', round(p, 4))
print('자유도:', dof)
""")
md("Q3_RESULT")

# ───────── Q4 ─────────
md("""
---
## 4. '병원/의료 IT' 기술명 단어 표현(Word2Vec) → '감염'과 유사한 단어
""")
code("""
# 병원/의료 IT 기업만 추출
it_df = df[df['헬스케어 세부분류'] == '병원/의료 IT']
print('기업 수:', len(it_df))

# 기술명마다 명사 리스트 만들기 (Word2Vec 입력 형태: [['단어', '단어'], ['단어', ...], ...])
sentences = []
for text in it_df['기술명']:
    nouns = okt.nouns(text)
    sentences.append([n for n in nouns if len(n) > 1])

# 'EMR', 'Smart Hospital'처럼 영어로만 된 기술명은 Okt가 명사를 뽑지 못해 빈 리스트가 됨
print(sentences[:5])
print('빈 리스트 개수:', sum(1 for s in sentences if len(s) == 0))
""")
code("""
# '감염'이 들어간 기술명 확인
print('감염 등장 횟수:', sum(s.count('감염') for s in sentences))
it_df[it_df['기술명'].str.contains('감염')]['기술명']
""")
code("""
from gensim.models import Word2Vec

# Skip-Gram (sg=1): 중심 단어로 주변 단어를 예측 → 데이터가 적고 드문 단어에 유리
# 문장이 183개로 적어서 min_count=1(모든 단어 사용), epochs를 크게
model_sg = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=1, epochs=200, seed=42, workers=1)

print("=== Skip-Gram: '감염'과 유사한 단어 ===")
for word, score in model_sg.wv.most_similar('감염', topn=10):
    print(word, round(score, 3))
""")
code("""
# CBOW (sg=0): 주변 단어로 중심 단어를 예측
model_cbow = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=0, epochs=200, seed=42, workers=1)

print("=== CBOW: '감염'과 유사한 단어 ===")
for word, score in model_cbow.wv.most_similar('감염', topn=10):
    print(word, round(score, 3))
""")
code("""
# 유사어로 나온 단어가 어떤 기술명에 쓰였는지 확인
for word in ['수술실', '병상']:
    print(f'[{word}]')
    print(it_df[it_df['기술명'].str.contains(word)]['기술명'].head(3).tolist())
""")
md("Q4_RESULT")

# ───────── Q5 ─────────
md("""
---
## 5. 대출금액을 가장 많이 집행한 은행 → 해당 은행의 세부분류별 대출금액
""")
code("""
# 대출금액에 '금액없음' 문자가 섞여 있음 → 숫자형인 '대출금' 컬럼 사용 (금액없음 = 0)
print(df['대출금액'].unique()[:10])

# 은행별 대출금 합계 (실제 집행된 = 승인된 건만)
approved_df = df[df['진행상태'] == '여신승인']
bank_sum = approved_df.groupby('신청은행')['대출금'].sum().sort_values(ascending=False)
bank_sum
""")
code("""
top_bank = bank_sum.index[0]
print('대출금액을 가장 많이 집행한 은행:', top_bank)
print('전체 대비 비중:', round(bank_sum.iloc[0] / bank_sum.sum() * 100, 1), '%')
""")
code("""
# 해당 은행의 헬스케어 세부분류별 대출금 합계
bank_df = approved_df[approved_df['신청은행'] == top_bank]
bank_sub = bank_df.groupby('헬스케어 세부분류')['대출금'].agg(['sum', 'count']).sort_values('sum', ascending=False)
bank_sub.head(10)
""")
md("Q5_RESULT")

# ───────── Q6 ─────────
md("""
---
## 6. 기술명·대출금액·기술분류·신청은행·자본금 → 대출 승인 여부 예측 (Scikit-learn)
- 기술명(텍스트) → **TF-IDF**
- 기술분류, 신청은행(범주) → **원-핫 인코딩**(get_dummies)
- 대출금액, 자본금(숫자) → 그대로 사용
""")
code("""
# 목표 변수: 승인 = 1, 거부 = 0
df['승인'] = (df['진행상태'] == '여신승인').astype(int)

# 대출금액의 '금액없음'을 0으로 바꾸고 숫자로 변환
df['대출금액_숫자'] = pd.to_numeric(df['대출금액'], errors='coerce').fillna(0)

# 기술명을 명사만 남긴 문장으로 변환 (TF-IDF 입력용)
df['기술명_명사'] = [' '.join([n for n in okt.nouns(t) if len(n) > 1]) for t in df['기술명']]
df[['기술명', '기술명_명사']].head()
""")
code("""
# 범주형 → 원-핫 인코딩, 숫자형은 그대로
etc = pd.get_dummies(df[['기술분류', '신청은행']]).astype(int)
etc['대출금액'] = df['대출금액_숫자']
etc['자본금'] = df['자본금']
print(etc.shape)
etc.head()
""")
code("""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 학습용 80% / 평가용 20% 나누기 (stratify: 승인/거부 비율을 똑같이 유지)
train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df['승인'], random_state=42)

# TF-IDF는 학습 데이터로만 fit (평가 데이터 정보가 섞이지 않도록)
tfidf = TfidfVectorizer(max_features=300)
train_text = tfidf.fit_transform(df.loc[train_idx, '기술명_명사']).toarray()
test_text = tfidf.transform(df.loc[test_idx, '기술명_명사']).toarray()

# 텍스트 특성 + 나머지 특성 합치기
X_train = np.hstack([train_text, etc.loc[train_idx].values])
X_test = np.hstack([test_text, etc.loc[test_idx].values])
y_train = df.loc[train_idx, '승인']
y_test = df.loc[test_idx, '승인']
print(X_train.shape, X_test.shape)
""")
code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 랜덤포레스트 (거부가 적어서 class_weight='balanced'로 균형 맞춤)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print('정확도:', round(accuracy_score(y_test, pred), 3))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=['여신거부', '여신승인']))
""")
code("""
# 어떤 변수가 중요했는지 확인
feature_names = list(tfidf.get_feature_names_out()) + list(etc.columns)
importance = pd.Series(rf.feature_importances_, index=feature_names)
importance.sort_values(ascending=False).head(10)
""")
md("""
### 성능이 너무 높다 → 대출금액 확인
대출금액이 가장 중요한 변수로 나왔다. 대출금액과 승인 여부를 직접 비교해본다.
""")
code("""
# 대출금액이 0인지 여부 × 진행상태
pd.crosstab(df['대출금액_숫자'] == 0, df['진행상태'])
""")
code("""
# 대출금액을 빼고 다시 학습 (기술명, 기술분류, 신청은행, 자본금만 사용)
X_train2 = np.hstack([train_text, etc.drop(columns='대출금액').loc[train_idx].values])
X_test2 = np.hstack([test_text, etc.drop(columns='대출금액').loc[test_idx].values])

rf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf2.fit(X_train2, y_train)
pred2 = rf2.predict(X_test2)

print('정확도:', round(accuracy_score(y_test, pred2), 3))
print(confusion_matrix(y_test, pred2))
print(classification_report(y_test, pred2, target_names=['여신거부', '여신승인']))
""")
md("Q6_RESULT")

# ───────── Q7 ─────────
md("""
---
## 7. 기술명 → 대출 승인 여부 신경망 모델 (model_loan.h5 저장)
순서: 텍스트 → 정수 시퀀스(Text to Sequence) → 길이 맞추기(Padding) → Embedding + LSTM
""")
code("""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 6번에서 나눈 학습/평가 데이터 그대로 사용
train_texts = df.loc[train_idx, '기술명_명사']
test_texts = df.loc[test_idx, '기술명_명사']

# 1) Tokenizer: 단어마다 번호 붙이기 (학습 데이터 기준, 처음 보는 단어는 'OOV')
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1     # +1은 패딩(0)용
print('단어 수:', vocab_size)

# 2) Text to Sequence: 문장 → 숫자 리스트
train_seq = tokenizer.texts_to_sequences(train_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)
print(train_texts.iloc[0], '→', train_seq[0])
""")
code("""
# 3) Padding: 길이를 똑같이 맞추기 (가장 긴 문장 길이 기준)
max_len = max(len(s) for s in train_seq)
print('최대 길이:', max_len)

X_train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')
print(X_train_pad[0])
""")
code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf

tf.random.set_seed(42)

model = Sequential()
model.add(Embedding(vocab_size, 64))            # 단어 번호 → 64차원 벡터
model.add(LSTM(32))                             # 단어 순서를 고려해 문장 특징 추출
model.add(Dropout(0.5))                         # 과적합 방지
model.add(Dense(1, activation='sigmoid'))       # 승인 확률 (0~1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
""")
code("""
# 거부 데이터가 적어서 거부에 가중치를 더 줌 (거부 405건 : 승인 1808건 ≈ 1 : 4.5)
class_weight = {0: 4.5, 1: 1}

history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32,
                    validation_split=0.2, class_weight=class_weight)
""")
code("""
# 평가
loss, acc = model.evaluate(X_test_pad, y_test)
print('평가 정확도:', round(acc, 3))
print('(비교) 전부 승인으로 찍었을 때 정확도:', round(y_test.mean(), 3))

pred_nn = (model.predict(X_test_pad) > 0.5).astype(int).ravel()
print(confusion_matrix(y_test, pred_nn))
print(classification_report(y_test, pred_nn, target_names=['여신거부', '여신승인']))
""")
code("""
# 학습 과정 그래프 (학습 정확도 vs 검증 정확도)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
""")
code("""
# 모델 저장
model.save('model_loan.h5')

# 저장한 모델 불러와서 새 기술명 예측해보기
from tensorflow.keras.models import load_model
loaded_model = load_model('model_loan.h5')

new_text = 'AI 기반 감염관리 모니터링 시스템 개발'
new_seq = tokenizer.texts_to_sequences([' '.join([n for n in okt.nouns(new_text) if len(n) > 1])])
new_pad = pad_sequences(new_seq, maxlen=max_len, padding='post')
print(new_text, '→ 승인 확률:', round(float(loaded_model.predict(new_pad)[0][0]), 3))
""")
code("""
# 코랩에서 파일 다운로드
files.download('model_loan.h5')
""")
md("Q7_RESULT")
md("SUMMARY")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
results = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
for c in nb.cells:
    if c.cell_type == "markdown" and (c.source.endswith("_RESULT") or c.source == "SUMMARY"):
        c.source = results.get(c.source, "**결과 해석**: (실행 후 작성)")
nbf.write(nb, "mission4_tech_credit.ipynb")
print("written", len(nb.cells), "cells")
