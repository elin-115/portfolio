"""[미션2] DUR 금기 의약품 - 코랩 제출용 노트북 생성기 (기본 난이도)."""
import json
import sys

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# [미션2] DUR 금기 의약품 목록 분류 데이터 분석
""")

# ───────── 0. 준비 ─────────
md("""
## 0. 준비

**데이터 업로드 방법**: 병용금기 파일이 255MB로 크기 때문에 `files.upload()`는 매우 느립니다.
구글 드라이브에 `07_DUR_Data` 폴더를 만들어 CSV 5개를 올린 뒤 아래처럼 마운트하는 방법을 권장합니다.
""")
code("""
# 코랩에 없는 라이브러리 설치
!pip install konlpy gensim
""")
code("""
# 구글 드라이브 마운트 (드라이브 내 07_DUR_Data 폴더에 CSV 5개를 올려둔 경우)
from google.colab import drive
drive.mount('/content/drive')

PATH = '/content/drive/MyDrive/07_DUR_Data/'     # 폴더 경로 (끝에 / 포함)

import os
print(os.listdir(PATH))
""")
code("""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# CSV 파일이 cp949(한글 윈도우) 인코딩이라 encoding='cp949' 필수
# (생략하면 UnicodeDecodeError 발생)
def read_dur(name):
    return pd.read_csv(PATH + f'의약품안전사용서비스(DUR)_{name} 품목리스트 2026.6.csv',
                       encoding='cp949')

임부 = read_dur('임부금기')
연령 = read_dur('연령금기')
노인 = read_dur('노인주의')
노인해열 = read_dur('노인주의(해열진통소염제)')

print('임부금기:', 임부.shape, list(임부.columns))
임부.head(3)
""")
code("""
# 병용금기는 87만 행(255MB)이라 필요한 컬럼만 읽어서 메모리 절약
병용 = pd.read_csv(PATH + '의약품안전사용서비스(DUR)_병용금기 품목리스트 2026.6.csv',
                  encoding='cp949',
                  usecols=['성분명A', '성분코드A', '제품코드A', '성분명B', '성분코드B', '제품코드B',
                           '고시번호', '고시일자', '상세정보', '비고'])
print('병용금기:', 병용.shape)
병용.head(3)
""")

md("""
### 전처리: 제품 단위 → 성분(조합) 단위로 집계
- 원본 파일은 **제품 1개 = 1행**(같은 성분의 제품이 수백 개씩 반복)
- 변수 정의서의 분석 단위는 **성분(조합) 1개 = 1행**이고, 여기에 **제품수**(해당 금기에 걸리는 제품 수) 컬럼이 붙음
- 따라서 성분(조합)별로 행 수를 세어 `제품수`를 만들고, 중복 행을 제거
""")
code("""
# groupby(...).transform('size') : 같은 그룹의 행이 몇 개인지를 각 행에 붙여줌
# drop_duplicates() : 그룹마다 첫 행만 남김 → 성분(조합) 단위 데이터 완성

# (1) 병용금기: 성분코드A + 성분코드B 조합 단위
병용['제품수'] = 병용.groupby(['성분코드A', '성분코드B'])['제품코드A'].transform('size')
병용_성분 = 병용.drop_duplicates(['성분코드A', '성분코드B']).copy()
병용_성분 = 병용_성분.rename(columns={'성분코드A': '성분코드', '성분명A': '성분명',
                                  '성분코드B': '병용성분코드', '성분명B': '병용성분명'})
병용_성분['금기유형'] = '병용금기'
print('병용금기 조합 수:', len(병용_성분), '/ 제품수 최대:', 병용_성분['제품수'].max())

# (2) 임부금기: 성분코드 단위
임부['제품수'] = 임부.groupby('성분코드')['제품코드'].transform('size')
임부_성분 = 임부.drop_duplicates('성분코드').copy()
임부_성분['금기유형'] = '임부금기'
print('임부금기 성분 수:', len(임부_성분), '/ 제품수 최대:', 임부_성분['제품수'].max())
""")
code("""
# (3) 연령금기 / 노인주의도 같은 방식으로 정리
연령['제품수'] = 연령.groupby(['성분코드', '특정연령', '연령처리조건'])['제품코드'].transform('size')
연령_성분 = 연령.drop_duplicates(['성분코드', '특정연령', '연령처리조건']).copy()
연령_성분['금기유형'] = '연령금기'

# 노인주의 파일은 컬럼명이 달라서 통일 (약품상세정보 → 상세정보, 공고일자 → 고시일자)
노인 = 노인.rename(columns={'약품상세정보': '상세정보', '공고일자': '고시일자', '공고번호': '고시번호'})
노인['제품수'] = 노인.groupby('성분코드')['제품코드'].transform('size')
노인_성분 = 노인.drop_duplicates('성분코드').copy()
노인_성분['금기유형'] = '노인주의'

노인해열 = 노인해열.rename(columns={'약품상세정보': '상세정보'})
노인해열['제품수'] = 노인해열.groupby('성분코드')['제품코드'].transform('size')
노인해열_성분 = 노인해열.drop_duplicates('성분코드').copy()
노인해열_성분['금기유형'] = '노인주의_해열진통소염제'

print(len(연령_성분), len(노인_성분), len(노인해열_성분))
""")
code("""
# (4) 5개 금기유형을 하나로 합치기 (변수 정의서의 통합 테이블)
사용컬럼 = ['금기유형', '성분코드', '성분명', '병용성분코드', '병용성분명', '금기등급',
          '특정연령', '특정연령단위', '연령처리조건', '상세정보', '비고', '고시번호', '고시일자', '제품수']

dur = pd.concat([병용_성분, 임부_성분, 연령_성분, 노인_성분, 노인해열_성분], ignore_index=True)
dur = dur.reindex(columns=사용컬럼)      # 없는 컬럼은 자동으로 결측(NaN) 처리

print(dur.shape)
print(dur['금기유형'].value_counts())
dur.head(3)
""")
code("""
# 형태소 분석기 준비 (상세정보 텍스트 분석용)
from konlpy.tag import Okt
okt = Okt()

print(okt.nouns('임부 투여금기, 동물실험에서 간독성의 증거 나타남'))
""")

# ───────── Q1 ─────────
md("""
---
## 1. 병용금기 중 제품수 합계 상위 100개 성분의 상세정보 최빈 단어
""")
code("""
# 병용금기만 뽑아서 성분명별 제품수 합계 → 상위 100개
병용금기 = dur[dur['금기유형'] == '병용금기']
top100 = 병용금기.groupby('성분명')['제품수'].sum().sort_values(ascending=False).head(100)

print('상위 100개 성분의 제품수 합계 범위:', top100.min(), '~', top100.max())
top100.head(10)
""")
code("""
# 상위 100개 성분에 해당하는 행들의 상세정보(금기 사유) 모으기
top100_data = 병용금기[병용금기['성분명'].isin(top100.index)]
print('행 수:', len(top100_data), '/ 상세정보 종류:', top100_data['상세정보'].nunique())

# Okt로 명사 추출 (2글자 이상만)
nouns_list = []
for text in top100_data['상세정보'].dropna():
    nouns_list.extend([n for n in okt.nouns(str(text)) if len(n) > 1])

word_counts = pd.Series(nouns_list).value_counts()
print('=== 상세정보 최빈 단어 상위 15 ===')
display(word_counts.head(15))
""")
code("""
# 같은 성분 조합이 수천 건씩 반복되면 같은 문장이 그만큼 여러 번 세어짐
# → 문장(상세정보) 중복을 제거하고 다시 세어 비교
unique_text = top100_data['상세정보'].dropna().drop_duplicates()
print('중복 제거 후 문장 수:', len(unique_text))

nouns_unique = []
for text in unique_text:
    nouns_unique.extend([n for n in okt.nouns(str(text)) if len(n) > 1])

print('=== 중복 제거 기준 상위 15 ===')
display(pd.Series(nouns_unique).value_counts().head(15))
""")
md("Q1_RESULT")

# ───────── Q2 ─────────
md("""
---
## 2. 고시 경과기간 계산 + 임부금기 등급(1/2)에 따른 차이 검정
""")
code("""
임부금기 = dur[dur['금기유형'] == '임부금기'].copy()

# 고시일자 → 날짜형으로 변환 후 경과기간(연) 계산
임부금기['고시일자'] = pd.to_datetime(임부금기['고시일자'], errors='coerce')
오늘 = pd.Timestamp.today()
임부금기['경과기간'] = (오늘 - 임부금기['고시일자']).dt.days / 365.25   # 365.25 = 윤년 포함 평균

print('고시일자 범위:', 임부금기['고시일자'].min().date(), '~', 임부금기['고시일자'].max().date())
print('금기등급 분포:'); print(임부금기['금기등급'].value_counts(dropna=False))
""")
code("""
# 등급이 1 또는 2인 데이터만 사용 (M 등급 40건, 결측 1건은 제외)
임부금기['금기등급'] = 임부금기['금기등급'].astype(str)
d2 = 임부금기[임부금기['금기등급'].isin(['1', '2'])].dropna(subset=['경과기간'])
print('검정 데이터:', len(d2))

d2.groupby('금기등급')['경과기간'].describe().round(2)
""")
code("""
from scipy import stats

grade1 = d2[d2['금기등급'] == '1']['경과기간']
grade2 = d2[d2['금기등급'] == '2']['경과기간']

# 정규성 검정 (p < 0.05 이면 정규분포가 아님)
print('정규성 - 1등급:', stats.shapiro(grade1).pvalue)
print('정규성 - 2등급:', stats.shapiro(grade2).pvalue)
""")
code("""
# 정규분포가 아니므로 비모수 검정(Mann-Whitney U) 사용
# 귀무가설: 1등급과 2등급의 고시 경과기간에 차이가 없다
print('1등급 중앙값:', round(grade1.median(), 2), '/ 2등급 중앙값:', round(grade2.median(), 2))
print(stats.mannwhitneyu(grade1, grade2))

# (참고) 평균 비교
print('1등급 평균:', round(grade1.mean(), 2), '/ 2등급 평균:', round(grade2.mean(), 2))
print(stats.ttest_ind(grade1, grade2, equal_var=False))
""")
md("Q2_RESULT")

# ───────── Q3 ─────────
md("""
---
## 3. 투여경로(A/B/C)와 임부금기 등급의 관계 검정 + 경로별 1등급 비율
- 성분코드 7번째 글자 = 투여경로 (A 내복 / B 주사 / C 외용 / D 기타)
""")
code("""
# 문자열에서 7번째 글자 = 인덱스 6 (0부터 시작)
임부금기['투여경로'] = 임부금기['성분코드'].str[6]
print(임부금기['투여경로'].value_counts())

# 예시 확인
임부금기[['성분코드', '투여경로']].head()
""")
code("""
# 등급 1·2, 투여경로 A·B·C만 사용
d3 = 임부금기[임부금기['금기등급'].isin(['1', '2']) & 임부금기['투여경로'].isin(['A', 'B', 'C'])]

route_table = pd.crosstab(d3['투여경로'], d3['금기등급'])
route_table['1등급비율'] = (route_table['1'] / (route_table['1'] + route_table['2'])).round(3)
route_table
""")
code("""
# 카이제곱 독립성 검정
# 귀무가설: 투여경로와 금기등급은 관계가 없다(독립이다)
chi2, p, dof, expected = stats.chi2_contingency(route_table[['1', '2']])
print('카이제곱 통계량:', round(chi2, 3))
print('p-value:', p)
print('자유도:', dof)
print('기대빈도 최솟값:', round(expected.min(), 1))     # 5 이상이면 검정 조건 만족
""")
md("Q3_RESULT")

md("""
### 3-1. (보충) 등급 × 투여경로별 사유 문구 정리
투여경로에 따라 위험도가 다르다면 **금기 사유 문구도 다르게 쓰이는지** 확인한다. 6·7번 모델 해석의 근거가 된다.
""")
code("""
# 결측을 '(결측)'으로 표시해 함께 집계
d3 = d3.copy()
d3['상세정보'] = d3['상세정보'].fillna('(결측)')

# 등급 × 경로 건수와, 셀마다 문구가 몇 종류인지
print(pd.crosstab(d3['투여경로'], d3['금기등급'], margins=True))
print()
print(d3.groupby(['금기등급', '투여경로'])['상세정보'].nunique())
""")
code("""
# 셀별 대표 문구 (상위 3개)
for g in ['1', '2']:
    for r in ['A', 'B', 'C']:
        top = d3[(d3['금기등급'] == g) & (d3['투여경로'] == r)]['상세정보'].value_counts().head(3)
        print(f'=== {g}등급 / {r} (n={len(d3[(d3.금기등급==g)&(d3.투여경로==r)])})')
        for text, cnt in top.items():
            print(f'  {cnt:4d}  {text[:70]}')
        print()
""")
code("""
# 같은 문구가 1등급과 2등급에 함께 쓰이는지 확인
#  nunique()가 2 = 그 문구가 1등급·2등급 양쪽에 모두 등장
공유 = d3.groupby('상세정보')['금기등급'].nunique()
공유문구 = 공유[공유 > 1].index
print('1·2등급에 함께 쓰인 문구:', len(공유문구), '종류 /', d3['상세정보'].isin(공유문구).sum(), '건')

표 = pd.crosstab(d3[d3['상세정보'].isin(공유문구)]['상세정보'], d3['금기등급'])
표['합계'] = 표['1'] + 표['2']
표.sort_values('합계', ascending=False).head(8)
""")
md("Q3B_RESULT")

# ───────── Q4 ─────────
md("""
---
## 4. 임부금기 상세정보 Word2Vec → '태아'와 유사한 단어
""")
code("""
from gensim.models import Word2Vec

# 상세정보 한 건을 문장 하나로 보고 명사 리스트 만들기
sentences = []
for text in 임부금기['상세정보'].dropna():
    sentences.append([n for n in okt.nouns(str(text)) if len(n) > 1])

print('문장 수:', len(sentences))
print("'태아' 등장 횟수:", sum(s.count('태아') for s in sentences))
print(sentences[:3])
""")
code("""
# Skip-Gram (sg=1): 중심 단어로 주변 단어를 예측
model_sg = Word2Vec(sentences, vector_size=100, window=3, min_count=2,
                    sg=1, epochs=100, seed=42, workers=1)

print("=== Skip-Gram: '태아'와 유사한 단어 ===")
for word, score in model_sg.wv.most_similar('태아', topn=10):
    print(word, round(score, 3))
""")
code("""
# CBOW (sg=0): 주변 단어로 중심 단어를 예측 → 자주 나오는 단어에 유리
model_cbow = Word2Vec(sentences, vector_size=100, window=3, min_count=2,
                      sg=0, epochs=100, seed=42, workers=1)

print("=== CBOW: '태아'와 유사한 단어 ===")
for word, score in model_cbow.wv.most_similar('태아', topn=10):
    print(word, round(score, 3))
""")
code("""
# 유사어가 실제로 어떤 문장에 쓰였는지 확인
for word in ['태자', '모체']:
    예시 = 임부금기[임부금기['상세정보'].fillna('').str.contains(word)]['상세정보']
    print(f'[{word}] {len(예시)}건')
    print(예시.head(2).tolist())
""")
md("Q4_RESULT")

# ───────── Q5 ─────────
md("""
---
## 5. 제품수 합계가 가장 많은 성분 → 그 성분이 어떤 병용성분과 제품수가 가장 많은지
""")
code("""
# 전체 금기유형을 합쳐 성분명별 제품수 합계
성분별 = dur.groupby('성분명')['제품수'].sum().sort_values(ascending=False)
성분별.head(10)
""")
code("""
top_ingredient = 성분별.index[0]
print('제품수 합계가 가장 많은 성분:', top_ingredient, '/ 합계:', 성분별.iloc[0])

# 그 성분의 병용금기 상대 성분별 제품수
상대성분 = (dur[(dur['성분명'] == top_ingredient) & (dur['금기유형'] == '병용금기')]
          .groupby('병용성분명')['제품수'].sum().sort_values(ascending=False))
상대성분.head(10)
""")
code("""
# 1위 조합의 금기 사유 확인
best = 상대성분.index[0]
print(f'{top_ingredient} + {best} : 제품수 {상대성분.iloc[0]}')
print(dur[(dur['성분명'] == top_ingredient) & (dur['병용성분명'] == best)]['상세정보'].values[:2])
""")
md("Q5_RESULT")

# ───────── Q6 ─────────
md("""
---
## 6. 상세정보 + 투여경로 + 제품수 + 경과기간 → 임부금기 1등급 판별 (Scikit-learn)
- 상세정보(텍스트) → **TF-IDF**
- 투여경로(범주) → **원-핫 인코딩**
- 제품수, 경과기간(숫자) → 그대로 사용
""")
code("""
# 목표 변수: 1등급이면 1, 2등급이면 0
model_df = 임부금기[임부금기['금기등급'].isin(['1', '2'])].dropna(subset=['경과기간']).copy()
model_df['1등급'] = (model_df['금기등급'] == '1').astype(int)
print('데이터:', len(model_df), '/ 1등급 비율:', round(model_df['1등급'].mean(), 3))

# 상세정보 → 명사만 공백으로 이은 문장 (TF-IDF 입력용)
model_df['상세정보_명사'] = [' '.join([n for n in okt.nouns(str(t)) if len(n) > 1])
                        for t in model_df['상세정보'].fillna('')]
model_df[['상세정보', '상세정보_명사']].head(3)
""")
code("""
# 투여경로 원-핫 인코딩 + 숫자형 변수 붙이기
etc = pd.get_dummies(model_df['투여경로'], prefix='경로').astype(int)
etc['제품수'] = model_df['제품수']
etc['경과기간'] = model_df['경과기간']
etc.head(3)
""")
code("""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 학습 80% / 평가 20% (stratify: 1등급 비율 유지)
train_idx, test_idx = train_test_split(model_df.index, test_size=0.2,
                                       stratify=model_df['1등급'], random_state=42)

# TF-IDF는 학습 데이터로만 fit
tfidf = TfidfVectorizer(max_features=300)
train_text = tfidf.fit_transform(model_df.loc[train_idx, '상세정보_명사']).toarray()
test_text = tfidf.transform(model_df.loc[test_idx, '상세정보_명사']).toarray()

# 텍스트 + 나머지 변수 합치기
X_train = np.hstack([train_text, etc.loc[train_idx].values])
X_test = np.hstack([test_text, etc.loc[test_idx].values])
y_train = model_df.loc[train_idx, '1등급']
y_test = model_df.loc[test_idx, '1등급']
print(X_train.shape, X_test.shape)
""")
code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# class_weight='balanced' : 1등급이 16%로 적으므로 가중치 보정
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print('정확도:', round(accuracy_score(y_test, pred), 3))
print('(비교) 전부 2등급으로 찍었을 때:', round(1 - y_test.mean(), 3))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=['2등급', '1등급'], digits=3))
""")
code("""
# 변수 중요도
feature_names = list(tfidf.get_feature_names_out()) + list(etc.columns)
importance = pd.Series(rf.feature_importances_, index=feature_names)
importance.sort_values(ascending=False).head(10)
""")
code("""
# 텍스트 없이 정형 변수(투여경로·제품수·경과기간)만으로 학습하면?
n_etc = etc.shape[1]
rf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf2.fit(X_train[:, -n_etc:], y_train)
pred2 = rf2.predict(X_test[:, -n_etc:])

print('정형 변수만 정확도:', round(accuracy_score(y_test, pred2), 3))
print(classification_report(y_test, pred2, target_names=['2등급', '1등급'], digits=3))
""")
md("Q6_RESULT")

# ───────── Q7 ─────────
md("""
---
## 7. 상세정보 → 임부금기 1등급 판별 RNN 모델 (model_dur.h5 저장)
순서: 텍스트 → 정수 시퀀스(Text to Sequence) → 패딩(Padding) → Embedding + SimpleRNN
""")
code("""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 6번에서 나눈 학습/평가 데이터를 그대로 사용
train_texts = model_df.loc[train_idx, '상세정보_명사']
test_texts = model_df.loc[test_idx, '상세정보_명사']

# (1) Tokenizer: 단어마다 번호 부여 (학습 데이터 기준)
tokenizer = Tokenizer(oov_token='OOV')     # 학습에 없던 단어는 OOV 처리
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
print('단어 수:', vocab_size)

# (2) Text to Sequence
train_seq = tokenizer.texts_to_sequences(train_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)
print(train_texts.iloc[0], '→', train_seq[0])
""")
code("""
# (3) Padding: 길이를 똑같이 맞추기
#     상세정보는 길이 차이가 커서 95% 분위수로 자름 (너무 긴 문장 1~2건 때문에 전체가 길어지는 것 방지)
lengths = [len(s) for s in train_seq]
print('단어 수 평균:', round(np.mean(lengths), 1), '/ 최대:', max(lengths))

max_len = int(np.percentile(lengths, 95))
print('maxlen:', max_len)

X_train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post')
print(X_train_pad[0])
""")
code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
import tensorflow as tf

tf.random.set_seed(42)

# RNN 계열 알고리즘 사용 (SimpleRNN)
model = Sequential()
model.add(Embedding(vocab_size, 64))       # 단어 번호 → 64차원 벡터
model.add(SimpleRNN(32))                   # 단어를 순서대로 읽으며 문장 특징 추출
model.add(Dropout(0.5))                    # 과적합 방지
model.add(Dense(1, activation='sigmoid'))  # 1등급일 확률

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
""")
code("""
# 1등급(16%)이 적으므로 가중치 보정 후 학습
class_weight = {0: 1, 1: 5}

history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32,
                    validation_split=0.2, class_weight=class_weight)
""")
code("""
# 평가
loss, acc = model.evaluate(X_test_pad, y_test)
print('평가 정확도:', round(acc, 3))
print('(비교) 전부 2등급으로 찍었을 때:', round(1 - y_test.mean(), 3))

pred_rnn = (model.predict(X_test_pad) > 0.5).astype(int).ravel()
print(confusion_matrix(y_test, pred_rnn))
print(classification_report(y_test, pred_rnn, target_names=['2등급', '1등급'], digits=3))
""")
code("""
# 학습 과정 그래프
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.show()
""")
code("""
# 모델 저장 (.h5)
model.save('model_dur.h5')

# 저장한 모델 불러와 새 문장으로 예측
from tensorflow.keras.models import load_model
loaded_model = load_model('model_dur.h5')

new_text = '임부 투여금기, 동물실험에서 기형 유발이 보고됨'
new_seq = tokenizer.texts_to_sequences([' '.join([n for n in okt.nouns(new_text) if len(n) > 1])])
new_pad = pad_sequences(new_seq, maxlen=max_len, padding='post')
print(new_text, '→ 1등급 확률:', round(float(loaded_model.predict(new_pad)[0][0]), 3))
""")
code("""
# 파일 다운로드
from google.colab import files
files.download('model_dur.h5')
""")
md("Q7_RESULT")

# ───────── Q8 ─────────
md("""
---
## 8. 병용금기 상세정보(중복 제거) 토픽 모델링 (LDA)
- LDA(Latent Dirichlet Allocation): 문서를 여러 '주제(토픽)'의 혼합으로 보고, 주제별 주요 단어를 찾아내는 방법
""")
code("""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 병용금기 상세정보에서 중복 문장 제거
docs_raw = 병용금기['상세정보'].dropna().drop_duplicates()
print('중복 제거 후 문서 수:', len(docs_raw))

# 명사만 남기기
docs = [' '.join([n for n in okt.nouns(str(t)) if len(n) > 1]) for t in docs_raw]
print(docs[:3])
""")
code("""
# LDA 입력은 단어 빈도(BoW) 행렬
# max_df=0.6 : 60% 넘는 문서에 나오는 흔한 단어 제외
# min_df=2   : 1개 문서에만 나오는 단어 제외
cv = CountVectorizer(max_df=0.6, min_df=2)
X_lda = cv.fit_transform(docs)
print('문서-단어 행렬:', X_lda.shape)
""")
code("""
# 토픽 개수를 3~6개로 바꿔가며 perplexity(낮을수록 좋음) 비교
for k in [3, 4, 5, 6]:
    lda_tmp = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=30).fit(X_lda)
    print(f'토픽 {k}개 → perplexity {round(lda_tmp.perplexity(X_lda), 1)}')
""")
code("""
# 토픽 5개로 최종 학습
lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=30)
lda.fit(X_lda)

words = cv.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[-10:][::-1]]   # 비중 높은 단어 10개
    print(f'토픽 {i+1}:', ', '.join(top_words))
""")
code("""
# 각 토픽에 속하는 문장 예시 확인 (해석용)
topic_result = lda.transform(X_lda)          # 문서마다 토픽별 비중
docs_raw = docs_raw.reset_index(drop=True)

for i in range(5):
    print(f'--- 토픽 {i+1} 대표 문장')
    for idx in topic_result[:, i].argsort()[-2:][::-1]:
        print('  ', docs_raw[idx][:70])
""")
md("Q8_RESULT")
md("SUMMARY")
md("BUSINESS")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
results = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
for c in nb.cells:
    if c.cell_type == "markdown" and (c.source.endswith("_RESULT") or c.source in ("SUMMARY", "BUSINESS")):
        c.source = results.get(c.source, "**결과 해석**: (실행 후 작성)")
nbf.write(nb, "mission2_dur.ipynb")
print("written", len(nb.cells), "cells")
