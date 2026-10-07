"""코랩 제출용 노트북(mission4_tech_credit.ipynb) 생성기. cells 리스트 = (종류, 내용)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# [개별미션 4차] 헬스케어 기술보증 데이터 분석
M사 헬스케어 기술금융 심사 데이터(2,213개 기업)로 **어떤 기업 속성과 기술 내용이 대출 승인을 좌우하는지** 정형·텍스트 분석으로 규명하고, 승인 예측 모델을 구성한다.

| 문항 | 사용 기법 |
|---|---|
| Q1 | 형태소 분석(Okt) + **BoW**(CountVectorizer) |
| Q2 | 업력 계산 + 정규성 검정 → Mann-Whitney U / t-test |
| Q3 | 교차표 + 카이제곱 독립성 검정 |
| Q4 | 형태소 분석(Okt) + **Word2Vec**(Skip-Gram / CBOW) |
| Q5 | groupby 집계 |
| Q6 | **TF-IDF** + One-Hot + 수치형 → Scikit-learn 분류 모델 |
| Q7 | **Text to Sequence + Padding** → Embedding·LSTM 신경망 → `model_loan.h5` |
""")

md("## 0. 환경 준비")
code("""
# 코랩에는 konlpy, gensim이 기본 설치되어 있지 않으므로 설치 (tensorflow, sklearn은 기본 제공)
!pip install -q konlpy gensim
""")
code("""
# 데이터 업로드: 실행하면 파일 선택 창이 뜬다 → 06_Tech_Credit_Data.xlsx 선택
from google.colab import files
uploaded = files.upload()
PATH = list(uploaded.keys())[0]
""")
code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

pd.set_option('display.max_colwidth', 60)

# '기술데이터' 시트 = 기업 단위 원본 / '분류별_요약' 시트 = 세부분류별 건수 요약(분석엔 미사용)
df = pd.read_excel(PATH, sheet_name='기술데이터')
print(df.shape)
df.head()
""")
code("""
# 목표변수 Y: 진행상태(여신승인/여신거부) → 1/0
df['승인'] = (df['진행상태'] == '여신승인').astype(int)
print(df['진행상태'].value_counts())
print(f"승인률: {df['승인'].mean():.1%}")   # 81.7% → 불균형 데이터(거부가 소수)
""")

md("""
### 데이터 점검 (분석 전에 확인한 사항)
- `대출금액`: 숫자와 **'금액없음'(9건)** 문자가 섞인 object형 → `대출금`(정수형, 금액없음=0)과 같은 정보
- `설립일`: **1900/01/01**은 실제 날짜가 아니라 결측 코드(20건) → 업력 계산에서 제외
- `자본금`: 음수(자본잠식) 존재
- `총합` = `대출금` + `자본금` (100% 일치) → 대출금을 포함한 파생 변수
""")
code("""
print("대출금액 문자값:", df.loc[pd.to_numeric(df['대출금액'], errors='coerce').isna(), '대출금액'].unique())
print("설립연도 1900 건수:", (df['설립연도'] == 1900).sum())
print("자본금 음수 건수:", (df['자본금'] < 0).sum())
print("총합 = 대출금 + 자본금 :", (df['총합'] == df['대출금'] + df['자본금']).all())
""")

md("""
### 형태소 분석기 준비 (Okt)
기술명은 `~를 적용한 ~ 시스템 개발`처럼 조사·어미가 붙은 문장형이라 띄어쓰기로 자르면 `감염관리`, `적용한` 같은 덩어리가 단어가 된다.
→ **Okt 명사 추출**로 의미 단어만 남긴다. 한 글자 명사(`용`, `물` 등)는 의미가 약해 제외한다.
""")
code("""
from konlpy.tag import Okt
okt = Okt()

def okt_nouns(text):
    \"\"\"기술명 → 2글자 이상 명사 리스트\"\"\"
    return [w for w in okt.nouns(str(text)) if len(w) > 1]

print(okt.pos('실시간 대시보드를 적용한 감염관리 모니터링 시스템 개발'))
print(okt_nouns('실시간 대시보드를 적용한 감염관리 모니터링 시스템 개발'))

# 이후 Q1·Q4·Q6·Q7에서 반복 사용하므로 전체 기술명을 한 번만 토큰화해 둔다
df['토큰'] = df['기술명'].apply(okt_nouns)
df['토큰_str'] = df['토큰'].apply(' '.join)      # 벡터라이저 입력용 (공백으로 이은 문자열)
df[['기술명', '토큰']].head()
""")

# ───────────────── Q1 ─────────────────
md("""
---
## Q1. 승인 기업 중 자본금 상위 100개 기업 → 기술명 최빈 단어
**방법**: 승인 기업 필터 → 자본금 내림차순 상위 100개 → 기술명을 Okt 명사로 토큰화 → **BoW(CountVectorizer)**로 단어별 빈도 합산
""")
code("""
approved = df[df['승인'] == 1]
top100 = approved.nlargest(100, '자본금')      # 자본금 내림차순 상위 100행

print(f"승인 기업 {len(approved)}개 중 상위 100개, 자본금 범위: {top100['자본금'].min():,} ~ {top100['자본금'].max():,}")
top100[['회사명', '기술명', '헬스케어 세부분류', '자본금']].head(10)
""")
code("""
from sklearn.feature_extraction.text import CountVectorizer

# BoW: 문서(기술명) × 단어 빈도 행렬
#  - 이미 Okt로 명사만 뽑아 공백으로 이어둔 '토큰_str'을 입력 → 공백 기준으로만 자르도록 token_pattern 지정
cv = CountVectorizer(token_pattern=r'[^ ]+')
bow = cv.fit_transform(top100['토큰_str'])
print("BoW 행렬 크기(문서 수 × 단어 수):", bow.shape)

# 열(단어)별 합계 = 100개 기술명 전체에서의 등장 빈도
word_freq = pd.Series(np.asarray(bow.sum(axis=0)).ravel(), index=cv.get_feature_names_out())
word_freq.sort_values(ascending=False).head(15)
""")
code("""
# 기술·개발·제조처럼 어느 기술명에나 붙는 범용어는 '어떤 기술인지'를 알려주지 않음
# → 불용어로 지정해 다시 집계 (CountVectorizer의 stop_words 파라미터 활용)
stopwords = ['기술', '개발', '적용', '제조', '방법', '장치', '시스템', '이용', '기반',
             '사업', '상용', '양산', '설계', '고도화', '국산', '플랫폼', '포함', '조성']

cv_sw = CountVectorizer(token_pattern=r'[^ ]+', stop_words=stopwords)
bow_sw = cv_sw.fit_transform(top100['토큰_str'])
word_freq_sw = pd.Series(np.asarray(bow_sw.sum(axis=0)).ravel(), index=cv_sw.get_feature_names_out())
word_freq_sw.sort_values(ascending=False).head(15)
""")
code("""
# 해석 보조: 상위 단어가 어떤 세부분류에서 왔는지 확인
top100['헬스케어 세부분류'].value_counts().head(7)
""")
md("Q1_RESULT")

# ───────────────── Q2 ─────────────────
md("""
---
## Q2. 업력 계산 및 승인 여부에 따른 업력 대표값 차이 검정
- **업력 = (기준일 − 설립일) / 365.25** (윤년 포함 평균 연 길이)
- 기준일: 문항의 '현재까지' → 오늘 날짜. 신청 시점 업력도 참고로 함께 계산
- 가설: H0 = 승인·거부 기업의 업력 분포(대표값)가 같다 / H1 = 다르다
- 절차: 정규성(Shapiro) → 정규성 기각 시 **Mann-Whitney U**(중앙값 비교, 비모수), 참고로 Welch t-test(평균 비교)
""")
code("""
TODAY = pd.Timestamp.today().normalize()
est = pd.to_datetime(df['설립일자'])
est = est.where(est.dt.year > 1900)           # 1900-01-01 결측 코드 → NaT

df['업력'] = (TODAY - est).dt.days / 365.25
df['업력_신청시'] = (pd.to_datetime(df['신청일자']) - est).dt.days / 365.25

print("기준일:", TODAY.date())
df.groupby('진행상태')['업력'].describe().round(2)
""")
code("""
g = df.dropna(subset=['업력'])
yes = g.loc[g['승인'] == 1, '업력']
no  = g.loc[g['승인'] == 0, '업력']
print(f"유효 {len(g)}건 (설립일 결측 {df['업력'].isna().sum()}건 제외)")

# 1) 정규성 검정 (Shapiro는 표본이 크면 과민 → 승인 그룹은 500개 무작위 추출)
print("Shapiro p  승인:", stats.shapiro(yes.sample(500, random_state=0)).pvalue,
      "/ 거부:", stats.shapiro(no).pvalue)

# 2) 등분산 검정
print("Levene p:", stats.levene(yes, no).pvalue)

# 3) 정규성 기각 → 비모수 Mann-Whitney U 검정 (대표값 = 중앙값)
u = stats.mannwhitneyu(yes, no, alternative='two-sided')
print(f"\\n[Mann-Whitney U] 승인 중앙값 {yes.median():.2f}년 vs 거부 중앙값 {no.median():.2f}년 → U={u.statistic:.0f}, p={u.pvalue:.4f}")

# 참고: 평균 비교 (Welch t-test)
t = stats.ttest_ind(yes, no, equal_var=False)
print(f"[Welch t-test]   승인 평균 {yes.mean():.2f}년 vs 거부 평균 {no.mean():.2f}년 → t={t.statistic:.3f}, p={t.pvalue:.4f}")
""")
code("""
# 민감도 확인: '오늘 기준'은 모든 기업에 같은 날짜를 빼므로, 심사 당시 업력(신청일 기준)으로도 검정
b = df.dropna(subset=['업력_신청시'])
u2 = stats.mannwhitneyu(b.loc[b['승인'] == 1, '업력_신청시'], b.loc[b['승인'] == 0, '업력_신청시'])
print(f"신청 시점 업력 중앙값  승인 {b.loc[b['승인']==1,'업력_신청시'].median():.2f}년 / 거부 {b.loc[b['승인']==0,'업력_신청시'].median():.2f}년 → p={u2.pvalue:.4f}")
""")
md("Q2_RESULT")

# ───────────────── Q3 ─────────────────
md("""
---
## Q3. 평가등급과 승인 여부의 관계 (카이제곱 독립성 검정) + 등급별 승인 비율
- H0 = 평가등급과 승인 여부는 독립이다 / H1 = 관련이 있다
- `T3/A`처럼 **기술등급/신용등급**을 병기한 값(47건)이 섞여 있어 신용등급 부분으로 통일
- 카이제곱 검정은 기대빈도 5 미만 셀이 20%를 넘으면 신뢰도가 떨어짐 → 표본이 극히 적은 등급(AAA 2건, CCC·CC·C 7건)은 인접 등급과 묶어 검정
""")
code("""
# (1) 원 평가등급별 승인 비율
ct_raw = pd.crosstab(df['평가등급'], df['진행상태'])
ct_raw['건수'] = ct_raw.sum(axis=1)
ct_raw['승인비율'] = (ct_raw['여신승인'] / ct_raw['건수']).round(3)
ct_raw.sort_values('건수', ascending=False)
""")
code("""
# (2) 'T3/A' → 'A' 처럼 신용등급 부분만 사용. 'T1','T2'처럼 신용등급이 없는 3건은 제외
grade = df['평가등급'].str.split('/').str[-1]
df['신용등급'] = grade.where(~grade.str.match(r'^T\\d$'))

order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
ct = pd.crosstab(df['신용등급'], df['진행상태']).reindex(order)
ct['건수'] = ct.sum(axis=1)
ct['승인비율'] = (ct['여신승인'] / ct['건수']).round(3)
ct
""")
code("""
# (3) 카이제곱 검정: 소표본 등급을 묶어 5구간으로
band_map = {'AAA': 'A이상', 'AA': 'A이상', 'A': 'A이상', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B',
            'CCC': 'CCC이하', 'CC': 'CCC이하', 'C': 'CCC이하'}
df['등급구간'] = df['신용등급'].map(band_map)
ct5 = pd.crosstab(df['등급구간'], df['진행상태']).reindex(['A이상', 'BBB', 'BB', 'B', 'CCC이하'])

chi2, p, dof, expected = stats.chi2_contingency(ct5)
cramers_v = np.sqrt(chi2 / (ct5.values.sum() * (min(ct5.shape) - 1)))   # 효과크기(0~1)
print(ct5.assign(승인비율=(ct5['여신승인'] / ct5.sum(axis=1)).round(3)))
print(f"\\nχ²={chi2:.2f}, df={dof}, p={p:.4f}, Cramér's V={cramers_v:.3f}")
print("기대빈도 최소값:", expected.min().round(1), "→ CCC이하 구간(7건)은 기대빈도 5 미만")

# 기대빈도 조건을 완전히 만족시키기 위해 CCC이하 제외 후 재검정
chi2b, pb, dofb, expb = stats.chi2_contingency(ct5.drop('CCC이하'))
print(f"[CCC이하 제외] χ²={chi2b:.2f}, df={dofb}, p={pb:.4f}, 기대빈도 최소 {expb.min():.1f}")
""")
md("Q3_RESULT")

# ───────────────── Q4 ─────────────────
md("""
---
## Q4. '병원/의료 IT' 기술명 Word2Vec → '감염'과 유사한 단어
- 대상: 헬스케어 세부분류 = `병원/의료 IT` (183개 기술명)
- 밀집표현 **Word2Vec** 두 방식 비교
  - **Skip-Gram (sg=1)**: 중심 단어 → 주변 단어 예측. 드물게 나오는 단어에 유리
  - **CBOW (sg=0)**: 주변 단어 → 중심 단어 예측
- 코퍼스가 작아(183문장) `min_count=1`, 반복 `epochs`를 크게, 결과 재현을 위해 `seed` 고정·`workers=1`
""")
code("""
from gensim.models import Word2Vec

it = df[df['헬스케어 세부분류'] == '병원/의료 IT']
sentences = it['토큰'].tolist()            # [['실시간', '대시보드', '적용', '감염', '관리', ...], ...]
print("문장 수:", len(sentences))
print("'감염' 등장 횟수:", sum(s.count('감염') for s in sentences))
it.loc[it['기술명'].str.contains('감염'), '기술명'].tolist()
""")
code("""
params = dict(vector_size=100, window=3, min_count=1, epochs=200, seed=42, workers=1)

w2v_sg   = Word2Vec(sentences, sg=1, **params)   # Skip-Gram
w2v_cbow = Word2Vec(sentences, sg=0, **params)   # CBOW

print("어휘 수:", len(w2v_sg.wv.index_to_key))
result = pd.DataFrame({
    'Skip-Gram': [f"{w} ({s:.3f})" for w, s in w2v_sg.wv.most_similar('감염', topn=10)],
    'CBOW':      [f"{w} ({s:.3f})" for w, s in w2v_cbow.wv.most_similar('감염', topn=10)],
}, index=range(1, 11))
result
""")
code("""
# 해석 검증 1) '감염'과 실제로 같은 기술명에 함께 나온 단어(동시출현)
co = Counter(w for s in sentences if '감염' in s for w in set(s) if w != '감염')
print("동시출현:", co.most_common(8))

# 해석 검증 2) 유사어 상위 단어들이 쓰인 기술명 — '감염'과 같은 자리(무엇을 관리하는 시스템인가)에 오는지 확인
for w, _ in w2v_sg.wv.most_similar('감염', topn=3):
    print(f"\\n[{w}]", it.loc[it['토큰'].apply(lambda t: w in t), '기술명'].head(3).tolist())
""")
code("""
# 해석 검증 3) 코퍼스가 작으면 초기값(seed)에 따라 순위가 흔들린다 → seed 5개로 반복해 자주 등장하는 단어 확인
stable = Counter()
for s in range(5):
    m = Word2Vec(sentences, sg=1, **{**params, 'seed': s})
    stable.update(w for w, _ in m.wv.most_similar('감염', topn=10))
print("5회 중 Top10 등장 횟수:", stable.most_common(12))
""")
md("Q4_RESULT")

# ───────────────── Q5 ─────────────────
md("""
---
## Q5. 대출금액 최다 집행 은행 → 그 은행의 세부분류별 집행액
- '집행'은 **승인된 건의 대출금**으로 정의 (정수형 `대출금` 사용, '금액없음'은 0)
""")
code("""
executed = df[df['승인'] == 1]
bank = executed.groupby('신청은행')['대출금'].agg(합계='sum', 건수='count', 평균='mean')
bank['점유율'] = (bank['합계'] / bank['합계'].sum()).round(3)
bank = bank.sort_values('합계', ascending=False)
bank.head(8)
""")
code("""
top_bank = bank.index[0]
sub = (executed[executed['신청은행'] == top_bank]
       .groupby('헬스케어 세부분류')['대출금']
       .agg(합계='sum', 건수='count', 평균='mean', 최대='max')
       .sort_values('합계', ascending=False))
sub['비중'] = (sub['합계'] / sub['합계'].sum()).round(3)
print(f"최다 집행 은행: {top_bank}")
sub.head(10)
""")
md("Q5_RESULT")

# ───────────────── Q6 ─────────────────
md("""
---
## Q6. 기술명 + 대출금액 + 기술분류 + 신청은행 + 자본금 → 승인 여부 (Scikit-learn)
| 입력 | 전처리 |
|---|---|
| 기술명 (텍스트) | Okt 명사 → **TF-IDF** (희소표현) |
| 기술분류, 신청은행 (범주) | **One-Hot Encoding** |
| 대출금액, 자본금 (수치) | '금액없음' 결측 처리 → log 변환 → 표준화 |

- `ColumnTransformer`로 세 종류 전처리를 하나로 묶고 `Pipeline`으로 모델과 연결 → 교차검증 시 전처리도 fold 안에서만 학습(데이터 누수 방지)
- 불균형(승인 82%) → `class_weight='balanced'`, 평가지표는 정확도 외에 **AUC·균형정확도·거부 재현율** 함께 확인
""")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df['대출금액_num'] = pd.to_numeric(df['대출금액'], errors='coerce')   # '금액없음' → NaN
print("'금액없음' 9건의 진행상태:", df.loc[df['대출금액_num'].isna(), '진행상태'].value_counts().to_dict())

X = df[['토큰_str', '대출금액_num', '기술분류', '신청은행', '자본금']]
y = df['승인']

def signed_log1p(a):
    # 자본금 음수(자본잠식)도 처리하도록 부호 유지 로그
    return np.sign(a) * np.log1p(np.abs(a))

def make_pipeline(model, num_cols):
    pre = ColumnTransformer([
        ('tfidf', TfidfVectorizer(token_pattern=r'[^ ]+', min_df=2, ngram_range=(1, 2), sublinear_tf=True), '토큰_str'),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),   # 결측 + 결측여부 지시변수
                          ('log', FunctionTransformer(signed_log1p, feature_names_out='one-to-one')),
                          ('sc',  StandardScaler())]), num_cols),
        ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=5), ['기술분류', '신청은행']),
    ])
    return Pipeline([('pre', pre), ('model', model)])

models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=2000),
    'RandomForest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced',
                                           random_state=42, n_jobs=-1),
}
""")
code("""
# 5-fold 층화 교차검증 (승인/거부 비율을 fold마다 유지)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'AUC': 'roc_auc', '정확도': 'accuracy', '균형정확도': 'balanced_accuracy'}

rows = []
for name, model in models.items():
    s = cross_validate(make_pipeline(model, ['대출금액_num', '자본금']), X, y, cv=cv5, scoring=scoring)
    rows.append({'모델': name, **{k: round(s[f'test_{k}'].mean(), 3) for k in scoring}})
print("다수 클래스만 찍는 기준선: 정확도 0.817 / AUC 0.5 / 균형정확도 0.5")
pd.DataFrame(rows)
""")
code("""
# 최종 모델: 학습 80% / 평가 20% 분리 후 상세 성능 확인
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = make_pipeline(LogisticRegression(class_weight='balanced', max_iter=2000), ['대출금액_num', '자본금'])
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("AUC:", round(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]), 3))
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=['여신거부', '여신승인'], digits=3))
""")
md("""
### ⚠️ 점검: 성능이 너무 높다 → 대출금액이 '결과'를 담고 있는지 확인
대출금액은 심사 **결과로 집행된 금액**이라면 승인 여부를 예측하는 입력으로 쓸 수 없다(심사 시점에는 알 수 없는 정보 = 데이터 누수).
""")
code("""
print(pd.crosstab(df['대출금'] == 0, df['진행상태'], rownames=['대출금=0']))
rule = (df['대출금'] > 0).astype(int)
print("\\n'대출금 > 0 이면 승인' 한 줄 규칙의 정확도:", round((rule == y).mean(), 4))
""")
code("""
# 대출금액을 뺀 나머지 4개 입력(기술명, 기술분류, 신청은행, 자본금)만으로 다시 교차검증
rows = []
for name, model in models.items():
    s = cross_validate(make_pipeline(model, ['자본금']), X, y, cv=cv5, scoring=scoring)
    rows.append({'모델': name + ' (대출금액 제외)', **{k: round(s[f'test_{k}'].mean(), 3) for k in scoring}})
pd.DataFrame(rows)
""")
code("""
# 로지스틱 회귀 계수로 어떤 입력이 승인 판단을 이끄는지 확인 (전체 데이터 재학습)
clf.fit(X, y)
names = clf.named_steps['pre'].get_feature_names_out()
coef = pd.Series(clf.named_steps['model'].coef_[0], index=names)
print("승인 쪽 상위 계수:\\n", coef.sort_values(ascending=False).head(5).round(2))
print("\\n거부 쪽 상위 계수:\\n", coef.sort_values().head(5).round(2))
""")
md("Q6_RESULT")

# ───────────────── Q7 ─────────────────
md("""
---
## Q7. 기술명만으로 승인 여부 판별 — 신경망 모델 → `model_loan.h5`
1. **Text to Sequence**: Okt 명사 토큰 → Keras `Tokenizer`로 단어마다 정수 인덱스 부여 → 문장을 정수 시퀀스로
2. **Padding**: 길이가 제각각인 시퀀스를 `pad_sequences`로 같은 길이로 맞춤
3. **신경망**: `Embedding`(단어 → 밀집 벡터, 학습) → `LSTM`(단어 순서 반영) → `Dense(sigmoid)`(승인 확률)
4. 불균형 보정: `class_weight`, 과적합 방지: `Dropout` + `EarlyStopping`
""")
code("""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

texts = df['토큰_str'].values
labels = df['승인'].values

# 학습/검증/평가 분리 (Tokenizer는 학습 데이터로만 fit → 평가 데이터 단어 정보 누수 방지)
txt_tr, txt_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)

tokenizer = Tokenizer(oov_token='<OOV>')   # 학습에 없던 단어는 <OOV>(1번)로
tokenizer.fit_on_texts(txt_tr)
vocab_size = len(tokenizer.word_index) + 1   # +1: 패딩용 0번
print("단어 집합 크기:", vocab_size)

# 1) Text to Sequence
seq_tr = tokenizer.texts_to_sequences(txt_tr)
seq_te = tokenizer.texts_to_sequences(txt_te)
print("예시:", txt_tr[0], "→", seq_tr[0])

# 2) Padding: 문장 길이 분포를 보고 최대 길이 결정
lens = [len(s) for s in seq_tr]
print(f"토큰 수  평균 {np.mean(lens):.1f}, 최대 {max(lens)}, 95% 분위 {np.percentile(lens, 95):.0f}")
max_len = int(np.percentile(lens, 95))
X_tr_pad = pad_sequences(seq_tr, maxlen=max_len, padding='post', truncating='post')
X_te_pad = pad_sequences(seq_te, maxlen=max_len, padding='post', truncating='post')
print("패딩 후:", X_tr_pad.shape, X_tr_pad[0])
""")
code("""
# 3) 신경망 구성
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),   # mask_zero: 패딩 0은 계산에서 무시
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),                                   # 승인 확률
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()
""")
code("""
# 4) 학습: 거부(소수 클래스) 가중치를 키워 '전부 승인'으로 찍는 것을 방지
n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
class_weight = {0: len(y_tr) / (2 * n0), 1: len(y_tr) / (2 * n1)}
print("class_weight:", {k: round(v, 2) for k, v in class_weight.items()})

# 검증 손실(val_loss)이 3에폭 연속 나빠지면 멈추고, 가장 좋았던 가중치로 되돌림
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_tr_pad, y_tr, validation_split=0.2, epochs=30, batch_size=32,
                    class_weight=class_weight, callbacks=[es], verbose=2)
""")
code("""
# 5) 평가 (학습에 쓰지 않은 20%)
prob = model.predict(X_te_pad, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
print("AUC:", round(roc_auc_score(y_te, prob), 3))
print(confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=['여신거부', '여신승인'], digits=3))
""")
code("""
# 결과가 우연(데이터 분할 운)인지 확인: 같은 구조의 신경망을 5-fold 층화 교차검증으로 반복
def build_model():
    m = Sequential([Input(shape=(max_len,)),
                    Embedding(vocab_size, 64, mask_zero=True), LSTM(32), Dropout(0.5),
                    Dense(16, activation='relu'), Dense(1, activation='sigmoid')])
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

all_pad = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding='post', truncating='post')
cv_auc = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(all_pad, labels)):
    m = build_model()
    m.fit(all_pad[tr], labels[tr], validation_split=0.2, epochs=30, batch_size=32, class_weight=class_weight,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)], verbose=0)
    cv_auc.append(roc_auc_score(labels[va], m.predict(all_pad[va], verbose=0).ravel()))
print("fold별 AUC:", np.round(cv_auc, 3), "→ 평균", round(np.mean(cv_auc), 3))
# (참고: 여기서는 간단히 전체 데이터로 fit한 tokenizer를 재사용 — 단어→번호 매핑만 하므로 성능에 주는 영향은 작음)
""")
code("""
# 비교 기준: 같은 '기술명만' 입력을 TF-IDF + 로지스틱 회귀로 (신경망이 단순 모델보다 나은지 확인)
base = Pipeline([('tfidf', TfidfVectorizer(token_pattern=r'[^ ]+')),
                 ('lr', LogisticRegression(class_weight='balanced', max_iter=2000))]).fit(txt_tr, y_tr)
print("TF-IDF + 로지스틱 회귀 AUC:", round(roc_auc_score(y_te, base.predict_proba(txt_te)[:, 1]), 3))
""")
code("""
# 6) 모델 저장 (.h5 = HDF5 형식) + 새 기술명 예측에 필요한 Tokenizer도 함께 저장
import pickle
model.save('model_loan.h5')
with open('tokenizer_loan.pkl', 'wb') as f:
    pickle.dump({'tokenizer': tokenizer, 'max_len': max_len}, f)

# 저장한 모델을 다시 불러와 새 기술명으로 예측해보기
from tensorflow.keras.models import load_model
loaded = load_model('model_loan.h5')

def predict_loan(tech_name):
    seq = tokenizer.texts_to_sequences([' '.join(okt_nouns(tech_name))])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return float(loaded.predict(pad, verbose=0)[0, 0])

for name in ['AI 기반 감염관리 모니터링 시스템 개발', '홍삼 추출물을 함유하는 건강기능식품 조성물']:
    print(f"{name} → 승인 확률 {predict_loan(name):.3f}")
""")
code("""
# 코랩에서 파일 내려받기
files.download('model_loan.h5')
""")
md("Q7_RESULT")

md("SUMMARY")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}

import sys
results = {}
if len(sys.argv) > 1:   # 결과 해석 마크다운 채우기
    import json
    results = json.load(open(sys.argv[1]))
for c in nb.cells:
    if c.cell_type == "markdown" and c.source in results:
        c.source = results[c.source]
    elif c.cell_type == "markdown" and c.source.endswith("_RESULT") or c.source == "SUMMARY":
        c.source = "**결과 해석**: (실행 후 작성)"
nbf.write(nb, "mission4_tech_credit_심화.ipynb")
print("written", len(nb.cells), "cells")
