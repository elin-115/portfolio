# -*- coding: utf-8 -*-
"""분석 계획서 1단계 — 데이터 정제 및 파생 변수 생성.

실행: python soms/preprocess.py
출력: soms/clean.csv (정제 데이터), 콘솔에 처리 리포트

정제 원칙
  · 활력징후 0 은 '측정불가'를 0으로 기록한 값 → 결측(NaN) + 플래그로 분리
  · '-' 는 결측이 아니라 구조적 '해당없음' → 명시 범주로 유지하고 플래그 생성
  · 시각은 날짜와 결합해 datetime 으로 변환, 자정 경과 구간 보정
  · 복합 범주는 구분자로 분해해 다중 원-핫 생성
"""
import os, sys, re
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = 'data/SOMS_야전응급처치_타임라인_가상데이터.csv'
VITALS = ['호흡수(회/분)', '맥박수(회/분)', '수축기혈압(mmHg)']
MIL    = {'전투', '훈련사고'}

# '-' 가 '해당없음'을 뜻하는 컬럼과 그 의미
NA_MEANING = {
    '후송우선순위': '미결정', '후송수단': '해당없음', '부상부위': '부위무관',
    '손등번호': '미부여',   '다음권고조치': '없음',  '음성근거': '발화없음',
}
# 복합 범주 컬럼과 구분자
MULTI = {'부상기전': '+', '사용도구': ', ', '음성근거': ' / ', '부상부위': '·'}

# 중증도 분류 시점 이후 생성되어 정답을 누설하는 컬럼
LEAK_TRIAGE = ['환자평가', '유형', 'MARCH코드', '조치내용', '조치결과', '사용도구',
               '다음권고조치', '손등번호', '후송우선순위', '후송수단',
               '인지→조치(초)', '조치소요(초)']
# 1:1 대응 쌍 — 각 쌍에서 규칙 엔진이 쓰는 쪽만 남긴다
DUP_PAIRS = [('분류색상', '환자평가'), ('MARCH코드', '유형')]

AVPU_SCORE = {'A': 3, 'V': 2, 'P': 1, 'U': 0}


def load_raw():
    return pd.read_csv(RAW, encoding='utf-8-sig', dtype=str, keep_default_na=False)


def _hms(t):
    """'HH:MM:SS' 또는 'MM:SS' → 초. 분/시가 60을 넘어도 그대로 누적 처리."""
    p = [int(v) for v in t.split(':')]
    return p[0]*3600 + p[1]*60 + (p[2] if len(p) > 2 else 0) if len(p) > 2 \
        else p[0]*60 + p[1]


def clean(df, log):
    d = df.copy()

    # ── 수치형 변환 ──
    for c in VITALS + ['인지→조치(초)', '조치소요(초)']:
        d[c] = pd.to_numeric(d[c], errors='coerce')

    # ── 활력징후 0 = 측정불가 → 결측 + 플래그 ──
    zero = (d[VITALS] == 0).all(axis=1)
    d['활력징후_측정불가'] = zero
    d.loc[zero, VITALS] = np.nan
    log.append(('활력징후 0 → 결측 처리', f'{int(zero.sum())}행 '
                f'(분류: {d.loc[zero, "분류색상"].value_counts().to_dict()})'))

    # ── '-' 플레이스홀더 → 의미 있는 범주 + 플래그 ──
    for c, meaning in NA_MEANING.items():
        m = d[c] == '-'
        d[f'{c}_해당없음'] = m
        d.loc[m, c] = meaning
        log.append((f"'-' 정규화 : {c}", f'{int(m.sum())}행 → "{meaning}"'))

    # ── 비고: 공란 다수 → 명시 범주 ──
    blank = d['비고'].str.strip() == ''
    d.loc[blank, '비고'] = '없음'
    log.append(('비고 공란 → "없음"', f'{int(blank.sum())}행'))

    # ── 시각 → datetime (자정 경과 보정) ──
    base = pd.to_datetime(d['날짜'], format='%Y-%m-%d')
    for c, new in [('식별시간', '식별_dt'), ('조치시작시간', '조치시작_dt'), ('조치완료시간', '조치완료_dt')]:
        d[new] = base + pd.to_timedelta(d[c].map(_hms), unit='s')
    rollover = d['조치완료_dt'] < d['조치시작_dt']
    d.loc[rollover, '조치완료_dt'] += pd.Timedelta(days=1)
    log.append(('자정 경과 보정', f'{int(rollover.sum())}행 (조치완료 < 조치시작)'))

    # ── 영상 구간 → 초 ──
    d['영상_시작_초'] = d['영상_시작'].map(_hms)
    d['영상_종료_초'] = d['영상_종료'].map(_hms)
    d['영상_길이_초'] = d['영상_종료_초'] - d['영상_시작_초']

    # ── 기록값 대비 재계산 검증 ──
    calc = (d['조치시작_dt'] - d['식별_dt']).dt.total_seconds()
    gap = (calc - d['인지→조치(초)']).abs()
    log.append(('인지→조치(초) 재계산 대조', f'불일치 {int((gap > 1).sum())}행'))
    calc2 = (d['조치완료_dt'] - d['조치시작_dt']).dt.total_seconds()
    gap2 = (calc2 - d['조치소요(초)']).abs()
    log.append(('조치소요(초) 재계산 대조', f'불일치 {int((gap2 > 1).sum())}행'))
    return d


def add_derived(d, log):
    d['AVPU_점수']   = d['의식수준(AVPU)'].map(AVPU_SCORE)
    d['의식저하']     = d['의식수준(AVPU)'].isin(['P', 'U'])
    d['쇼크지수']     = (d['맥박수(회/분)'] / d['수축기혈압(mmHg)']).round(3)
    d['쇼크의심']     = d['쇼크지수'] > 0.9                      # 임상 통용 기준
    d['호흡이상']     = (d['호흡수(회/분)'] < 10) | (d['호흡수(회/분)'] > 30)
    d['저혈압']       = d['수축기혈압(mmHg)'] < 90
    d['빈맥']         = d['맥박수(회/분)'] > 120
    d['야간']         = d['조명환경'].str.contains('야간|무조명|조명탄|헤드랜턴')
    d['이동이벤트']   = d['장소'].str.contains('→')
    d['월']           = pd.to_datetime(d['날짜']).dt.month
    d['요일']         = pd.to_datetime(d['날짜']).dt.dayofweek
    log.append(('파생 변수 생성', 'AVPU_점수 · 쇼크지수 · 쇼크의심 · 호흡이상 · 저혈압 · '
                                  '빈맥 · 야간 · 이동이벤트 · 월 · 요일'))
    log.append(('쇼크의심 비율', f'{d["쇼크의심"].mean()*100:.1f}%  '
                f'(쇼크지수 중앙값 {d["쇼크지수"].median():.2f})'))
    return d


def add_strata(d, log):
    """층별 분석용 키 — 재난유형 · 구역 · 역할 · 영역"""
    d['영역']   = np.where(d['상황구분'].isin(MIL), '군', '공공')
    d['재난유형'] = d['상황']
    d['구역']   = d['구분'].map({'CUF': 'Hot Zone', 'TFC': 'Warm Zone', 'TEC': 'Cold Zone'})
    d['역할군'] = np.select(
        [d['조치자_역할'].str.contains('구급|구조'),
         d['조치자_역할'].str.contains('DMAT|전문의|간호사|소장'),
         d['조치자_역할'].str.contains('이송')],
        ['현장 처치', '의료 지휘', '이송'], default='기타')
    d['환자키'] = d['시나리오ID'] + '_' + d['부상자ID']
    log.append(('층별 키 생성', f"영역 {d['영역'].nunique()} · 재난유형 {d['재난유형'].nunique()} · "
                f"구역 {d['구역'].nunique()} · 역할군 {d['역할군'].nunique()}"))
    return d


def explode_multi(d, col, sep=None, prefix=None, min_count=30):
    """복합 범주를 분해해 다중 원-핫 컬럼을 만든다."""
    sep = sep or MULTI[col]
    parts = d[col].str.split(re.escape(sep), regex=True)
    vocab = pd.Series([v.strip() for lst in parts for v in lst]).value_counts()
    keep = vocab[vocab >= min_count].index
    out = pd.DataFrame({f'{prefix or col}__{v}': parts.map(lambda l, v=v: v in [x.strip() for x in l])
                        for v in keep}, index=d.index)
    return out, len(keep), len(vocab)


def feature_set(task='triage'):
    """과제별 입력 컬럼. 누수·중복은 사전 배제한다."""
    if task == 'triage':                       # 중증도 분류
        return ['의식수준(AVPU)', 'AVPU_점수'] + VITALS + ['쇼크지수'], '분류색상'
    if task == 'march':                        # 필요 처치단계 추정
        return ['의식수준(AVPU)'] + VITALS + ['부상기전', '부상부위', '재난유형',
                '위협수준', '조명환경'], 'MARCH코드'
    raise ValueError(task)


def run(save=True):
    log = []
    raw = load_raw()
    log.append(('원본', f'{len(raw):,}행 × {raw.shape[1]}열'))
    d = clean(raw, log)
    d = add_derived(d, log)
    d = add_strata(d, log)

    # 복합 범주 원-핫 (참고용 — 필요 시 조인)
    for col in ('부상기전', '사용도구'):
        oh, kept, total = explode_multi(d, col)
        log.append((f'{col} 다중 원-핫', f'{total}종 중 30건 이상 {kept}종'))
        d = d.join(oh)

    print(f"\n{'='*70}\n전처리 리포트\n{'='*70}")
    for k, v in log:
        print(f"  {k:<24} {v}")
    print(f"\n  결과 : {len(d):,}행 × {d.shape[1]}열 "
          f"(원본 대비 +{d.shape[1]-raw.shape[1]}열)")
    print(f"  영역 : {d['영역'].value_counts().to_dict()}")
    print(f"  누수 배제 대상 {len(LEAK_TRIAGE)}개 · 중복 쌍 {len(DUP_PAIRS)}쌍")
    print(f"  결측 보유 컬럼 : {d.columns[d.isna().any()].tolist()}")
    if save:
        out = os.path.join(HERE, 'clean.csv')
        d.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"\n  저장 : {out}")
    print('='*70)
    return d


if __name__ == '__main__':
    run()
