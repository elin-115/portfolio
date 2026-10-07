# -*- coding: utf-8 -*-
"""TCCC MARCH-PAWS 규칙 엔진.

이 데이터의 '다음권고조치'는 '조치내용'의 결정론적 함수(623/623 단일값)라 학습 타깃으로
쓸 수 없다. 대신 최우선 조치는 표준 프로토콜로 구현하고, 데이터는 그 규칙의 검증에만 쓴다.

MARCH는 '활력징후를 보고 무엇을 할지 고르는' 분류 문제가 아니라, 환자가 필요로 하는 처치를
정해진 우선순위대로 소진하는 순서 프로토콜이다. 엔진도 그 구조를 따른다.
"""

# TCCC MARCH-PAWS 우선순위. 숫자가 작을수록 먼저.
PRIORITY = {
    'M': 1,   # Massive hemorrhage  대량출혈
    'A': 2,   # Airway              기도
    'R': 3,   # Respiration         호흡
    'C': 4,   # Circulation         순환
    'H': 5,   # Hypothermia/Head    저체온증·열손상
    'W': 6,   # Wounds/Splinting    창상·골절·화상
    'P': 7,   # Pain                통증
}
# 임상 처치가 아닌 흐름 코드 — 우선순위 판정에서 제외
FLOW = {'T', 'RA', 'E', 'D'}

TYPE_TO_MARCH = {          # 데이터에서 실측한 매핑 (유형 → MARCH, 100% 결정론)
    '대량출혈': 'M', '기도': 'A', '호흡': 'R', '순환': 'C',
    '저체온증': 'H', '열손상': 'H',
    '창상': 'W', '골절': 'W', '화상': 'W',
    '통증': 'P', '제독': 'D', '재평가': 'RA', '환자분류': 'T', '후송': 'E',
}

# --- 환자가 어떤 MARCH 단계를 필요로 하는가 (부상 정보·활력징후 기반) ---
MECH_M = ('총상', '파편', '절단', '열상', '압궤', '낙하물')          # 대량출혈 기전
SITE_A = ('기도', '안면', '경부', '두경부', '두부')                  # 기도 위협 부위
SITE_R = ('흉부', '늑골')                                        # 호흡 손상 부위

def needs(mechanism='', site='', avpu='A', rr=None, pr=None, sbp=None):
    """환자 상태에서 필요한 MARCH 단계 집합을 추정한다."""
    n = set()
    if any(k in mechanism for k in MECH_M):           n.add('M')
    if avpu in ('P', 'U') or any(k in site for k in SITE_A): n.add('A')
    if (rr is not None and (rr < 10 or rr > 30)) or any(k in site for k in SITE_R): n.add('R')
    if (sbp is not None and sbp < 90) or (pr is not None and pr > 120):             n.add('C')
    if '저체온' in mechanism or '열손상' in mechanism or '연기흡입' in mechanism:      n.add('H')
    if '화상' in mechanism or '둔상' in mechanism or site:                           n.add('W')
    return n

def next_action(done, phase='TFC', required=None):
    """완료된 처치(done)와 TCCC 단계(phase)를 받아 다음 최우선 조치를 반환한다.

    CUF(교전 중·위험구역)에서는 TCCC상 대량출혈 지혈 외 처치를 하지 않는다.
    """
    required = required or set(PRIORITY)
    if phase == 'CUF':
        return 'M' if 'M' not in done else 'E'        # 지혈 끝나면 엄폐·후송
    todo = [c for c in required if c in PRIORITY and c not in done]
    if todo:
        return min(todo, key=lambda c: PRIORITY[c])
    return 'RA' if 'RA' not in done else 'E'

def violates_order(sequence):
    """기록된 처치 시퀀스가 MARCH 우선순위를 역행하는 지점을 찾는다."""
    clinical = [(i, c) for i, c in enumerate(sequence) if c in PRIORITY]
    bad = []
    for (i, a), (j, b) in zip(clinical, clinical[1:]):
        if PRIORITY[b] < PRIORITY[a]:
            bad.append((i, a, j, b))
    return bad
