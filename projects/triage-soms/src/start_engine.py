# -*- coding: utf-8 -*-
"""START(Simple Triage And Rapid Treatment) 분류 규칙 엔진 — 공공 다수사상자 표준.

변수정의서가 `분류색상`을 "START/MASCAL 환자분류 색상"으로 정의하고 있으므로,
중증도 분류도 학습 모델이 아니라 표준 알고리즘으로 구현하고 데이터로 검증한다.

원 알고리즘은 보행가능 여부와 모세혈관 재충혈(CRT)을 쓰지만 이 데이터에는 없다.
보행가능 → AVPU=A + 활력징후 정상, CRT>2초 → 수축기혈압 < 90 으로 대치했다.
"""

def start_triage(avpu, rr, pr, sbp):
    """활력징후에서 START 분류색상을 반환한다."""
    if rr is None or rr == 0:                       # 기도 개방 후에도 무호흡
        return '흑색'
    if rr > 30 or rr < 10:                          # 호흡수 이상
        return '적색'
    if sbp is not None and sbp < 90:                # 관류 불량 (CRT>2초 대치)
        return '적색'
    if pr is not None and pr > 120:                 # 빈맥
        return '적색'
    if avpu in ('P', 'U'):                          # 명령 수행 불가
        return '적색'
    if avpu == 'A' and 10 <= rr <= 26 and (pr or 0) <= 110 and (sbp or 999) >= 95:
        return '녹색'                                # 보행 가능군 대치
    return '황색'

ORDER = {'녹색': 0, '황색': 1, '적색': 2, '흑색': 3}
