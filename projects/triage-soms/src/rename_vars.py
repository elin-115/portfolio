# -*- coding: utf-8 -*-
"""한글 변수명 → 영어. tokenize 로 NAME 토큰만 바꾸므로 문자열 안 컬럼명은 안전하다."""
import tokenize, io, keyword

MAP = {
 # 데이터프레임 / 목록
 '활력':'VITALS', '경로':'path', '공공':'pub', '분류시점':'triage', '자료':'data',
 '학습':'train_df', '검증':'test_df', '군집자료':'cl_data', '맞춘자료':'cl_scaled',
 # 전처리
 '영':'all_zero', '의미':'na_meaning', '뜻':'meaning', '초로_바꾸기':'to_seconds',
 '문자':'text', '부분':'parts', '날짜':'date_col', '자정넘음':'past_midnight',
 '계산':'calc', '점수표':'avpu_map', '첫발견':'first_seen',
 '쪼갠것':'split_list', '모든기전':'all_mech', '목록':'items', '기전':'mech',
 '빈도':'freq', '사용할기전':'keep_mech',
 # 탐색·검정
 '순서':'ORDER', '교차표':'ctab', '비율':'ratio', '유형별':'by_type', '표':'table',
 '카이':'chi2', 'p값':'p_val', '자유도':'dof', '기대값':'expected',
 '구역별':'by_zone', '역할':'role', '해당':'sub', '상위':'top3', '설명':'desc',
 '표본':'sample', '정규p':'p_norm', '집단':'groups', '크루스칼p':'p_kw', '정규여부':'norm_txt',
 '상관':'corr', '케이민즈':'kmeans', '군집번호':'cl_id', '결과표':'cl_df',
 # 모델링
 '입력열':'FEATURES', '목표열':'TARGET', '템플릿목록':'templates', '검증템플릿':'test_templates',
 '모델들':'models', '이름':'name', '모델':'model', '예측':'pred', '결과':'preds',
 '숫자로':'to_num', '글자로':'to_label', '등급순서':'LEVEL', '실제숫자':'y_num', '예측숫자':'p_num',
 '과소':'under', '과대':'over', '적색실제':'is_red', '적색놓침':'red_miss',
 '최종':'best', '행렬':'cm', '숲':'rf', '중요도':'importance',
 '상호':'mi', '순열':'perm', '비교':'compare',
 # 규칙
 'start_분류':'start_triage', '의식':'avpu', '호흡':'rr', '맥박':'pr', '혈압':'sbp',
 '규칙예측':'rule_pred', '결':'res', '모델예측':'model_pred', '결합예측':'combo_pred',
 '규칙점수':'rule_lv', '모델점수':'model_lv', '높은쪽':'higher', '거꾸로':'back',
 '비교표':'summary',
 # 기타
 '시스템':'os_name', '열':'col', '개수':'n_uniq',
 'X_전체':'X_all', 'y_전체':'y_all', '그룹':'groups_all', '숲2':'rf2',
 '예측_전체':'pred_all', '행렬2':'cm2',
}
assert not any(keyword.iskeyword(v) for v in MAP.values())

def rename(src):
    edits = []                                   # (row, col_start, col_end, new)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in MAP:
            edits.append((tok.start[0], tok.start[1], tok.end[1], MAP[tok.string]))
    lines = src.split('\n')
    for row, c0, c1, new in sorted(edits, reverse=True):   # 뒤에서부터 고쳐야 위치가 안 밀림
        L = lines[row-1]
        lines[row-1] = L[:c0] + new + L[c1:]
    return '\n'.join(lines)

if __name__ == '__main__':
    src = open('soms_analysis.py', encoding='utf-8').read()
    out = rename(src)
    open('soms_analysis.py', 'w', encoding='utf-8').write(out)
    # 남은 한글 식별자 확인
    import re
    left = set()
    for tok in tokenize.generate_tokens(io.StringIO(out).readline):
        if tok.type == tokenize.NAME and re.search(r'[가-힣]', tok.string):
            left.add(tok.string)
    print('변환 완료. 남은 한글 이름:', sorted(left) if left else '없음')
