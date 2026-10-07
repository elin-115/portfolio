import pandas as pd
PATH = 'data/SOMS_야전응급처치_타임라인_가상데이터.csv'
def load():
    df = pd.read_csv(PATH, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    for c in ['호흡수(회/분)','맥박수(회/분)','수축기혈압(mmHg)','인지→조치(초)','조치소요(초)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['환자키'] = df['시나리오ID'] + '_' + df['부상자ID']
    return df
