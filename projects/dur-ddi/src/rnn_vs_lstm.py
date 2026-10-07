"""SimpleRNN vs LSTM 하이퍼파라미터 실험 (노드 8/32/128, 층 1/2/3)."""
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from konlpy.tag import Okt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tf.keras.utils.set_random_seed(1234)

DATA = "data/의약품안전사용서비스(DUR)_통합_성분단위_2026.6.csv"
STOP = "data/stopword.txt"

dur = pd.read_csv(DATA)
im = dur[dur["금기유형"] == "임부금기"].copy()
im["금기등급"] = im["금기등급"].astype(str)
im = im[im["금기등급"].isin(["1", "2"])].copy()
im["y"] = (im["금기등급"] == "1").astype(int)

stopwords = [w.strip() for w in open(STOP, encoding="utf-8") if w.strip()]
if stopwords[0] == "불용어":
    stopwords = stopwords[1:]
okt = Okt()
im["명사"] = [" ".join([n for n in okt.nouns(str(t)) if len(n) > 1 and n not in stopwords])
             for t in im["상세정보"].fillna("")]

Xtr_txt, Xte_txt, Y_train, Y_test = train_test_split(
    im["명사"].values, im["y"].values, test_size=0.2, stratify=im["y"], random_state=1234)
tok = Tokenizer(oov_token="OOV")
tok.fit_on_texts(Xtr_txt)
vocab = len(tok.word_index) + 1
s_tr, s_te = tok.texts_to_sequences(Xtr_txt), tok.texts_to_sequences(Xte_txt)
max_len = int(np.percentile([len(s) for s in s_tr], 95))
X_train_vec = pad_sequences(s_tr, maxlen=max_len, padding="post", truncating="post")
X_test_vec = pad_sequences(s_te, maxlen=max_len, padding="post", truncating="post")
print(f"학습 {X_train_vec.shape} 검증 {X_test_vec.shape} 단어수 {vocab} maxlen {max_len}")

rows = []
for cell_name, Cell in (("SimpleRNN", SimpleRNN), ("LSTM", LSTM)):
    for nodes, layers in ((8, 1), (32, 1), (128, 1), (32, 2), (32, 3)):
        tf.keras.utils.set_random_seed(1234)
        m = Sequential([Input(shape=(max_len,)), Embedding(vocab, 64)])
        for i in range(layers):
            m.add(Cell(nodes, return_sequences=(i < layers - 1)))
        m.add(Dropout(0.5))
        m.add(Dense(1, activation="sigmoid"))
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        m.fit(X_train_vec, Y_train, epochs=10, batch_size=32,
              validation_data=(X_test_vec, Y_test), class_weight={0: 1, 1: 5}, verbose=0)
        sec = time.time() - t0
        ptr = (m.predict(X_train_vec, verbose=0) > 0.5).astype(int).ravel()
        pte = (m.predict(X_test_vec, verbose=0) > 0.5).astype(int).ravel()
        rows.append({"셀": cell_name, "노드": nodes, "층": layers,
                     "파라미터수": m.count_params(),
                     "순환층파라미터": m.count_params() - m.layers[0].count_params() - (nodes + 1),
                     "학습시간": round(sec, 1),
                     "학습정확도": round(accuracy_score(Y_train, ptr), 3),
                     "검증정확도": round(accuracy_score(Y_test, pte), 3),
                     "F1": round(f1_score(Y_test, pte), 3)})
        print(rows[-1], flush=True)

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("/private/tmp/claude-501/-Users-mac-Desktop-claude/2ddc187a-50b4-4341-88a2-e181eda8c01f/scratchpad/rnn_lstm.csv", index=False)
