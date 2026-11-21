Test set: official OffensEval-TR 2020 test split, n = 3528 (20.3% offensive)

Latency hardware: arm | 14 logical cores | torch threads = 10 | Darwin
Latency: batch size 1, mean / p95 over 200 runs, budget = 50.0 ms

| key       | model                            |   macro_F1 |   OFF_F1 |   accuracy |   size_MB |   latency_ms_mean |   latency_ms_p95 | meets_budget   |
|:----------|:---------------------------------|-----------:|---------:|-----------:|----------:|------------------:|-----------------:|:---------------|
| lexicon   | Lexicon + edit distance (rules)  |     0.6212 |   0.4363 |     0.7115 |       0   |              0.02 |             0.06 | True           |
| tfidf     | tf-idf + Logistic Regression     |     0.7671 |   0.6299 |     0.8478 |       5.1 |              0.55 |             0.73 | True           |
| w2v_ffn   | word2vec + feed-forward NN       |     0.6768 |   0.4958 |     0.7781 |      23.7 |              0.05 |             0.09 | True           |
| bert_fp32 | BERTurk fine-tuned (FP32)        |     0.8122 |   0.6949 |     0.8855 |     442.5 |             28.43 |            34.34 | True           |
| bert_int8 | BERTurk dynamic INT8 (quantized) |     0.8178 |   0.7033 |     0.8897 |     186   |             15.51 |            28.28 | True           |