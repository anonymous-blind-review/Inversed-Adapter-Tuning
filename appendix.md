# Appendix

## Performance on other BEIR datasets

Some datasets are exluded from the experiments in the paper due to the following reasons:
- Some datasets are already used to pretrain the BERT model, which is the backbone of our Dense Retrieval (DR) model. However, in our paper, the target scenario is that the vendors cannot access the target corpus and thus cannot pretrain the model on it. Therefore, we exclude these datasets from the experiments.
- Some datasets are not suitable for ranking evaluation. For example, the ArguAna dataset is designed for the task of **counter**-argument retrieval. Previous study ([link](https://arxiv.org/pdf/2204.11447.pdf)) has shown that the ranking evaluation on this dataset is not meaningful. Therefore, we exclude these datasets from the experiments.

Nevertheless, we report the NDCG@10 performance of Dense Retrieval (DR) with and without IAT on these datasets in the following table.

| Dataset  | Exclusion Reason | DR | DR + IAT |
| :---:    | :---:  | :---: | :---: |
| NQ       | Included in BERT pretraining corpus | 0.473 | 0.479 |
| HotpotQA | Included in BERT pretraining corpus | 0.590 | 0.593 |
| DBPedia | Included in BERT pretraining corpus | 0.395 | 0.392 |
| FEVER | Included in BERT pretraining corpus | 0.703 | 0.700 |
| Climate-FEVER | Included in BERT pretraining corpus | 0.228 | 0.234 |
| ArguAna | Not suitable for ranking evaluation ([link](https://arxiv.org/pdf/2204.11447.pdf)) | 0.432 | 0.440 |
| Touche-2020 | Not suitable for ranking evaluation ([link](https://arxiv.org/pdf/2204.11447.pdf)) | 0.165 | 0.170 |


We can see that IAT does not harm the performance on these datasets. Therefore, deploying IAT for neural-ranking-as-a-service is safe when the corpus is already included in the pretraining corpus or when the task is different from web search.
