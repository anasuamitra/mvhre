# MV-HRE: Multi-View Heterogeneous Relation Embedding
This is the official code-repository for the ICDM'22 paper [Revisiting Link Prediction on Heterogeneous Graphs with A Multi-view Perspective](https://dl.acm.org/doi/abs/10.1145/3447548.3467443).

## Motivation
<p align="center">
    <img width=600 src="views.png">
</p>
MV-HRE is a multi-view network representation learning framework to incorporate structural intuitions and enrich the triplet representations for link prediction on heterogeneous graphs. MV-HRE incorporates Metapath-view and rarely studied Community-view for the task of link
prediction in HINs besides local-contexts. It proposes first-of-its-kind Community-view for a triplet. It effectively aggregates multiple views at difference scales to provide enriched structural cues to predict links. Analysis of view importance suggests that all the chosen candidate views are indeed important and complementary in achieving the best performance on heterogeneous link prediction.

## Execution
To run MV-HRE, execute the sample command:
```python
python main.py --dataset <dataset> --clustering <0/1> --hidden_dim <hidden dimension> --lr <learning rate> --weight_decay <weight decay> --num_heads <number of heads> --num_layers <number of layers> --dropout <dropout> --context_hops <context_hops> --max_path_len <max_path_len> --path_samples <path_samples> --cluster_coeff <clustering coefficients> --num_clusters <number of clusters> --gpu_num <gpu_id>

Example: python main.py --dataset ddb --clustering 1 --hidden_dim 64 --lr 0.005 --weight_decay 0.001 --num_heads 2 --num_layers 4 --dropout 0.1 --context_hops 4 --max_path_len 5 --path_samples 5 --cluster_coeff 0.5 --num_clusters 25 --gpu 0
```
