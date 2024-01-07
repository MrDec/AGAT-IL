# STAGAM
SAGAT: Attributed Graph Clustering Based on Self-training Adaptive Graph Attention Networks with Laplacian Filter

```
python (tested on 3.9.12)
pytorch (tested on 1.11.0)
```

Then, run the command:
```
pip install -r requirements.txt
```
## Run

Run SAGAT on Cora dataset:
```
python adpativeGAT.py --dataset cora --upth_st 0.011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5
```

