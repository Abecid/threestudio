# Running zero123 to gather preference data
1. Download checkpoint
```
cd load/zero123/
bash download.sh
```

2. Run the following command
```
CUDA_VISIBLE_DEVICES=0 python run.py --train
```