export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 3 \
--data_path "feature_extractor/graphs" \
--train_set "data/train_set.txt" \
--val_set "data/val_set.txt" \
--model_path "graph_transformer/saved_models" \
--log_path "graph_transformer/runs/" \
--task_name "GraphCAM" \
--resume "graph_transformer/saved_modelsGraphCAM.pth" \
--batch_size 8 \
--train \
--log_interval_local 6 \
