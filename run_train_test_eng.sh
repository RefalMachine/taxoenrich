python train_hypernym_predictor.py --train_path data/tasks/train_nouns_en_1.6_0.05_14_01_21.tsv \
--task_json_path configs/nouns_en_ft_wkt.json \
--model_save_path data/models/nouns_en_ft_wkt_model.bin \
--thesaurus_dir data/wordnets/WordNet-1.6 

# 1.6-3.0
python predict_hypernym_predictor.py --test_path data/tasks/no_labels_nouns_en.1.6-3.0.tsv \
--model_path data/models/nouns_en_ft_wkt_model.bin \
--task_json_path configs/nouns_en_ft_wkt.json \
--thesaurus_dir data/wordnets/WordNet-1.6 \
--save_path data/results/nouns_1.6-3.0_en_ft_wkt.tsv

python eval_hypernym_predictor.py --predict_path data/results/nouns_1.6-3.0_en_ft_wkt.tsv \
--reference_path data/tasks/nouns_en.1.6-3.0.tsv

# 1.7-3.0
python predict_hypernym_predictor.py --test_path data/tasks/no_labels_nouns_en.1.7-3.0.tsv \
--model_path data/models/nouns_en_ft_wkt_model.bin \
--task_json_path configs/nouns_en_ft_wkt.json \
--thesaurus_dir data/wordnets/WordNet-1.7 \
--save_path data/results/nouns_1.7-3.0_en_ft_wkt.tsv

python eval_hypernym_predictor.py --predict_path data/results/nouns_1.7-3.0_en_ft_wkt.tsv \
--reference_path data/tasks/nouns_en.1.7-3.0.tsv

# 2.0-3.0
python predict_hypernym_predictor.py --test_path data/tasks/no_labels_nouns_en.2.0-3.0.tsv \
--model_path data/models/nouns_en_ft_wkt_model.bin \
--task_json_path configs/nouns_en_ft_wkt.json \
--thesaurus_dir data/wordnets/WordNet-2.0 \
--save_path data/results/nouns_2.0-3.0_en_ft_wkt.tsv

python eval_hypernym_predictor.py --predict_path data/results/nouns_2.0-3.0_en_ft_wkt.tsv \
--reference_path data/tasks/nouns_en.2.0-3.0.tsv