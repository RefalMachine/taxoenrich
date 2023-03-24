python train.py --thesaurus_dir /hdd_var/mtikhomi/work_folder/projects/taxoenrich_old/data/wordnets/RuWordNet \
--embeddings_path /hdd_var/mtikhomi/work_folder/projects/taxoenrich_old/data/vectors/ft_extended_vocab_20_11_20.wv \
--output_path data/test_1_23_03_23 \
--wkt --wiktionary_dump_path /hdd_var/mtikhomi/work_folder/projects/taxoenrich_old/data/wiki/ruwiktionary-20210201-pages-articles-multistream.xml \
--lang ru --pos N --processes 8 --search_by_word --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.2

python predict.py --model_dir data/test_1_23_03_23 --input_path /hdd_var/mtikhomi/work_folder/projects/taxoenrich_old/data/tasks/ru_private_nouns.tsv \
--output_path data/ru_private_nouns_predict.tsv

python eval.py --predict_path data/ru_private_nouns_predict.tsv --reference_path /hdd_var/mtikhomi/work_folder/projects/taxoenrich_old/data/tasks/nouns_private_subgraphs.tsv