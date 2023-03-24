python train.py --thesaurus_dir data/models/RuWordNet \
--embeddings_path data/emb/araneum_fasttextskipgram_14_12_21.wv --output_path data/models/test_1_23_03_23 \
--lang ru --pos N --processes 8 --search_by_word --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.1

python predict.py --model_dir data/models/test_1_23_03_23 \
--lang ru --pos N --processes 8 --search_by_word --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.1