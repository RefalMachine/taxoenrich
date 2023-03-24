call  C:\ProgramData\Anaconda\Scripts\activate.bat

python train.py --thesaurus_dir D:\WorkFolder\data\models\RuWordNet ^
--embeddings_path D:\WorkFolder\emb\ft_extended_vocab_20_11_20.wv --output_path data\models\test_1_23_03_23 ^
--lang ru --pos N --processes 8 --search_by_word --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.1
python predict.py --model_dir data/models/test_1_23_03_23 --input_path data/tasks/ru_private_nouns.tsv --output_path data/res/ru_private_nouns_predict.tsv
python eval.py --predict_path data/res/ru_private_nouns_predict.tsv --reference_path data/tasks/nouns_private_subgraphs.tsv

python train.py --thesaurus_dir D:\WorkFolder\data\models\RuWordNet ^
--embeddings_path D:\WorkFolder\emb\ft_extended_vocab_20_11_20.wv --output_path data\models\test_2_23_03_23 ^
--lang ru --pos N --processes 8 --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.1
python predict.py --model_dir data/models/test_2_23_03_23 --input_path data/tasks/ru_private_nouns.tsv --output_path data/res/ru_private_nouns_predict.tsv
python eval.py --predict_path data/res/ru_private_nouns_predict.tsv --reference_path data/tasks/nouns_private_subgraphs.tsv