time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
--model CAEME --epochs 20 --result_path ../../data/models/allin_1152_oent_caeme_triplet_16_04_21.wv \
--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 5 2 --ruthes