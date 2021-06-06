#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_e20_1_22_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_e20_2_22_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_e20_3_22_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/vectors/glove_300_13_01_21.wv  ../../data/vectors/w2v_w5_nlpl_300_07_01_21.wv \
#--model CONCAT --epochs 20 --result_path w2v_ft_glove_eng_ext_concat_cos_loss_e20_26_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/vectors/glove_300_13_01_21.wv  ../../data/vectors/w2v_w5_nlpl_300_07_01_21.wv \
#--model SVD --epochs 20 --result_path w2v_ft__glove_eng_ext_svd_cos_loss_e20_26_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/vectors/glove_300_13_01_21.wv  ../../data/vectors/w2v_w5_nlpl_300_07_01_21.wv \
#--model CAEME --epochs 20 --result_path w2v_ft__glove_eng_ext_caeme_cos_loss_e20_26_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &
#
#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_caeme_cos_loss_constrains_e20_28_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../data/RuWordNet --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/vectors/glove_300_13_01_21.wv  ../../data/vectors/w2v_w5_nlpl_300_07_01_21.wv \
#--model CAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_caeme_cos_loss_constrains_e20_28_01_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &


#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_constrains_a0.01_c3_26_03_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.01 --logging_every_epochs 1 --constrains_count 3

#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_constrains_a0.01_c5_26_03_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 0.01 --logging_every_epochs 1 --constrains_count 5

#python run.py --vectors ../../../taxonomy_enrichment_export/data/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_ext_aaeme_cos_loss_constrains_cos_a1.0_c5_26_03_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../../taxonomy_enrichment_export/data/ruwordnet --alpha 1.0 --logging_every_epochs 1 --constrains_count 5


#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_aaeme_cos_loss_e20_25_02_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/WordNet-1.6 --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model CAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_caeme_cos_loss_e20_25_02_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/WordNet-1.6 --alpha 0.5 --logging_every_epochs 1 --constrains_count 5 &

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/allin_1111_oent_aaeme_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/concepts_lite.json --alpha 10.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 1 1

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/allin_1122_oent_aaeme_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/concepts_lite.json --alpha 10.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 2 2

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/allin_1133_oent_aaeme_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/concepts_lite.json --alpha 10.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 3 3

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/allin_1152_oent_aaeme_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/concepts_lite.json --alpha 10.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 5 2

#time python run.py --vectors ../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model CAEME --epochs 20 --result_path ../../data/models/w2v_ft_ext_caeme_26_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25 --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

#time python run.py --vectors ../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model CAEME --epochs 20 --result_path ../../data/models/w2v_ft_ext_caeme_triplet_26_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
--model CAEME --epochs 20 --result_path ../../data/models/allin_1152_oent_caeme_triplet_16_04_21.wv \
--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 5 2 --ruthes

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model CAEME --epochs 20 --result_path ../../data/models/allin_1152_oent_caeme_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 5 2

#time python run.py --vectors ../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model CONCAT --epochs 20 --result_path ../../data/models/w2v_ft_ext_concat_26_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25 --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#--model CONCAT --epochs 20 --result_path ../../data/models/w2v_ft_inner_concat_26_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model CONCAT --epochs 20 --result_path ../../data/models/allin_concat_26_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

#time python run.py --vectors ../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv ../../data/models/ft_extended_vocab_20_11_20.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/w2v_ft_ext_ru_aaeme_a100_11_25_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/RuWordNet --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#--model SVD --epochs 20 --result_path ../../data/models/w2v_ft_inner_svd_16_04_21.wv \
#--lr 2e-4 --emb_dim 450 --dev_size 0.25  --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 200.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 --ruthes

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model SVD --epochs 20 --result_path ../../data/models/allin_svd_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/concepts_lite.json --alpha 10.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 1 1

#time python run.py --vectors ../../data/models/w2v_cybersec_w3_v300_e10_13_04_21_gensim.wv  ../../data/models/ft_cybersec_w3_v300_e10_13_04_21_gensim.wv \
#../../data/models/araneum_w2v_rusvectores_28_08_20_gensim.wv  ../../data/models/ft_ext_ruthes_16_03_21.wv \
#--model AAEME --epochs 20 --result_path ../../data/models/allin_1152_oent_aaeme_triplet_a200_16_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25 --thes_constrains --thes_path ../../data/concepts_lite.json --alpha 200.0 --logging_every_epochs 1 --constrains_count 5 --lang ru \
#--margin 0.1 --distance_type mse --num_workers 8 --wv_weights 1 1 5 2 --ruthes


#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model CAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_caeme_cos_loss_constrains_mse_m0.1_c5_a50_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 50.0 --logging_every_epochs 1 --constrains_count 5 --lang en \
#--margin 0.1 --distance_type mse --num_workers 8

#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model CAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_caeme_cos_loss_constrains_mse_m0.1_c5_a100_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang en \
#--margin 0.1 --distance_type mse --num_workers 8

#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_aaeme_cos_loss_constrains_mse_m0.1_c10_a200_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 200.0 --logging_every_epochs 1 --constrains_count 10 --lang en \
#--margin 0.1 --distance_type mse --num_workers 8

#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_aaeme_cos_loss_constrains_mse_m0.2_c5_a50_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 50.0 --logging_every_epochs 1 --constrains_count 5 --lang en \
#--margin 0.2 --distance_type mse --num_workers 8

#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_aaeme_cos_loss_constrains_mse_m0.2_c5_a100_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 100.0 --logging_every_epochs 1 --constrains_count 5 --lang en \
#--margin 0.2 --distance_type mse --num_workers 8

#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_aaeme_cos_loss_constrains_mse_m0.2_c5_a200_03_04_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_constrains --thes_path ../../data/WordNet-1.6 --alpha 200.0 --logging_every_epochs 1 --constrains_count 5 --lang en \
#--margin 0.2 --distance_type mse --num_workers 8


#time python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model AAEME --epochs 20 --result_path w2v_ft_glove_eng_ext_caeme_cos_loss_29_03_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --thes_path ../../data/WordNet-1.6 --alpha 1.0 --logging_every_epochs 1 --constrains_count 5 --lang en
#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model CONCAT --epochs 20 --result_path w2v_ft_glove_eng_ext_concat_22_02_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --logging_every_epochs 1 &

#python run.py --vectors ../../data/vectors/ft_v2_07_01_21.wv  ../../data/glove_300_vocab_ext_22_02_21.wv  ../../data/w2v_w5_nlpl_300_vocab_ext_22_02_21.wv \
#--model SVD --epochs 20 --result_path w2v_ft_glove_eng_ext_svd_22_02_21.wv \
#--lr 2e-4 --emb_dim 600 --dev_size 0.25  --logging_every_epochs 1 &

