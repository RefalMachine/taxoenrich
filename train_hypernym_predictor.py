import argparse
import pandas as pd
import codecs
import json
import gensim
import os
import joblib
from taxoenrich.models import EnWordNet, RuWordNet, RuThes
from taxoenrich.utils import VectorsWithHash, calc_most_sim, load_word2patterns, \
    get_synset_candidates, get_train_true_info, get_features_df, train_predict_cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--task_json_path', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--thesaurus_dir', type=str)
    parser.add_argument('--ruthes', action='store_true')
    args = parser.parse_args()

    config = json.load(codecs.open(args.task_json_path, 'r', 'utf-8'))
    if config['lang'] == 'en':
        import wiktionary_processing.utils_en as wkt
    else:
        import wiktionary_processing.utils as wkt
        

    train_df = pd.read_csv(args.train_path).rename(columns={'target_word': 'word'})
    #print(train_df.head())
    train_target_words = [w.lower() for w in train_df['word'].tolist()]

    print('Loading thesaurus')
    ThesClass = EnWordNet if config['lang'] == 'en' else RuThes if args.ruthes else RuWordNet
    thesaurus = ThesClass(args.thesaurus_dir)
    thesaurus.filter_thesaurus(train_df['word'])

    print('Loading vectors')
    embeddings_path = config['embeddings_path']

    vector_model = VectorsWithHash(gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False))

    modes = config['modes'].split(',')
    wiktionary = {}
    if 'wiktionary_dump_path' in config and 'wkt' in modes:
        print('Loading wiktionary')
        wiktionary = wkt.load_wiktionary(config['wiktionary_dump_path'], vector_model)

    print('Calculating train most sim')
    train_most_sim = calc_most_sim(vector_model, thesaurus, train_df, config['pos'], 100)

    print('Loading patterns and bert')
    syn_pattern_path = config['syn_pattern_path'] if config['syn_pattern_path'] is not None else None
    hyp_pattern_path = config['hyp_pattern_path'] if config['hyp_pattern_path'] is not None else None
    train_word2patterns_syn = load_word2patterns(syn_pattern_path) if 'p1' in modes else {}
    train_word2patterns_hyp = load_word2patterns(hyp_pattern_path) if 'p2' in modes else {}

    #bert_model = BertCls(args.bert_dir, no_cuda=False) if 'bert' in modes else None

    print('Calculating train candidates with features')
    train_synset_candidates = get_synset_candidates(train_most_sim, thesaurus, vector_model, config['pos'], config['train_synset_cand_count'],
                                                    train_word2patterns_syn, train_word2patterns_hyp, bert_model=None, wiktionary=wiktionary)

    train_ref, train_w2true = get_train_true_info(train_df)
    train_df_features = get_features_df(train_synset_candidates, w2true=train_w2true)
  
    print('Training')
    models, features = train_predict_cv(train_df_features, train_ref, folds=3)

    joblib.dump({'models': models, 'features': features}, args.model_save_path)

