
import argparse
import pandas as pd
import codecs
import json
from tqdm import tqdm
import gensim
import sys
import joblib
import os

from taxoenrich.models import EnWordNet, RuWordNet, RuThes
from taxoenrich.utils import VectorsWithHash, load_word2patterns, predict_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--task_json_path', type=str)
    parser.add_argument('--thesaurus_dir', type=str)
    parser.add_argument('--ruthes', action='store_true')
    parser.add_argument('--save_path')
    
    args = parser.parse_args()

    config = json.load(codecs.open(args.task_json_path, 'r', 'utf-8'))
    if config['lang'] == 'en':
        import wiktionary_processing.utils_en as wkt
    else:
        import wiktionary_processing.utils as wkt

    model_info = joblib.load(args.model_path)
    models = model_info['models']
    features = model_info['features']

    predict_df = pd.read_csv(args.test_path, header=None, sep='\t').rename(columns={0: 'word'})
    predict_df['word'] = predict_df['word'].apply(lambda x: x.lower())

    print('Loading thesaurus')
    ThesClass = EnWordNet if config['lang'] == 'en' else RuThes if args.ruthes else RuWordNet
    thesaurus = ThesClass(args.thesaurus_dir)
    #thesaurus.filter_thesaurus(predict_df['word'])

    print(len([w for w in predict_df['word'].tolist() if w in thesaurus.senses]))

    print('Loading vectors')
    embeddings_path = config['embeddings_path']
    vector_model = VectorsWithHash(gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False))

    modes = config['modes'].split(',')
    wiktionary = {}
    if 'wiktionary_dump_path' in config and 'wkt' in modes:
        print('Loading wiktionary')
        wiktionary = wkt.load_wiktionary(config['wiktionary_dump_path'], vector_model)


    print('Loading patterns and bert')
    syn_pattern_path = config['syn_pattern_path'] if config['syn_pattern_path'] is not None else None
    hyp_pattern_path = config['hyp_pattern_path'] if config['hyp_pattern_path'] is not None else None
    word2patterns_syn = load_word2patterns(syn_pattern_path) if 'p1' in modes else {}
    word2patterns_hyp = load_word2patterns(hyp_pattern_path) if 'p2' in modes else {}

    #bert_model = BertCls(args.bert_dir, no_cuda=False) if 'bert' in modes else None
    model_info = joblib.load(args.model_path)
    models = model_info['models']
    features = model_info['features']

    predict_test(predict_df, thesaurus, vector_model, models, features, config['pos'], args.save_path,
                word2patterns_syn=word2patterns_syn, word2patterns_hyp=word2patterns_hyp,
                predict_synset_cand_count=config['predict_synset_cand_count'], lang=config['lang'],
                wiktionary=wiktionary)


    
