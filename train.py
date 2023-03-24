import argparse
import pandas as pd
from taxoenrich.models import HypernymPredictModel
from taxoenrich.utils import create_train_dataset_broad, reinit_vector_model
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thesaurus_dir', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--pos', default=['N'])
    parser.add_argument('--lang', default='ru')
    parser.add_argument('--search_by_word', action='store_true')
    parser.add_argument('--processes', default=1, type=int)
    parser.add_argument('--train_fraction', default=0.05, type=float)
    parser.add_argument('--allowed_rels',  nargs='+', default=['hypernym'])
    parser.add_argument('--include_synset', action='store_true')
    parser.add_argument('--only_leafs', action='store_true')
    parser.add_argument('--include_second_order', action='store_true')
    parser.add_argument('--use_def', action='store_true' )
    parser.add_argument('--topk', default=40, type=int)
    args = parser.parse_args()

    config = {
        'pos': args.pos,
        'topk': args.topk,
        'lang': args.lang,
        'ruthes': False,
        'embeddings_path': args.embeddings_path,
        'search_by_word': args.search_by_word,
        'processes': args.processes,
        'thesaurus_dir': args.thesaurus_dir,
        'allowed_rels': args.allowed_rels,
        'include_synset': args.include_synset,
        'use_def': args.use_def
    }
    print(args)
    model = HypernymPredictModel(config)
    if args.train_path == None:
        #train_df = create_train_dataset(model.thesaurus, args.pos, fraction=args.train_fraction)
        train_df = create_train_dataset_broad(
            model.thesaurus, args.pos, allowed_rels=args.allowed_rels, fraction=args.train_fraction, 
            include_synset=args.include_synset, only_leafs=args.only_leafs, include_second_order=args.include_second_order)
    else:
        train_df = pd.read_csv(args.train_path)
        if 'word' not in train_df.columns:
            train_df.rename(columns={'target_word': 'word'}, inplace=True)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    train_path = os.path.join(args.output_path, 'train.csv')
    train_df.to_csv(train_path, index=False)

    if not args.search_by_word:
        reinit_vector_model(model, train_df['word'])

    model.train(train_df)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model_path = os.path.join(args.output_path)
    model.save(model_path)

