import gensim
import numpy as np
import argparse
import codecs

def get_max_available_pref(word, w2v_model, min_pref_len=4):
    if word in w2v_model:
        return w2v_model[word]
    for i in range(0, len(word) - min_pref_len):
        pref = word[:-(i+1)]
        if pref in w2v_model:
            return w2v_model[pref]
        
    return None

def get_sense_vec(sense, w2v_model):
    vectors = [get_max_available_pref(w, w2v_model) for w in sense.split('_')]
    vectors = [w for w in vectors if w is not None]
    if len(vectors) == 0:
        return None

    return np.mean(vectors, axis=0)

def load_model(model_path):
    try:
        model = gensim.models.KeyedVectors.load(model_path)
    except:
        try:
            model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
        except:
            model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--extended_model_path')
    parser.add_argument('--ext_vocab_path')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    model = load_model(args.model_path)
    new_words = []
    new_vectors = []
    with codecs.open(args.ext_vocab_path, 'r', 'utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0 and line not in model.wv:
                vec = get_sense_vec(line, model)
                if vec is not None:
                    new_words.append(line)
                    new_vectors.append(vec)

    model.add_vectors(keys=new_words, weights=new_vectors, replace=args.force)
    model.fill_norms(force=True)
    model.save(args.extended_model_path)
