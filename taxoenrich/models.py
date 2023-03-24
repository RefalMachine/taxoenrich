from functools import lru_cache
import os
import codecs
import json
from tqdm import tqdm
import codecs
from multiprocessing import Pool
import pandas as pd
import numpy as np
import joblib
import gensim
from collections import OrderedDict
from taxoenrich.utils import StaticVectorModel, LogRegScaler, get_score, create_graph, reinit_vector_model
from taxoenrich.core import EnWordNet, RuWordNet, RuThes
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


class HypernymPredictModel():
    def __init__(self, config):
        self.config = config
        self._init_model_state(config)
        self._load_resources(config)

    def train(self, train_df):
        if not self.config['include_synset']:
            self.thesaurus.filter_synsets_with_words(train_df['word'])

        if self.config['processes'] > 1:
            with Pool(processes=self.config['processes']) as process_pool:
                run_query_args = tqdm([(self, row['word'], row['def'] if 'def' in row else '') for i, row in train_df.iterrows()], total=len(train_df['word']))
                features = process_pool.starmap(HypernymPredictModel._calculate_features, run_query_args, chunksize=128)
        else:
            features = []
            for i, row in tqdm(train_df.iterrows()):
                features.append(self._calculate_features(row['word'], row['def'] if 'def' in row else ''))
        train_features_df = pd.concat(features).reset_index(drop=True)

        ref, word2true = self._get_train_true_info(train_df)
        self._add_true(train_features_df, word2true)
        self.models, self.features = self._train_predict_cv(train_features_df, ref, 3)

    def predict(self, new_words, no_pred=False, chunksize=129):
        if self.config['processes'] > 1:
            with Pool(processes=self.config['processes']) as process_pool:
                run_query_args = tqdm([(self, word) for word in new_words], total=len(new_words))
                features = process_pool.starmap(HypernymPredictModel._calculate_features, run_query_args, chunksize=chunksize)
                print('ok')
        else:
            features = []
            for word in tqdm(new_words):
                features.append(self._calculate_features(word))
        if no_pred:
            return None

        features = [f for f in features if f is not None]
        if len(features) == 0:
            return None
        test_features_df = pd.concat(features).reset_index(drop=True)

        test_features_df[f'predict'] = [0] * test_features_df.shape[0]
        for model in self.models:
            test_features_df[f'predict'] += model.predict_proba(test_features_df[self.features])[:,1]
        test_features_df[f'predict'] /= len(self.models)

        test_features_df = test_features_df.sort_values(by=['word', 'predict'], ascending=False)
        test_features_df['word'] = test_features_df['word'].apply(lambda x: x.upper())
        test_features_df['cand_name'] = test_features_df['cand'].apply(lambda x: self.thesaurus.synsets[x].synset_name)

        return self._create_predict_df(test_features_df)

    @lru_cache(maxsize=50000)
    def predict_one(self, word, topk=10, definition=''):
        features = self._calculate_features(word, definition)
        if features is None:
            return None
        
        features = [features]
        test_features_df = pd.concat(features).reset_index(drop=True)
        test_features_df[f'predict'] = [0] * test_features_df.shape[0]
        for model in self.models:
            test_features_df[f'predict'] += model.predict_proba(test_features_df[self.features])[:,1]
        test_features_df[f'predict'] /= len(self.models)

        test_features_df = test_features_df.sort_values(by=['word', 'predict'], ascending=False)
        test_features_df['word'] = test_features_df['word'].apply(lambda x: x.upper())
        test_features_df['cand_name'] = test_features_df['cand'].apply(lambda x: self.thesaurus.synsets[x].synset_name)

        return self._create_predict_df(test_features_df, topk=topk)


    def save(self, model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump({'models': self.models, 'features': self.features}, model_path)

        config_path = os.path.join(model_dir, 'config')
        json.dump(self.config, codecs.open(config_path, 'w', 'utf-8'))
     
    @staticmethod
    def from_pretrained(model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        model_info = joblib.load(model_path)
        models = model_info['models']
        features = model_info['features']

        config_path = os.path.join(model_dir, 'config')
        config = json.load( codecs.open(config_path, 'r', 'utf-8'))

        hypernym_model = HypernymPredictModel(config)
        hypernym_model.models = models
        hypernym_model.features = features

        if not config.get('search_by_word', False):
            reinit_vector_model(hypernym_model)

        return hypernym_model

    def _init_model_state(self, config):
        self.word_types = config['pos']
        self.topk = config['topk']
        self.lang = config['lang']
        self.search_by_word = config['search_by_word']


        self.wkt_on = config['wkt']
        global wkt
        if self.lang == 'en':
            import wiktionary_processing.utils_en as wkt
        else:
            import wiktionary_processing.utils as wkt

    def _load_resources(self, config):
        self._load_thesaurus(config)
        self._load_vectors(config)
        self.G = create_graph(self.thesaurus, self.word_types, self.config['allowed_rels'])
        self._load_wkt(config)

    def _load_thesaurus(self, config):
        print('Loading Thesaurus')
        ThesClass = EnWordNet if config['lang'] == 'en' else RuThes if config['ruthes'] else RuWordNet
        self.thesaurus = ThesClass(config['thesaurus_dir'])
        
    def _load_vectors(self, config):
        print('Loading Vectors')
        embeddings_path = config['embeddings_path']
        try:
            self.vector_model = StaticVectorModel(gensim.models.KeyedVectors.load(embeddings_path))
        except:
            self.vector_model = StaticVectorModel(gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False))

    def _load_wkt(self, config):
        self.wiktionary = {}
        if self.wkt_on:
            print('Loading Wiktionary')
            self.wiktionary = wkt.load_wiktionary(config['wiktionary_dump_path'], self.vector_model)

    def _calculate_features(self, word, definition=''):
        candidate2features = self._calculate_candidates(word, definition=definition)
        if len(candidate2features) == 0:
            return None
        candidate_col = []
        features = []
        for synset_id in candidate2features:
            if synset_id not in self.thesaurus.synsets:
                continue
            synset = self.thesaurus.synsets[synset_id]

            init_features = self._calculate_init_features(synset_id, candidate2features)
            wkt_features = self._calculate_wiktionary_features(word, synset)
            synset_features = self._calculate_synset_similarity(word, synset)

            candidate_col.append(synset_id)
            features.append(init_features + wkt_features + synset_features)
            #features.append(init_features + synset_features)
            if self.config['use_def']:
                def_features = self._calculate_def_features(word, synset, definition)
                features[-1] += def_features

        features = np.array(features)

        columns = OrderedDict()
        columns['word'] = [word] * len(candidate_col)
        columns['cand'] = candidate_col

        for i in range(features.shape[1]):
            columns[f'f{i}'] = features[:,i]

        df = pd.DataFrame(columns)
        return df

    def _calculate_def_features(self, word, synset, definition):
        features = [0, 0]
        definition = definition.lower().replace(' ', '_')
        for synset_word in synset.synset_words:
            if synset_word in definition:
                features[0] = 1
            if synset_word in '_'.join(definition.split('_')[:5]):
                features[1] = 1
        return features

    def _calculate_candidates(self, word, definition=''):
        
        if word not in self.vector_model:
            return {}
        most_similar_words = self.vector_model.most_similar(word, topn=10000) # must be larger then topk
        
        if self.search_by_word:
            most_similar_words = self._filter_most_sim_by_type(word, most_similar_words)
            most_similar_words = most_similar_words[:self.topk]

            candidates = []
            for cand_word, similarity in most_similar_words:
                if cand_word in self.thesaurus.sense2synid:
                    for synid in self.thesaurus.sense2synid[cand_word]:
                        if self.thesaurus.synsets[synid].synset_type not in self.word_types:
                            continue
                        
                        candidates.append([synid, 0, similarity])
                        for rsid in self.G[synid]:
                            candidates.append([rsid, 1, similarity])
                            for rrsid in self.G[rsid]:
                                candidates.append([rrsid, 2, similarity])
                        '''
                        for hid in self.thesaurus.synsets[synid].rels.get('hypernym', []):
                            candidates.append([hid, 1, similarity])
                            h = self.thesaurus.synsets[hid]
                            for hhid in h.rels.get('hypernym', []):
                                candidates.append([hhid, 2, similarity])
                        '''
            definition = definition.lower().replace(' ', '_')
            for cand_word in definition.split('_'):
                if cand_word in self.vector_model and cand_word in self.thesaurus.sense2synid:
                    for synid in self.thesaurus.sense2synid[cand_word]:
                        if self.thesaurus.synsets[synid].synset_type not in self.word_types:
                            continue
                        
                        candidates.append([synid, 0, self.vector_model.similarity(word, cand_word)])
        else:
            most_similar_words = self._filter_most_sim(word, most_similar_words)
            most_similar_words = most_similar_words[:self.topk]
            candidates = []
            for synid, _ in most_similar_words:
                if self.thesaurus.synsets[synid].synset_type not in self.word_types:
                    continue
                
                candidates.append([synid, 0])
                for hid in self.thesaurus.synsets[synid].rels.get('hypernym', []):
                    candidates.append([hid, 1])
                    h = self.thesaurus.synsets[hid]
                    for hhid in h.rels.get('hypernym', []):
                        candidates.append([hhid, 2])
                            
        
        candidate2features = {}
        for cand_info in candidates:
            synset_id = cand_info[0]
            features = cand_info[1:]
            if synset_id not in candidate2features:
                candidate2features[synset_id] = [0]
                for f in features:
                    candidate2features[synset_id].append([])

            candidate2features[synset_id][0] += 1
            for i, f in enumerate(features):
                candidate2features[synset_id][i + 1].append(f)

        return candidate2features


    def _calculate_init_features(self, synset_id, candidate2features):
        features = []
        init_features = candidate2features[synset_id]

        features.append(init_features[0])
        features.append(np.log2(2 + init_features[0]))

        features.append(np.min(init_features[1]))
        features.append(np.mean(init_features[1]))
        features.append(np.max(init_features[1]))

        return features


    def _calculate_wiktionary_features(self, target_word, synset):
        # 1 feature for direct syn, 1 feature for hypo syn, 1 feature for direct hyper, 1 feature for hypo hyper
        if len(self.wiktionary) == 0:
            return []
        
        features = [0] * 6
        direct_syn_feature_idx = 0
        hypo_syn_feature_idx = 1
        direct_hyper_feature_idx = 2
        hypo_hyper_feature_idx = 3
        direct_meaning_feature_idx = 4
        hypo_meaning_feature_idx = 5

        def get_all_wkt(word, wiktionary, tag):
            all_words = set([word])
            for wikt_doc_info in wiktionary.get(word, []):
                all_words.update(wikt_doc_info[tag])
            return all_words

        tw_synonyms = get_all_wkt(target_word, self.wiktionary, 'synonym')
        tw_hypernyms = get_all_wkt(target_word, self.wiktionary, 'hypernym')
        tw_meaning = []
        for wikt_doc_info in self.wiktionary.get(target_word, []):
            tw_meaning.append('_'.join(wikt_doc_info['meaning']).replace(' ', '_'))
        tw_meaning = '_'.join(tw_meaning)

        synset_synonyms = set()
        for word in synset.synset_words:
            synset_synonyms.update(get_all_wkt(word, self.wiktionary, 'synonym'))

        hypo_synonyms = set()
        hypo_words = set()
        for hypoid in synset.rels.get('hyponym', []):
            hypo = self.thesaurus.synsets[hypoid]
            hypo_words.update(hypo.synset_words)
            for word in hypo.synset_words:
                hypo_synonyms.update(get_all_wkt(word, self.wiktionary, 'synonym'))

        features[direct_syn_feature_idx] = int(len(tw_synonyms.intersection(synset_synonyms)) > 0)
        features[hypo_syn_feature_idx] = int(len(tw_synonyms.intersection(hypo_synonyms)) > 0)
        features[direct_hyper_feature_idx] = int(len(tw_hypernyms.intersection(set(synset.synset_words))) > 0)
        features[hypo_hyper_feature_idx] = int(len(tw_hypernyms.intersection(hypo_words)) > 0)

        for w in synset_synonyms:
            if w in tw_meaning:
                features[direct_meaning_feature_idx] = 1

        for w in hypo_synonyms:
            if w in tw_meaning:
                features[hypo_meaning_feature_idx] = 1

        return features

    def _calculate_synset_similarity(self, w, synset):
        f_lists = [[], [], [], []]
        for synset_word in synset.synset_words:
            if synset_word in self.vector_model:
                f_lists[0].append(self.vector_model.similarity(w, synset_word))
        for hypoid in synset.rels.get('hyponym', []):
            hyponym = self.thesaurus.synsets[hypoid]
            hyponym_sim = []
            for synset_word in hyponym.synset_words:
                if synset_word in self.vector_model:
                    hyponym_sim.append(self.vector_model.similarity(w, synset_word))
            if len(hyponym_sim) == 0:
                hyponym_sim.append(0)
            f_lists[1].append(np.max(hyponym_sim))
            f_lists[2].append(np.mean(hyponym_sim))
            f_lists[3].append(np.min(hyponym_sim))
        results = []
        for f_list in f_lists:
            if len(f_list) == 0:
                f_list.append(0)
            results.append(np.max(f_list))
            results.append(np.mean(f_list))
            results.append(np.min(f_list))
        return results

    @staticmethod
    def _get_train_true_info(train_df):
        reference = {}
        w2true = {}
        for _, row in train_df.iterrows():
            word = row['word']
            if type(row['target_gold']) == str:
                target_gold = json.loads(row['target_gold'])
            else:
                target_gold = row['target_gold']
            reference[word] = target_gold
            w2true[word] = set()
            for t in target_gold:
                w2true[word].update([str(c) for c in t])

        return reference, w2true

    @staticmethod
    def _add_true(df, word2true):
        true_col = []
        for i, row in df.iterrows():
            word = row['word']
            cand = row['cand']
            label = int(cand in word2true[word])
            true_col.append(label)

        df['label'] = true_col

    def _train_predict_cv(self, df_features, ref, folds=3):
        non_features_col_num = 3
        features_len = len(df_features.columns) - non_features_col_num
        features = [f'f{i}' for i in range(features_len)]
        print(df_features.shape[0], df_features['label'].sum())
        kf = KFold(n_splits=folds)

        results = []
        models = []
        for train_index, test_index in kf.split(df_features['word'].unique()):
            train_words = df_features['word'].unique()[train_index]
            test_words = df_features['word'].unique()[test_index]

            train_df = df_features[df_features['word'].apply(lambda x: x in train_words)]
            test_df = df_features[df_features['word'].apply(lambda x: x in test_words)]

            clf = LogRegScaler()

            X_train = train_df[features]
            X_test = test_df[features]
            y_train = train_df['label']
            y_test = test_df['label']

            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)

            test_df['predict'] = y_pred[:,1]
            roc_auc = roc_auc_score(y_test, y_pred[:,1])
            test_df = test_df.sort_values(by=['word', 'predict'], ascending=False)

            cur_ref = {w: ref[w] for w in ref if w in set(test_words)}
            mean_ap, mean_rr = get_score(cur_ref, self._from_df_to_pred(test_df), k=10)
            eval_res = [mean_ap, mean_rr, roc_auc]

            models.append(clf)

            print(eval_res)
            results.append(eval_res)

        print(f'Averaged results = {np.mean(results, axis=0)}')
        return models, features

    def _filter_most_sim_by_type(self, word, most_similar_words):
        filtered_word_list = []
        banned_words = set()
        for synid in self.thesaurus.sense2synid.get(word, []):
            synset = self.thesaurus.synsets[synid]
            banned_words.update(synset.synset_words)

        for w, score in most_similar_words:
            w = w.replace('ё', 'е')
            if w not in self.thesaurus.sense2synid:
                continue

            if w in banned_words:
                continue

            found_sense = False
            for synid in self.thesaurus.sense2synid[w]:
                if self.thesaurus.synsets[synid].synset_type in self.word_types:
                    found_sense = True

            if found_sense is True:
                filtered_word_list.append([w, score])

        return filtered_word_list

    def _filter_most_sim(self, word, most_similar_words):
        filtered_word_list = []
        for w, score in most_similar_words:
            if not w.startswith('__s'):
                continue
            
            synid = w[len('__s'):]
            if synid in self.thesaurus.synsets and self.thesaurus.synsets[synid].synset_type in self.word_types:
                filtered_word_list.append([synid, score])

        return filtered_word_list

    @staticmethod
    def _from_df_to_pred(df):
        pred = {}
        for i, row in df.iterrows():
            word = row['word']
            if word not in pred:
                pred[word] = []
            cand = str(row['cand'])
            pred[word].append(cand)
        return pred

    @staticmethod
    def _create_predict_df(df, topk=10):
        target_words = []
        predict = []
        probas = []
        cand_names = []
        counter = 0
        last_w = ''
        for i, row in df.iterrows():
            if row['word'] != last_w:
                counter = 0
                last_w = row['word']
            if counter < topk:
                target_words.append(row['word'].upper())
                predict.append(row['cand'])
                probas.append(row['predict'])
                cand_names.append(row['cand_name'])
            counter += 1
        return pd.DataFrame({'word': target_words, 'cand': predict, 'cand_name': cand_names, 'prob': probas})
