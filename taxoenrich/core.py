import os
import codecs
import json
from tqdm import tqdm
import pymorphy2
import xml.etree.ElementTree as ET
import codecs

class SynSet:
    def __init__(self, synset_id, synset_name, synset_type, words, synset_names_all=[]):
        self.synset_type = synset_type
        self.synset_name = synset_name
        self.synset_names_all = synset_names_all
        self.synset_id = synset_id
        self.synset_words = words
        self.rels = {}
        self.total_hyponyms = None

    def add_rel(self, synset_id, rel_type):
        if rel_type not in self.rels:
            self.rels[rel_type] = []

        self.rels[rel_type].append(synset_id)

class WordNet:
    def __init__(self):
        self.synsets = {}
        self.senses = set()
        self.sense2synid = {}

    def filter_synsets_with_words(self, words):
        for w in words:
            if w not in self.sense2synid:
                continue
            synsets_ids = self.sense2synid[w]
            try:
                for synset_id in synsets_ids:
                    synset = self.synsets[synset_id]
                    parents = synset.rels.get('hypernym', [])
                    childs = synset.rels.get('hyponym', [])
                    parents = [self.synsets[sid] for sid in parents]
                    childs = [self.synsets[sid] for sid in childs]
                    for parent_synset in parents:
                        del parent_synset.rels['hyponym'][parent_synset.rels['hyponym'].index(synset_id)]
                    for child_synset in childs:
                        del child_synset.rels['hypernym'][child_synset.rels['hypernym'].index(synset_id)]

                    for parent_synset in parents:
                        for child_synset in childs:
                            if child_synset.synset_id in parent_synset.rels['hyponym']:
                                continue
                            parent_synset.rels['hyponym'].append(child_synset.synset_id)
                            child_synset.rels['hypernym'].append(parent_synset.synset_id)

                    del self.synsets[synset_id]

                    for sense in synset.synset_words:
                        if sense in self.senses:
                            self.senses.remove(sense)
                        if sense in self.sense2synid:
                            del self.sense2synid[sense]
            except Exception as e:
                print(w)
                print(synset_id)
                print(e)

class EnWordNet(WordNet):
    def __init__(self, wordnet_root):
        super().__init__()
        self.index = []
        self.rel_map = {
            '@': 'hypernym',
            '@i': 'hypernym',
            '~': 'hyponym',
            '~i': 'hyponym'
            }
        self._load_wordnet(wordnet_root)

    def _load_wordnet(self, wordnet_root):
        self._load_synsets(wordnet_root)

    def _load_index(self, wordnet_root):
        synsets_paths = {
            'N': os.path.join(wordnet_root, os.path.join('dict', 'index.noun')),
            'A': os.path.join(wordnet_root, os.path.join('dict', 'index.adj')),
            'V': os.path.join(wordnet_root, os.path.join('dict', 'index.verb')),
            'R': os.path.join(wordnet_root, os.path.join('dict', 'index.adv')),
        }

        index = {t: {} for t in synsets_paths}
        for synset_type, synset_path in synsets_paths.items():
            with codecs.open(synset_path, 'r', 'utf-8') as file:
                for line in file:
                    if line.startswith('\t') or line.startswith(' '):
                        continue
                    line_content = line.strip().split()
                    word = line_content[0]
                    synset_count = int(line_content[2])
                    ptr_count = int(line_content[3])
                    index[synset_type][word] = line_content[4 + ptr_count + 2:]
                    assert len(line_content[4 + ptr_count + 2:]) == synset_count

        return index


    def _load_synsets(self, wordnet_root, black_list_synsets=None, black_list_senses=None):
        synsets_paths = {
            'N': os.path.join(wordnet_root, os.path.join('dict', 'data.noun')),
            'A': os.path.join(wordnet_root, os.path.join('dict', 'data.adj')),
            'V': os.path.join(wordnet_root, os.path.join('dict', 'data.verb')),
            'R': os.path.join(wordnet_root, os.path.join('dict', 'data.adv')),
        }
        self.index = self._load_index(wordnet_root)
        for synset_type, synset_path in synsets_paths.items():
            with codecs.open(synset_path, 'r', 'utf-8') as file:
                for line in file:
                    if line.startswith('\t') or line.startswith(' '):
                        continue
                    try:
                        synset_info = self._read_line(line)
                        synset_id = synset_info['id']
                        synset_name = synset_info['name']
                        synset_name_idx = self.index[synset_type][synset_name].index(synset_info['id'][:-1]) + 1
                        synset_name = f'{synset_name}.{synset_type.lower()}.{self._to2digit(synset_name_idx)}'
                        synset_words = synset_info['words']
                        synset_names_all = []
                        for word in synset_info['words']:
                            word_name_idx = self.index[synset_type][word].index(synset_info['id'][:-1]) + 1
                            synset_names_all.append(f'{word}.{synset_type.lower()}.{self._to2digit(word_name_idx)}')

                    except Exception as e:
                        print(line)
                        raise e

                    if black_list_synsets is not None and synset_id in black_list_synsets:
                        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set())
                    else:
                        if black_list_senses is not None:
                            synset_words = [w for w in synset_words if w not in black_list_senses]

                        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set(synset_words), synset_names_all)
                        self.senses.update(synset_words)
                        for sense in synset_words:
                            if sense not in self.sense2synid:
                                self.sense2synid[sense] = []
                            self.sense2synid[sense].append(synset_id)

                    for rel_type, rel_synset_id in synset_info['rels']:
                        self.synsets[synset_id].add_rel(rel_synset_id, rel_type)

    def _read_line(self, line):
        synset_info = {}

        ID_IDX = 0
        POS_IDX = 2
        W_LEN_IDX = 3
        WORDS_SHIFT = 4

        line_content = line.strip().split()
        synset_id = line_content[ID_IDX]
        pos = line_content[POS_IDX]
        if pos == 's':
            pos = 'a'
        synset_id += pos

        synset_info['id'] = synset_id

        synset_words_len = int(line_content[W_LEN_IDX], 16)
        synset_words = []
        
        for i in range(synset_words_len):
            synset_words.append(line_content[WORDS_SHIFT + i * 2])


        synset_info['words'] = []
        for w in synset_words:
            if '(' in w:
                w = w[:w.find('(')]
            w = w.lower()
            if w not in synset_info['words']:
                synset_info['words'].append(w)
            
        synset_name = synset_info['words'][0]
        synset_info['name'] = synset_name

        RELS_SHIFT = WORDS_SHIFT + synset_words_len * 2 + 1
        rels_count = int(line_content[RELS_SHIFT - 1])
        cur_rel_shift = RELS_SHIFT
        rels = []
        while len(rels) != rels_count and cur_rel_shift < len(line_content) - 2 and line_content[cur_rel_shift] != '|':
            rel_type = self.rel_map.get(line_content[cur_rel_shift], line_content[cur_rel_shift])

            rel_synset_id = line_content[cur_rel_shift + 1]
            pos = line_content[cur_rel_shift + 2]
            if pos == 's':
                pos = 'a'
            rel_synset_id += pos
            rels.append((rel_type, rel_synset_id))

            cur_rel_shift += 4

        assert len(rels) == rels_count
        rels = [rel for rel in rels if rel[1][-1] == synset_id[-1]]
        synset_info['rels'] = rels

        return synset_info

    def _to2digit(self, num):
        return '0' * (2 - len(str(num))) + str(num)


class RuWordNet(WordNet):
    def __init__(self, wordnet_root):
        super().__init__()
        self._load_wordnet(wordnet_root)

    def _load_wordnet(self, wordnet_root):
        self._load_synsets(wordnet_root)
        self._load_rels(wordnet_root)

    def _load_synsets(self, wordnet_root, black_list_synsets=None, black_list_senses=None):
        synsets_paths = {
            'N': os.path.join(wordnet_root, 'synsets.N.xml'),
            'A': os.path.join(wordnet_root, 'synsets.A.xml'),
            'V': os.path.join(wordnet_root, 'synsets.V.xml')
        }

        morph_analizer = pymorphy2.MorphAnalyzer()
        for synset_type, synset_path in synsets_paths.items():
            root = ET.parse(synset_path).getroot()
            for synset in list(root):
                synset_name = synset.attrib['ruthes_name'].lower()
                synset_id = synset.attrib['id']
                if black_list_synsets is not None and synset_id in black_list_synsets:
                    self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, set())
                    continue
                synset_words = set()
                for sense in list(synset):
                    word = sense.text.lower().replace('ё', 'е')
                    split_word = word.split()
                    split_word = [morph_analizer.parse(w)[0].normal_form.replace('ё', 'е') for w in split_word]
                    sense = '_'.join(split_word)
                    if black_list_senses is not None and sense in black_list_senses:
                        continue
                    synset_words.add(sense)

                self.senses.update(synset_words)
                self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, synset_words)
                for sense in synset_words:
                    if sense not in self.sense2synid:
                        self.sense2synid[sense] = []
                    self.sense2synid[sense].append(synset_id)

    def _load_rels(self, wordnet_root):
        synsets_rels_paths = {
            'N': os.path.join(wordnet_root, 'synset_relations.N.xml'),
            'A': os.path.join(wordnet_root, 'synset_relations.A.xml'),
            'V': os.path.join(wordnet_root, 'synset_relations.V.xml')
        }

        for synset_rel_type, synset_rels_path in synsets_rels_paths.items():
            root = ET.parse(synset_rels_path).getroot()
            for synset_rel in list(root):
                synset_id = synset_rel.attrib['parent_id']
                rel_synset_id = synset_rel.attrib['child_id']
                rel_type = synset_rel.attrib['name']
                #if rel_type not in ['hyponym', 'hypernym', 'instance hypernym', 'instance hyponym']:
                #    continue

                if rel_type == 'instance hypernym':
                    rel_type = 'hypernym'
                    
                if rel_type == 'instance hyponym':
                    rel_type = 'hyponym'

                if synset_id not in self.synsets or rel_synset_id not in self.synsets:
                    continue

                self.synsets[synset_id].add_rel(rel_synset_id, rel_type)

class RuThes(WordNet):
    def __init__(self, concepts_path):
        super().__init__()
        self.rels_map = {
            'ВЫШЕ': 'hypernym',
            'НИЖЕ': 'hyponym',
            'ЦЕЛОЕ': 'whole',
            'ЧАСТЬ': 'part',
            'АССОЦ': 'assoc',
        }

        self._load_synsets(concepts_path)
        self._load_rels(concepts_path)
        
        #self._calc_hypo_rels()

    def _load_synsets(self, concepts_path):
        with codecs.open(concepts_path, 'r', 'utf-8') as file:
            for line in tqdm(file):
                concept_info = json.loads(line)
                self._add_concept(concept_info)

    def _load_rels(self, concepts_path):
        with codecs.open(concepts_path, 'r', 'utf-8') as file:
            for line in tqdm(file):
                concept_info = json.loads(line)
                self._add_rel(concept_info)


    def _add_concept(self, concept_info):
        synset_id = str(concept_info['conceptid'])
        synset_name = concept_info['conceptstr']
        synset_words = concept_info['synonyms']

        if synset_id == '0' and synset_name == 'empty':
            return
        synset_type = 'N'

        synset_words = [sense_info['lementrystr'].lower().replace(' ', '_') for sense_info in synset_words]
        self.senses.update(synset_words)
        self.synsets[synset_id] = SynSet(synset_id, synset_name, synset_type, synset_words)
        for sense in synset_words:
            if sense not in self.sense2synid:
                self.sense2synid[sense] = []
            self.sense2synid[sense].append(synset_id)

    def _add_rel(self, concept_info):
        synset_id = str(concept_info['conceptid'])
        for rel in concept_info['relats']:
            if rel['relationstr'] in self.rels_map:
                rel_type = self.rels_map[rel['relationstr']]
                if rel_type == 'assoc':
                    rel_type += rel['aspect']

                self.synsets[synset_id].add_rel(str(rel['conceptid']), rel_type)
