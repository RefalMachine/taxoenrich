import argparse
from model import CAEME, AAEME
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler, Dataset
import gensim
from tqdm import tqdm
import numpy as np
import codecs
import os
import math
from transformers import get_linear_schedule_with_warmup
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds, eigs

import sys
sys.path.append(r'D:\WorkFolderNew\projects')
sys.path.append(r'D:\WorkFolderNew\projects\taxoenrich')
from taxoenrich.models import RuWordNet, EnWordNet, RuThes

lang = ''
def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors

def save_vectors(wv, wv_path):
    with codecs.open(wv_path, 'w', 'utf-8') as file_descr:
        wv_size = wv[list(wv.keys())[0]].shape[0]

        nwords = len({w: wv[w] for w in wv if wv[w].shape[0] == wv_size})
        file_descr.write(f'{nwords} {wv_size}')
        for w in tqdm(wv):
            if wv[w].shape[0] != wv_size:
                continue
            vector = ' '.join([str(val) for val in wv[w]])
            file_descr.write(f'\n{w[3:]} {vector}')

def find_closes_w_vec(w, word_vectors_list, wv_index):
    most_sim = {}
    if f'_name{w}' in word_vectors_list[wv_index]:
        return word_vectors_list[wv_index][f'_name{w}']

    return np.zeros(word_vectors_list[wv_index].vector_size, dtype=np.float32)

def get_cooc_vectors(word_vectors_list):
    total_vocab = {}
    min_w_count = 1
    for wv in word_vectors_list:
        for w in wv.key_to_index:
            if w not in total_vocab:
                total_vocab[w] = 0
            total_vocab[w] += 1
    union_vocab = sorted(list([w for w in total_vocab if total_vocab[w] == len(word_vectors_list)]))
    oov_vocab = sorted(list([w for w in total_vocab if total_vocab[w] < len(word_vectors_list) and total_vocab[w] >= min_w_count]))

    total_vocab = union_vocab + oov_vocab
    matrixes = []
    for i in range(len(word_vectors_list)):
        matrixes.append([])
    matrix = []
    for w in tqdm(total_vocab):
        v_list = []
        for i, wv in enumerate(word_vectors_list):
            if w not in wv:
                v_list.append(find_closes_w_vec(w, word_vectors_list, i))
                matrixes[i].append(find_closes_w_vec(w, word_vectors_list, i))
            else:
                v_list.append(wv[w])
                matrixes[i].append(wv[w])
        matrix.append(np.array(v_list))

    return total_vocab, matrix, matrixes

def get_concat_vectors(vectors):
    vocab, _, matrixes = get_cooc_vectors(vectors)
    conc_wv = np.concatenate(matrixes, axis=1)
    conc_wv_dict = {f'{lang}_{vocab[i]}': conc_wv[i] for i in range(len(vocab))}
    print('Concat')

    return conc_wv_dict

def get_svd_vectors(vectors, vectors_size):
    vocab, _, matrixes = get_cooc_vectors(vectors)
    conc_wv = np.concatenate(matrixes, axis=1)
    print('Concat')

    conc_wv_svd, _, _ = svds(conc_wv, k=vectors_size)
    conc_wv_svd_dict = {f'{lang}_{vocab[i]}': conc_wv_svd[i] for i in range(len(vocab))}
    print('SVD')

    return conc_wv_svd_dict

class DatasetWithConctrains(Dataset):
    def __init__(self, matrix, constrains, targets):
        self.matrix = [torch.tensor(np.vstack(matrix[i])) for i in range(matrix.shape[0])]
        self.constrains = constrains
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        inputs = [self.matrix[i][idx] for i in range(len(self.matrix))]
        constrains = self.constrains[idx]
        constrains = [self.matrix[i][constrains] for i in range(len(self.matrix))]
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return inputs, constrains, targets

def create_dataset(vocab, full_matrix, indices=None, thes_constrains=False, thesaurus=None, constrains_count=3):
    if not thes_constrains:
        if indices is not None:
            matrix = [torch.tensor(np.vstack(full_matrix[i][indices])) for i in range(len(full_matrix))]
        else:
            matrix = [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))]
        print([m.shape for m in matrix])
        return TensorDataset(*matrix)

    sense2synid = {w: [] for w in thesaurus.senses}
    for synset_id in thesaurus.synsets:
        for sense in thesaurus.synsets[synset_id].synset_words:
            sense2synid[sense].append(synset_id)

    if indices is None:
        indices = [i for i in range(len(vocab))]

    word2id = {}
    id2word = {}
    for i in indices:
        w = vocab[i]
        word2id[w] = i
        id2word[i] = w

    all_pos_constrains = []
    for i in tqdm(indices):
        w = id2word[i]
        pos_constrains = []
        if w in sense2synid:
            for synid in sense2synid[w]:
                pos_constrains += [word2id[synset_w] for synset_w in thesaurus.synsets[synid].synset_words if synset_w != w and synset_w in word2id]
            for hypo in thesaurus.synsets[synid].rels.get('hyponym', []):
                pos_constrains += [word2id[synset_w] for synset_w in thesaurus.synsets[hypo].synset_words if synset_w != w and synset_w in word2id]
            for hyper in thesaurus.synsets[synid].rels.get('hypernym', []):
                pos_constrains += [word2id[synset_w] for synset_w in thesaurus.synsets[hyper].synset_words if synset_w != w and synset_w in word2id]

        pos_constrains = list(set(pos_constrains))
        all_pos_constrains.append(pos_constrains)

    if indices is not None:
        matrix = [torch.tensor(np.vstack(full_matrix[i][indices])) for i in range(len(full_matrix))]
    else:
        matrix = [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))]
    return DatasetWithConctrainsOnline(vocab, [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))], matrix, all_pos_constrains, constrains_count=constrains_count)

class DatasetWithConctrainsOnline(Dataset):
    def __init__(self, full_vocab, full_matrix, matrix, pos_constrains, constrains_count=3):
        self.full_vocab = full_vocab
        self.full_matrix = full_matrix
        self.matrix = matrix
        self.pos_constrains = pos_constrains
        self.constrains_count = constrains_count

    def __len__(self):
        return len(self.matrix[0])

    def __getitem__(self, idx):
        inputs = [self.matrix[i][idx] for i in range(len(self.matrix))]
        def normalize_vec(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        if  len(self.pos_constrains[idx]) > 0:
            pos_constrains = np.random.choice(self.pos_constrains[idx], self.constrains_count)
            pos_constrains = [self.full_matrix[i][pos_constrains] for i in range(len(self.full_matrix))]
        else:
            pos_constrains = [[] for i in range(len(inputs))]
            for i, m in enumerate(inputs):

                for j in range(self.constrains_count):
                    noised_vec = m + torch.tensor(normalize_vec(np.random.rand(m.shape[0]) - 0.5) / 10, dtype=torch.float32)
                    pos_constrains[i].append(noised_vec)

                pos_constrains[i] = torch.stack(pos_constrains[i])

        neg_constrains = [np.random.randint(0, len(self.full_vocab)) for i in range(self.constrains_count)]
        neg_constrains = [self.full_matrix[i][neg_constrains] for i in range(len(self.full_matrix))]


        return inputs, pos_constrains, neg_constrains

def eval(args, model, dev_dataloader, epoch):
    model.eval()
    epoch_losses = []
    for batch_idx, batch in tqdm(enumerate(dev_dataloader)):
        batch = [b for b in batch]
        if args.thes_constrains:
            inputs, pos_constrains, neg_constrains = batch
        else:
            inputs = batch
        inputs = [d.to(torch.device('cuda')) for d in inputs]

        if args.thes_constrains:
            pos_constrains = [d.to(torch.device('cuda')) for d in pos_constrains]
            neg_constrains = [d.to(torch.device('cuda')) for d in neg_constrains]
        else:
            pos_constrains = None
            neg_constrains = None
        loss = model(inputs, pos_constrains, neg_constrains)

        if args.thes_constrains and len(loss) == 3:
            loss, loss_direct, loss_constrains = loss

            epoch_losses.append((float(loss.detach().cpu().numpy()), float(loss_direct.detach().cpu().numpy()), float(loss_constrains.detach().cpu().numpy())))
        else:
            epoch_losses.append(loss.detach().cpu().numpy())

    loss_info = np.mean(epoch_losses, axis=0)
    print(f'Dev loss after {epoch} epoch = {loss_info}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', nargs='+')
    parser.add_argument('--model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--logging_every_epochs', default=10, type=int)
    parser.add_argument('--result_path')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--wv_weights', nargs='+', default=None, type=float)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--thes_constrains', action='store_true')
    parser.add_argument('--thes_path', type=str)
    parser.add_argument('--constrains_count', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dev_size', type=float, default=None)
    parser.add_argument('--lang', type=str, default='ru')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--distance_type', type=str, default='mse')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--ruthes', action='store_true')
    args = parser.parse_args()

    lang = args.lang
    vectors = []
    for wv_path in args.vectors:
        try:
            vectors.append(gensim.models.KeyedVectors.load(wv_path))
        except:
            vectors.append(gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=False))

        vectors[-1].vectors = normalize(vectors[-1].vectors, axis=1)

    wv_shapes = [wv.vector_size for wv in vectors]
    if args.model == 'CAEME':
        model = CAEME(wv_shapes, args.wv_weights, alpha=args.alpha,
                      margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'AAEME':
        model = AAEME(wv_shapes, args.emb_dim, args.wv_weights,
                      alpha=args.alpha, margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'SED':
        model = SED(wv_shapes, args.emb_dim, args.wv_weights,
                    alpha=args.alpha, margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'SVD':
        meta_dict = get_svd_vectors(vectors, args.emb_dim)
        save_vectors(meta_dict, args.result_path)
        exit()
    elif args.model == 'CONCAT':
        meta_dict = get_concat_vectors(vectors)
        save_vectors(meta_dict, args.result_path)
        exit()

    else:
        raise NotImplementedError

    model.to(torch.device('cuda'))
    thesaurus = None
    if args.thes_constrains:
        try:
            thesaurus = RuThes(args.thes_path) if args.ruthes else RuWordNet(args.thes_path)
        except:
            thesaurus = EnWordNet(args.thes_path)
    vocab, _, matrix, = get_cooc_vectors(vectors)
    matrix = np.array(matrix)

    if args.dev_size is not None:
        assert args.dev_size > 0.0 and args.dev_size < 1.0
        train_indices, dev_indices = train_test_split(range(len(vocab)), test_size=args.dev_size, random_state=42)
    else:
        train_indices = None
        dev_dataset = None

    print('Creating Train Dataset')
    train_dataset = create_dataset(vocab, matrix, train_indices, thes_constrains=args.thes_constrains, thesaurus=thesaurus, constrains_count=args.constrains_count)
    if args.dev_size is not None:
        print('Creating Dev Dataset')
        dev_dataset = create_dataset(vocab, matrix, dev_indices, thes_constrains=args.thes_constrains, thesaurus=thesaurus, constrains_count=args.constrains_count)

    sampler = RandomSampler(train_dataset)
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=1024, num_workers=args.num_workers)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=1024, num_workers=args.num_workers)

    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(loader) * epochs
    )

    print('Start Training')
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for batch_idx, batch in tqdm(enumerate(loader)):
            batch = [b for b in batch]
            if args.thes_constrains:
                inputs, pos_constrains, neg_constrains = batch
            else:
                inputs = batch
            inputs = [d.to(torch.device('cuda')) for d in inputs]
            
            if args.thes_constrains and epoch >=  epochs // 2:
                pos_constrains = [d.to(torch.device('cuda')) for d in pos_constrains]
                neg_constrains = [d.to(torch.device('cuda')) for d in neg_constrains]
            else:
                pos_constrains = None
                neg_constrains = None

            loss = model(inputs, pos_constrains, neg_constrains)

            if args.thes_constrains and epoch >= epochs // 2 and len(loss) == 3:
                loss, loss_direct, loss_constrains = loss

                epoch_losses.append((float(loss.detach().cpu().numpy()), float(loss_direct.detach().cpu().numpy()), float(loss_constrains.detach().cpu().numpy())))
            else:
                epoch_losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step()

        print(np.mean(epoch_losses, axis=0))
        eval(args, model, dev_loader, epoch)
        if epoch % args.logging_every_epochs == 0:
            model.eval()



    model.eval()
    eval(args, model, dev_loader, epoch)
    meta_dict = {}
    with torch.no_grad():
        for i, w in tqdm(enumerate(vocab)):
            wv_list = [torch.tensor(matrix[wv_i][i].reshape(1, matrix[wv_i][i].shape[0])).to(torch.device('cuda')) for wv_i in range(len(vectors))]
            meta = model.extract(wv_list)
            meta_dict[f'{lang}_{w}'] = meta[0].detach().cpu().numpy()

    meta_dict = normalise_word_vectors(meta_dict)
    save_vectors(meta_dict, args.result_path)

