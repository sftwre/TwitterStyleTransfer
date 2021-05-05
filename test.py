import torch
import numpy as np
from trainer.vae import VAE
from trainer.dataset import TwitterDataset
import argparse


class Beam(object):
    """
    Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
    elements after every insertion operation. Insertion is O(n) (list is maintained in
    sorted order), access is O(1). Still fast enough for practical purposes for small beams.
    """
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

    def add(self, elt, score):
        """
        Adds the element to the beam with the given score if the beam has room or if the score
        is better than the score of the worst element currently on the beam

        :param elt: element to add
        :param score: score corresponding to the element
        """
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list contains the item with a lower score, remove it
        i = 0
        while i < len(self.elts):
            if self.elts[i] == elt and score > self.scores[i]:
                del self.elts[i]
                del self.scores[i]
            i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def head(self):
        return self.elts[0]


def beam_search(soft_words, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    myBeam = Beam(5)
    mytensor = soft_words[0]
    row = mytensor.tolist()
    for j in range(len(row)):
        if row[j] != 0.0:
            print(np.log(row[j]))
            myBeam.add([j], np.log(row[j]))
    for y in range(1, len(soft_words)):
        tensor = soft_words[y]
        row = tensor.tolist()
        newBeam = Beam(5)
        # expand each current candidate
        for ele, score in myBeam.get_elts_and_scores():
            for j in range(len(row)):
                    if row[j] != 0.0:
                        newBeam.add(ele + [j], score + np.log(row[j]))
        # order all candidates by score
        myBeam = newBeam
        print(myBeam.get_elts())

    return myBeam.get_elts()


def main(args):

    h_dim = 64
    z_dim = 64
    c_dim = 4

    account = args.account
    n_tweets = args.n_tweets
    beam = args.beam_search
    k = args.k

    accounts = ['elon', 'dril', 'donald', 'dalai']

    device = 'cuda:0' if args.gpu else 'cpu'

    model_path = './models/tweet_gen.pt'

    dataset = TwitterDataset(gpu=args.gpu)

    model = VAE(dataset.vocab_size, h_dim, z_dim, c_dim, gpu=args.gpu)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # load model on correct device
    model.to(device)

    dalai = np.array([1, 0, 0, 0]).reshape(1, -1)
    donald = np.array([0, 1, 0, 0]).reshape(1, -1)
    elon = np.array([0, 0, 1, 0]).reshape(1, -1)
    dril = np.array([0, 0, 0, 1]).reshape(1, -1)

    # get code for account
    if account not in accounts:
        print(f'{account} not supported yet :(')
        exit(-1)

    if account == 'donald':
        code = donald
    elif account == 'dalai':
        code = dalai
    elif account == 'dril':
        code = dril
    else:
        code = elon

    # generate tweets for account
    c = torch.FloatTensor(code).to(device)
    _, c_idx = torch.max(c, dim=1)

    print(f'Twitter account: @{dataset.idx2label(int(c_idx))}')

    import math
    from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
    # Load pre-trained model (weights)
    pmodel = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    pmodel.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score(sentence):
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss=pmodel(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)

    total_ratio = 0.0
    total_perp = 0.0
    for _ in range(n_tweets):
        # Samples latent and conditional codes randomly from prior
        z = model.sample_z_prior(1)

        sample_idxs = model.sample_sentence(z, c, beam=beam, temp=0.1)

        # use beam search to find k most likely sequences
        if beam:
            seqs = beam_search(sample_idxs, k)
            print(dataset.idxs2sentence(seqs[0][0]))
        else:
            sent = dataset.idxs2sentence(sample_idxs)
            print(sent)

            total_perp += score(sent)
            total_ratio += len(set(sent.split()))/len(sent.split())
    print("Perplexity: " + str(total_perp/n_tweets))
    print("Ratio: " + str(total_ratio/n_tweets))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tweet generation')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to run in the GPU')
    parser.add_argument('--n_tweets', default=100, type=int, help='number of tweets to generate')
    parser.add_argument('--account', required=True, type=str, help='account to generate tweets for')
    parser.add_argument('--beam_search', default=False, type=bool, help='whether to perform beam search or not')
    parser.add_argument('--k', required=False, type=int, help='size of beam')
    args = parser.parse_args()
    main(args)