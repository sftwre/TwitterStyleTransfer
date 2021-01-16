import argparse

import torch
import random
from vae import VAE
import numpy as np
from dataset import TwitterDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acctToTensor(account:str, k:int):
    """
    converts an account name passed in as an argument
    to a tensor representing a controllable parameter
    :param account:
    :param k: beam width
    :return:
    """

    accounts = ['elon', 'dril', 'donald', 'dalai']

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

    # convert account code to tensor
    c = torch.FloatTensor(code).to(device)
    _, dim = c.shape
    c = c.repeat(k, 1)

    return c

def beam_search(model:VAE, dataset:TwitterDataset, account:str, beam_size=3):
    """
    Performs beam search to compose tweets with given account.

    :param model: trained vae
    :param dataset:
    :param beam_size:
    :return: candidate sequences with highest scores
    """

    k = beam_size
    vocab_size = dataset.vocab_size

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([dataset.TEXT.vocab.stoi['<start>']] * k).to(device)  # (k, 1)

    # (seqLen, batch)
    k_prev_words = k_prev_words.unsqueeze(0)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    # z prior
    z = model.sample_z_prior(k)
    c = acctToTensor(account, k)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        *_, cell_state = model.forwardEncoder(k_prev_words)

        logits = model.forwardDecoder(k_prev_words, z, c, cell_state)  # (seqLen, bsize, vocab_size)

        scores = F.log_softmax(logits, dim=1)

        # update scores
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores.topk(k, 2, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != dataset.TEXT.vocab.stoi['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    return seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tweet generation, beam search')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to run in the GPU')
    parser.add_argument('--n_tweets', default=20, type=int, help='number of tweets to generate')
    parser.add_argument('--account', required=True, type=str, help='account to generate tweets for')
    args = parser.parse_args()

    model_path = './models/tweet_gen.pt'

    # init dataset
    dataset = TwitterDataset(gpu=args.gpu)

    # load model
    model = VAE(dataset.vocab_size, gpu=args.gpu)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    seqs = beam_search(model, dataset, args.account)
    print(seqs)
