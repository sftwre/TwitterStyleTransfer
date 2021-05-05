import os
import torch
import argparse
from trainer.vae import VAE
import torch.optim as optim
import torch.nn.functional as F
from trainer.dataset import TwitterDataset
from torch.utils.tensorboard import SummaryWriter

def main(args):

    # tensorboard writer
    writer = SummaryWriter()

    # hyperparams
    beta = args.beta
    lambda_c = args.lambda_c
    lambda_z = args.lambda_z
    lambda_u = args.lambda_u
    kl_weight_max = 0.4
    lr = args.lr

    epochs = args.epochs
    batch_size = args.batch_size
    gpu = args.gpu
    z_dim = 64
    h_dim = 64
    report_interval = 100

    dataset = TwitterDataset(gpu=gpu)

    # controllable parameter for each account
    c_dim = dataset.n_accounts

    model = VAE(dataset.vocab_size, h_dim, z_dim, c_dim, gpu=gpu)

    device = 'cpu' if not gpu else 'cuda:0'

    # load pre-trained generator
    model.load_state_dict(torch.load('../models/vae.pt', map_location=torch.device(device)))

    optim_G = optim.Adam(model.decoder_params, lr=lr)
    optim_D = optim.Adam(model.discriminator_params, lr=lr)
    optim_E = optim.Adam(model.encoder_params, lr=lr)

    # discriminator learning with wake-sleep algorithm
    for e in range(epochs):

        interval = 0

        # reset data loader
        dataset.resetTrainBatches()

        # generate samples and classify sentences
        for inputs, labels in dataset.trainIterator:

            """update params of discriminator, sleep phase"""
            b_size = inputs.size(1)

            x_gen, c_gen = model.generate_sentences(b_size)
            c_hat = torch.argmax(c_gen, dim=1)

            # classify twitter handle of sentences
            y_hat_real = model.forwardDiscriminator(inputs.transpose(0, 1))
            y_hat_gen = model.forwardDiscriminator(x_gen)

            # entropy used to obtain high confidence in predictions
            entropy = F.log_softmax(y_hat_gen, dim=1).mean()
            entropy = -entropy

            # supervised loss for semantic meaning
            loss_s = F.cross_entropy(y_hat_real, labels)
            # unsupervised loss
            loss_u = F.cross_entropy(y_hat_gen, c_hat)

            loss_d = loss_s + lambda_u * (loss_u + beta * entropy)

            writer.add_scalar('Discriminator/discriminator', loss_d, e)

            loss_d.backward()
            optim_D.step()
            optim_D.zero_grad()

            """ update params of generator, sleep phase """
            model.train()

            recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)

            x_gen_attr, target_z, target_c = model.generate_soft_embed(batch_size)

            """
            Feed soft generated sentence to discriminator
            to measure fitness to the target attribute.
            """
            y_z, *_ = model.forwardEncoderEmb(x_gen_attr.transpose(0, 1))
            y_c = model.forwardDiscEmbed(x_gen_attr)

            loss_vae = recon_loss + kl_weight_max * kl_loss
            loss_attr_c = F.cross_entropy(y_c, target_c)
            loss_attr_z = F.mse_loss(y_z, target_z)

            loss_g = loss_vae + lambda_c*loss_attr_c + lambda_z*loss_attr_z
            writer.add_scalar('Discriminator/generator', loss_g, e)

            loss_g.backward()

            optim_G.step()
            optim_G.zero_grad()

            """ update params of encoder, wake phase """
            recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)

            loss_e = recon_loss + kl_weight_max * kl_loss

            writer.add_scalar('Discriminator/encoder', loss_e, e)

            loss_e.backward()

            optim_E.step()
            optim_E.zero_grad()

            if interval % report_interval == 0:
                z = model.sample_z_prior(1)
                c = model.sample_c_prior(1)

                sample_idxs = model.sample_sentence(z, c)
                sample_sent = dataset.idxs2sentence(sample_idxs)

                print(f'Epoch-{e}; loss_D: {loss_d.item():.4f}; loss_G: {loss_g.item():.4f}')

                _, c_idx = torch.max(c, dim=1)

                print(f'c = {dataset.idx2label(int(c_idx))}')
                print(f'Sample: {sample_sent}')
                writer.add_text('Generator', sample_sent, e)

            interval += 1
            
    # save model parameters
    saveModel(model)
    writer.flush()
    writer.close()

def saveModel(model):

    if not os.path.exists('../models/'):
        os.makedirs('models/')

    PATH = '../models/tweet_gen.pt'

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--batch_size', default=32, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--lambda_c', default=0.1, type=float)
    parser.add_argument('--lambda_z', default=0.1, type=float)
    parser.add_argument('--lambda_u', default=0.1, type=float)
    parser.add_argument('--gpu', default=False, type=bool, help='Flag to run model on gpu')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')

    args = parser.parse_args()

    main(args)