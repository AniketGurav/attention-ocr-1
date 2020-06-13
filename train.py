import argparse
import random
import time
import pickle

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from model.attention_ocr import OCR
from utils.dataset import CaptchaDataset
from utils.train_util import train_batch, eval_batch

DEVICE = 'cpu'


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(n_epoch=100, max_len=4, batch_size=32, n_works=4, save_checkpoint_every=5):
    img_width = 160
    img_height = 60
    nh = 512

    teacher_forcing_ratio = 0.5    
    lr = 3e-4    

    ds_train = CaptchaDataset(img_width, img_height, 10000, max_len)
    ds_test = CaptchaDataset(img_width, img_height, 1000, max_len)

    tokenizer = ds_train.tokenizer

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_works)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=n_works)

    model = OCR(img_width, img_height, nh, tokenizer.n_token,
                max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=DEVICE)

    load_weights = torch.load('./inception_v3_google-1a9a5a14.pth')

    names = set()
    for k, w in model.incept.named_children():
        names.add(k)

    weights = {}
    for k, w in load_weights.items():
        if k.split('.')[0] in names:
            weights[k] = w

    model.incept.load_state_dict(weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss().cuda()

    def train_epoch():
        sum_loss_train = 0
        n_train = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)

            loss, acc, sentence_acc = train_batch(x, y, model, optimizer,
                                                  crit, teacher_forcing_ratio, max_len,
                                                  tokenizer)

            sum_loss_train += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_train += 1

        return sum_loss_train / n_train, sum_acc / n_train, sum_sentence_acc / n_train

    def eval_epoch():
        sum_loss_eval = 0
        n_eval = 0
        sum_acc = 0
        sum_sentence_acc = 0

        for bi, batch in enumerate(tqdm(test_loader)):
            x, y = batch
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)

            loss, acc, sentence_acc = eval_batch(x, y, model, crit, max_len, tokenizer)

            sum_loss_eval += loss
            sum_acc += acc
            sum_sentence_acc += sentence_acc

            n_eval += 1

        return sum_loss_eval / n_eval, sum_acc / n_eval, sum_sentence_acc / n_eval

    for epoch in range(n_epoch):
        train_loss, train_acc, train_sentence_acc = train_epoch()
        eval_loss, eval_acc, eval_sentence_acc = eval_epoch()

        print("Epoch %d" % epoch)
        print('train_loss: %.4f, train_acc: %.4f, train_sentence: %.4f' % (train_loss, train_acc, train_sentence_acc))
        print('eval_loss:  %.4f, eval_acc:  %.4f, eval_sentence:  %.4f' % (eval_loss, eval_acc, eval_sentence_acc))

        if epoch % save_checkpoint_every == 0 and epoch > 0:
            print('saving checkpoint...')
            torch.save(model.state_dict(), './chkpoint/time_%s_epoch_%s.pth' % (time.strftime('%Y-%m-%d_%H-%M-%S'), epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python train_.py --m=time_2020-06-12_19-31-05_epoch_10.pth --e=1 --cuda --v')
    parser.add_argument('--m', help='Path to a previous model to start with')
    parser.add_argument('--e', type=int, nargs='?', const=100, default=100, help='Number of epochs to train the model')
    parser.add_argument('--cuda', type=str2bool, nargs='?', const=False, default=False, help='use CUDA if available')
    parser.add_argument('--l', type=int, nargs='?', const=7, default=7, help='Max number of characters in captcha')
    parser.add_argument('--c', type=int, nargs='?', const=5, default=5, help='Save model every given number of epochs (checkpoint)')
    args = parser.parse_args()
            
    NUM_EPOCHS = args.e if args.e is not None else 100
    MAX_LEN = args.l if args.l is not None else 7
    CHECKPOINT = args.c if args.c is not None else 5

    print(f'Number of epochs: {NUM_EPOCHS}')

    main(n_epoch=NUM_EPOCHS, max_len=MAX_LEN, n_works=8, save_checkpoint_every=CHECKPOINT)
