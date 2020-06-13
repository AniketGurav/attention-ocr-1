import torch
import torch.nn as nn

import random
from PIL import Image

from torchvision import transforms

from captcha.image import ImageCaptcha

from model.attention_ocr import OCR
from utils.tokenizer import Tokenizer
from utils.img_util import display_images

img_width = 160
img_height = 60

nh = 512

device = 'cpu'

chars = list('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
gen = ImageCaptcha(img_width, img_height)
n_chars = 7

tokenizer = Tokenizer(chars)
model = OCR(img_width, img_height, nh, tokenizer.n_token,
                n_chars + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)

model.load_state_dict(torch.load('./chkpoint/time_2020-06-12_23-54-56_epoch_12.pth'))

img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
])

content = [random.randrange(0, len(chars)) for _ in range(n_chars)]
s = ''.join([chars[i] for i in content])
d = gen.generate(s)
d = Image.open(d)

model.eval()
with torch.no_grad():
    t = img_trans(d)    
    print(f'tensor shape: {t.shape}')
    print(f'unsqueezed tensor shape: {t.unsqueeze(0).shape}')
    pred = model(t.unsqueeze(0))
    
rst = tokenizer.translate(pred.squeeze(0).argmax(1))
print(rst)

display_images(t.numpy(), 1, 3)