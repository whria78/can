import torch
import dnnlib
import legacy
import PIL.Image
import numpy as np
#import imageio
#from tqdm.notebook import tqdm
import sys
import argparse

parser = argparse.ArgumentParser(description='CAN5600 & GAN5000 dataset; Han Seung Seog')

parser.add_argument('--source', type=str, default="", help='source image e.g /out_source_seed0001/projected_w.npz)')
parser.add_argument('--target', type=str, default="", help='target image e.g /out_source_seed0002/projected_w.npz)')
parser.add_argument('--network', type=str, default="mn500.pkl", help='Trained GAN model (mn500.pkl by default)')
parser.add_argument('--step', type=int, default=10, help='generate N morphing images')

parser.add_argument('--outdir', type=str, default="", help='outdir')


#source_=sys.argv[1]
#target_=sys.argv[2]
#lvec1 = np.load('./out_source_'+source_+'/projected_w.npz')['w']
#lvec2 = np.load('./out_source_'+target_+'/projected_w.npz')['w']
lvec1 = np.load(parser.source)['w']
lvec2 = np.load(parser.target)['w']

network_pkl = parser.network #"/home/ai/nvme/training-runs2/mn500_noaug.pkl"
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

diff = lvec2 - lvec1
step = diff / parser.step
current = lvec1.copy()
target_uint8 = np.array([512,512,3], dtype=np.uint8)

import os
#outdir='/home/ai/Dropbox/pub/result_'+source_ +'_'+ target_
outdir=parser.outdir
try:os.makedirs(outdir)
except:pass


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

#for j in tqdm(range(parser.step)):
for j in range(parser.step):
  z = torch.from_numpy(current).to(device)
  synth_image = G.synthesis(z, noise_mode='const')
  synth_image = (synth_image + 1) * (255/2)
  synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

  img_path=f'{outdir}/proj{j:02d}.png'
  img = PIL.Image.fromarray(synth_image, 'RGB')
  img.save(img_path)
   
  current = current + step



