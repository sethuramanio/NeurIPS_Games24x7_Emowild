import sys
import os
import argparse
import torch
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
sys.path.append('OFA/')
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
import pandas as pd
from pprint import pprint
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-gpuIds', '--gpuIds', type=list, default=[0], help='GPU ID used for testing')
parser.add_argument('-checkpointV', '--checkpointV', type=str, 
                    default='modelCheckpoints/fusionNet_EmoWild_Valence.ckpt', help='Valence prediction model')
parser.add_argument('-checkpointA', '--checkpointA', type=str, 
                    default='modelCheckpoints/fusionNet_EmoWild_Arousal.ckpt', help='Arousal prediction model')

args = parser.parse_args()

# Register caption task
tasks.register_task('caption',CaptionTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# Load pretrained ckpt & config
overrides={"bpe_dir":"OFA/utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths('OFA/checkpoints/caption.pt'),
        arg_overrides=overrides
    )

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda('cuda:0')
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']
    
class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()
        self.aggregate = torch.nn.Conv1d(in_channels=908, out_channels=768, kernel_size=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ofa_features, be_features):
        ofa_features = self.aggregate(ofa_features)
        x = torch.matmul(be_features, ofa_features)
        x = x[:, -1, :]
        out = self.linear_relu_stack(x)
        return out

# Build the model
fusionNet = FusionNetwork().to(args.gpuIds[0])
fusionNet = torch.nn.DataParallel(fusionNet, device_ids=args.gpuIds)

dir = "imageSamples"
for root, _, fnames in sorted(os.walk(dir)):
    for imgname in fnames:
        path = os.path.join(root, imgname)
        image = Image.open(path) #image path
        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
        # Run eval step for image features and caption
        with torch.no_grad():
            features, scores = eval_step(task, generator, models, sample, "feature")
            result, scores = eval_step(task, generator, models, sample, "caption")
        ofaFeature = features["encoder_out"][0].cpu().detach().numpy()
        ofaFeature = np.squeeze(ofaFeature)
        caption = result[0]['caption']
        
        be = get_sentence_embeding([caption])
        beFeature = be.numpy()
        
        fusionNet.load_state_dict(torch.load(args.checkpointV), strict=False)
        fusionNet.eval()
        
        with torch.no_grad():
            f1 = torch.from_numpy(ofaFeature)
            f1 = f1.unsqueeze_(0)
            f2 = torch.from_numpy(beFeature)
            f2 = f2.unsqueeze_(0)
            f1 = f1.to(args.gpuIds[0])
            f2 = f2.to(args.gpuIds[0])
            out = fusionNet(f1.float(), f2.float())
            out = out.to(args.gpuIds[0])
            print("Image Valence (", imgname, "): ", out)
            
        fusionNet.load_state_dict(torch.load(args.checkpointA), strict=False)
        fusionNet.eval()
        
        with torch.no_grad():
            f1 = torch.from_numpy(ofaFeature)
            f1 = f1.unsqueeze_(0)
            f2 = torch.from_numpy(beFeature)
            f2 = f2.unsqueeze_(0)
            f1 = f1.to(args.gpuIds[0])
            f2 = f2.to(args.gpuIds[0])
            out = fusionNet(f1.float(), f2.float())
            out = out.to(args.gpuIds[0])
            print("Image Arousal (", imgname, "): ", out)