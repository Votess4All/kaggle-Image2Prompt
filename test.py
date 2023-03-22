from glob import glob
import os
from tqdm import tqdm
import time
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, BlipForConditionalGeneration
import open_clip
from safetensors.numpy import load_file


# 整理思路
# 1. 对image提特征，使用clip的image-text retrieval的思路，从一个已有库中提取出当前image对应的text的描述（在作者的代码这里分成了三个部分，medium/movement/flavor）
# 2. 将上面得到的结果和BLIP产生的image capation进行结合，最后利用一定后处理将结果结合起来即可。
# 3. 可以尝试改进的地方

class CLIP_Dataset(Dataset):
    def __init__(self, image_paths, captions, preprocess):
        self.image_paths = image_paths
        self.captions = captions
        self.preprocess = preprocess

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        processed_image = self.preprocess(image).to(device)
        caption = self.captions[idx]
        return processed_image, caption


def setup_blip_model():
    # setup blip model
    blip_processor = AutoProcessor.from_pretrained("/kaggle/input/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("/kaggle/input/blip-image-captioning-large")
    
    return blip_processor, blip_model


def set_up_clip_model():
    
    # setup clip model
    clip_model = open_clip.create_model('ViT-H-14', precision='fp16' if device == 'cuda' else 'fp32')
    open_clip.load_checkpoint(clip_model, "/kaggle/input/skt-clip-interrogator/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin")
    clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
    clip_preprocess = open_clip.image_transform(
        clip_model.visual.image_size,
        is_train = False,
        mean = getattr(clip_model.visual, 'image_mean', None),
        std = getattr(clip_model.visual, 'image_std', None),
    )
    
    return clip_model, clip_tokenizer, clip_preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blip_processor, blip_model = setup_blip_model()
clip_model, clip_tokenizer, clip_preprocess = set_up_clip_model()


def load_labels(file_path):

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        labels = [line.strip() for line in f.readlines()]
        
    return labels


def load_labels_and_features(label_dir):

    mediums_labels = load_labels(f"{label_dir}mediums.txt")
    movements_labels = load_labels(f"{label_dir}movements.txt")
    flavors_labels = load_labels(f"{label_dir}flavors.txt")
        
    mediums_embeds = load_file(f"{label_dir}mediums.safetensors")['embeds']
    movements_embeds = load_file(f"{label_dir}movements.safetensors")['embeds']
    flavors_embeds = load_file(f"{label_dir}flavors.safetensors")['embeds']

    mediums_features_array = torch.stack([torch.from_numpy(t) for t in mediums_embeds]).to(device)
    movements_features_array = torch.stack([torch.from_numpy(t) for t in movements_embeds]).to(device)
    flavors_features_array = torch.stack([torch.from_numpy(t) for t in flavors_embeds]).to(device)
    
    return {
        "labels": {"medium": mediums_labels, "movements": movements_labels, "flavors": flavors_labels}, 
        "features": {"medium": mediums_features_array, "movements": movements_features_array, "flavors": flavors_features_array}
    }


def prompt_at_max_len(text, tokenize):
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def truncate_to_fit(text, tokenize):
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text


def interrogate(image_features, caption, label_dir):

    labels_and_features = load_labels_and_features(label_dir) 
    mediums_labels = labels_and_features["labels"]["medium"]
    movements_labels = labels_and_features["labels"]["movements"]
    flavors_labels = labels_and_features["labels"]["flavors"]

    mediums_features_array = labels_and_features["features"]["medium"] 
    movements_features_array = labels_and_features["features"]["movements"] 
    flavors_features_array = labels_and_features["features"]["flavors"] 

    cos = torch.nn.CosineSimilarity(dim=1)
    
    medium = [mediums_labels[i] for i in cos(image_features, mediums_features_array).topk(1).indices][0]
    movement = [movements_labels[i] for i in cos(image_features, movements_features_array).topk(1).indices][0]
    flaves = ", ".join([flavors_labels[i] for i in cos(image_features, flavors_features_array).topk(3).indices])

    if caption.startswith(medium):
        prompt = f"{caption}, {movement}, {flaves}"
    else:
        prompt = f"{caption}, {medium}, {movement}, {flaves}"

    return truncate_to_fit(prompt, clip_tokenizer)


def main():
    # 要搞清楚这个label是怎么来的
    label_dir = "/kaggle/input/skt-clip-interrogator/labels/CLIP-ViT-H-14-laion2B-s32B-b79K/"
    images_root = './images'
    image_ids = [i.split('.')[0] for i in os.listdir(images_root)]

    image_paths = glob(f'{images_root}/*')

    start_time = time.time()

    # run blip for image caption
    blip_model.to(device)

    blip_data_loader = DataLoader(image_paths, batch_size=128, shuffle=False)
    gen_kwargs = {"max_length": 20, "min_length": 5}
    blip_captions = []

    for batch in tqdm(blip_data_loader):

        images = []
        for image_path in batch:
            i_image = Image.open(image_path).convert("RGB")
            images.append(i_image)
        
        pixel_values = blip_processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = blip_model.generate(pixel_values=pixel_values, **gen_kwargs)
        captions = blip_processor.batch_decode(out, skip_special_tokens=True)

        blip_captions.extend(captions)
        
    blip_model.to("cpu")

    # run clip for image features
    clip_model.to(device) 
        
    clip_dataset = CLIP_Dataset(image_paths, blip_captions, clip_preprocess) 
    clip_data_loader = DataLoader(clip_dataset, batch_size=128, shuffle=False)

    pred_prompts = []

    for batch in tqdm(clip_data_loader):

        processed_images, captions = batch
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(processed_images)

        for image_feature, caption in zip(image_features, captions):
            prompt = interrogate(image_feature, caption, label_dir)
            pred_prompts.append(prompt)
            
    print("--- %s seconds ---" % (time.time() - start_time))
    print(pred_prompts)
    

if __name__ == "__main__":
    main()