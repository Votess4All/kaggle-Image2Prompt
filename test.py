from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from model.blip import setup_blip_model
from model.clip import set_up_clip_model
from model.sentence_transformer import set_up_stmodel
from utils import load_labels_and_features, truncate_to_fit


# 整理思路
# 1. 对image提特征，使用clip的image-text retrieval的思路，从一个已有库中提取出当前image对应的text的描述（在作者的代码这里分成了三个部分，medium/movement/flavor）
# 2. 将上面得到的结果和BLIP产生的image capation进行结合，最后利用一定后处理将结果结合起来即可。
# 3. 可以尝试改进的地方

class CFG:
    label_dir = "/kaggle/input/skt-clip-interrogator/skt-clip-interrogator/labels/CLIP-ViT-H-14-laion2B-s32B-b79K/"
    batch_size = 128
    blip_caption_gen_kwargs = {"max_length": 20, "min_length": 5}
    st_model_path = "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"
    

class ImgDataset(Dataset):
    def __init__(self, image_paths, captions, preprocess):
        self.image_paths = image_paths
        self.captions = captions
        self.preprocess = preprocess

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        processed_image = self.preprocess(image)
        caption = self.captions[idx]
        return processed_image, caption


def interrogate(image_features, caption, clip_tokenizer, label_dir, device):

    labels_and_features = load_labels_and_features(label_dir, device) 
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


def get_images_captions(model, processor, image_paths, device, batch_size, gen_kwargs):
    
    model.to(device)

    blip_data_loader = DataLoader(image_paths, batch_size=batch_size, shuffle=False)
    blip_captions = []

    for batch in tqdm(blip_data_loader):

        images = []
        for image_path in batch:
            i_image = Image.open(image_path).convert("RGB")
            images.append(i_image)
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values=pixel_values, **gen_kwargs)
        captions = processor.batch_decode(out, skip_special_tokens=True)

        blip_captions.extend(captions)
        
    model.to("cpu")
    
    return blip_captions 
    

def get_pred_prompts(model, processor, tokenizer, image_paths, captions, device, label_dir, batch_size):
    
    model.to(device) 
        
    dataset = ImgDataset(image_paths, captions, processor) 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_prompts = []

    for batch in tqdm(data_loader):
        
        processed_images, captions = batch
        processed_images = processed_images.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(processed_images)

        for image_feature, caption in zip(image_features, captions):
            prompt = interrogate(image_feature, caption, tokenizer, label_dir, device)
            pred_prompts.append(prompt)
    

    return pred_prompts


def encode_prompts_with_stmodel(prompts):
    st_model = set_up_stmodel(CFG.st_model_path)
    prompt_embeddings = st_model.encode(prompts['prompt']).flatten()
    
    return prompt_embeddings


def main():
    
    images_root = './images'
    image_paths = glob(f'{images_root}/*')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blip_processor, blip_model = setup_blip_model()
    clip_model, clip_tokenizer, clip_preprocess = set_up_clip_model(device)

    blip_captions = get_images_captions(
        blip_model, blip_processor, image_paths, device, 
        CFG.batch_size, CFG.blip_caption_gen_kwargs) 

    pred_prompts = get_pred_prompts(
        clip_model, clip_preprocess, clip_tokenizer, 
        image_paths, blip_captions, device, CFG.label_dir, CFG.batch_size) 
    
    prompts_embeds = encode_prompts_with_stmodel(pred_prompts)

if __name__ == "__main__":
    main()