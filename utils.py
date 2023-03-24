import torch
from safetensors.numpy import load_file


def load_labels(file_path):

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        labels = [line.strip() for line in f.readlines()]
        
    return labels


def load_labels_and_features(label_dir, device):

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