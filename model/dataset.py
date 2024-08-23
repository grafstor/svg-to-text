import random
import torch
from torch.utils.data import Dataset
import svg_converter as converter

class SvgDataset(Dataset):
    def __init__(self, svg_paths, seq_len, device):
        self.device = device

        self.seq_len = seq_len

        self.svg_paths = svg_paths

        self.converter = converter.Converter(self.seq_len)

    def __len__(self):
        return len(self.svg_paths)

    def load_svg(self, path):
        try:
            _, vector = self.converter.to_vector(self.converter.open(path), is_one_len=True)
            return vector
        
        except Exception as e:
            return None

    def __getitem__(self, idx):
        svg_name, svg_path = self.svg_paths[idx]
        vector = self.load_svg(svg_path)

        if vector:
            vector = torch.tensor(vector)
            return (
              vector,
              svg_name,
            )
        else:
            while True:
                svg_name, svg_path = random.choice(self.svg_paths)
                vector = self.load_svg(svg_path)

                if vector:
                    vector = torch.tensor(vector)
                    return (
                      vector,
                      svg_name,
                    )
