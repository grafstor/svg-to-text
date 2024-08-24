import random
import torch
from torch.utils.data import Dataset
import svg_converter as converter

def apply_random_adj(vec):
    active_dots = vec[:,-1] == -1
    rvec = 0.004*torch.randn_like(vec[active_dots, :2])
    vec[active_dots, :2] += rvec - rvec.roll(1, 0)

def apply_random_roll(vec):
    active_dots = vec[:,-1] == -1
    random_roll = random.randint(0, active_dots.sum())

    vec[active_dots] = vec[active_dots].roll(random_roll, 0)

    vec[random_roll,:2] += -vec[active_dots,:2].sum(dim=0)

def apply_random_div(vec):
    active_dots = vec[:,-1] == -1
    n = active_dots.sum()
    divs_props = torch.tensor([[0.64, 0.25, 0.08, 0.03]]*n)
    b = torch.multinomial(divs_props, num_samples=1) + 1
    b = b.flatten()
    indices = torch.arange(len(b))
    result = indices.repeat_interleave(b)
    result = result[:vec.size(0)]
    
    vec_doubles_scale = (1/b)[result]

    vec[:result.size(0)] = vec[result]
    vec[:result.size(0), :6] *= vec_doubles_scale.reshape(-1, 1)


class SvgDataset(Dataset):
    def __init__(self, svg_paths, seq_len, device, transforms=False):
        self.device = device

        self.seq_len = seq_len

        self.svg_paths = svg_paths

        self.converter = converter.Converter(self.seq_len)

        self.transforms = transforms

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
            if self.transforms:
                apply_random_roll(vector)
                apply_random_div(vector)
                apply_random_adj(vector)
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
                    if self.transforms:
                        apply_random_roll(vector)
                        apply_random_div(vector)
                        apply_random_adj(vector)
                    return (
                      vector,
                      svg_name,
                    )
