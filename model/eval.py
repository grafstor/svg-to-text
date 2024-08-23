
import io
import svgwrite
from PIL import Image

from IPython.display import SVG, display

import torch
import torch.nn.functional as F

def get_tgt_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def predict(model, input_sequence, device, p_value=0.5, max_length=15):
    model.eval()
    
    y_input = torch.tensor([[model.voc.bos_ix]]*input_sequence.size(0), dtype=torch.long, device=device)

    for _ in range(max_length):
        
        tgt_mask = get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        pred = pred.permute(1, 0, 2)
        pred = pred[:,-1,:]
        pred = pred.squeeze(-1)

        pred = F.softmax(pred, dim=1)
        
        next_item = sample_top_p(pred, p_value).to(device)

        y_input = torch.cat((y_input, next_item), dim=1)

    y_input = model.voc.to_lines(y_input)

    return y_input

def decode_bpe(encoded_sentence):
    return encoded_sentence.replace("@@ ", "").replace("@@", "").strip()

def sample(converter, model, dl, device, p_value=0.5):
    model.eval()
    with torch.no_grad():
        for vec, tgt in dl:
            results = predict(model, vec.to(device), device=device, p_value=p_value)
    
            for i in range(vec.size(0)):
                show(converter, vec[i].tolist())
                
                result = decode_bpe(results[i])
                target = decode_bpe(tgt[i])
                
                print("target:", target)
                print("result:", result)
            break
                        
def show(converter, vec, scale=150):
    svg = converter.to_svg(vec, scale=scale)
    dwg = svgwrite.Drawing(
        '',
        profile='tiny',
        fill_rule="evenodd",
        size=(f"{svg['width']}px", f"{svg['height']}px")
    )

    dwg.add(
        dwg.path( d=svg['d'].d(),
        fill="#000")
    )
    temp_file = io.BytesIO()
    temp_file.write(dwg.tostring().encode('utf-8'))

    display(SVG(temp_file.getvalue()))
