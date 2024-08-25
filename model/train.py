
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt

from .eval import *
from .vocab import *

# src_pad_token = torch.tensor([0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#            0.0000,  0.0000,  0.0000,  1.0000]).to(device)

def get_src_pad_mask(matrix, pad_token):
    mask = torch.zeros(matrix.size(0), matrix.size(1))

    pad_positions = (matrix == pad_token).all(dim=-1)
    
    pad_positions = torch.roll(pad_positions, shifts=1, dims=1) 
    pad_positions[:, 0] = 0
    
    mask[pad_positions] = 1
    
    return mask

def plot_metrics(metrics):
    plt.figure(figsize=(12,4))
    for i, (name, history) in enumerate(sorted(metrics.items())):
        plt.subplot(1, len(metrics), i + 1)
        plt.title(name)
        plt.plot(*zip(*history))
        plt.grid()
    plt.show()

def load_checkpoint(model, opt, save_path, name, version, device):
    checkpoint = torch.load(f'{save_path}/checkpoint_{name}_{version}.pth', map_location=device)

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore the vocabulary
    model.voc = Vocab(
        tokens=checkpoint['vocab_tokens'],
        bos=checkpoint['vocab_bos'],
        eos=checkpoint['vocab_eos'],
        unk=checkpoint['vocab_unk']
    )

    # Update step and initialize metrics
    step = checkpoint['step'] + 1
    metrics = {'train_loss': [], 'test_loss': []}
    
    return step, metrics

def save_checkpoint(save_path, name, version, model, opt, step, loss):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,
        'vocab_tokens': model.voc.tokens,  # Save the tokens from model.voc
        'vocab_token_to_ix': model.voc.token_to_ix,  # Save the token_to_ix mapping from model.voc
        'vocab_bos': model.voc.bos,  # Save the bos token
        'vocab_eos': model.voc.eos,  # Save the eos token
        'vocab_unk': model.voc.unk  # Save the unk token
    }, f'{save_path}/checkpoint_{name}_{version}.pth')

def train(model,
          opt,
          loss_fn,
          train_dl,
          test_dl,
          device,
          metrics,
          save_path,
          step,
          converter,
          epochs=10_000,
         ):
    for epoch in range(epochs):
        for vec, tgt in train_dl:
            torch.cuda.empty_cache()
            model.train()
            step += 1

            vec = vec.to(device)
            tgt = model.voc.to_matrix(tgt).to(device)

            tgt_input = tgt[:,:-1]
            tgt_expected = tgt[:,1:]

            tgt_pad_mask = model.voc.compute_mask(tgt_input).to(device)
            tgt_mask = get_tgt_mask(tgt_input.size(1)).to(device)

            out = model(vec, tgt_input,
                         tgt_mask=tgt_mask,
                         tgt_pad_mask=tgt_pad_mask)

            out = out.permute(1, 2, 0)

            loss = loss_fn(out, tgt_expected)

            opt.zero_grad()
            loss.backward()
            opt.step()

            metrics['train_loss'].append((step, loss.item()))

            if step%500 == 0:
                save_checkpoint(save_path,
                                'epoch',
                                f'{step}',
                                model=model,
                                opt=opt,
                                step=step,
                                loss=loss.item(),
                                )

            if step%100 == 0:
                with torch.no_grad():
                    opt.zero_grad()
                    model.eval()

                    for vec, tgt in test_dl:
                        vec = vec.to(device)
                        tgt = model.voc.to_matrix(tgt).to(device)

                        tgt_input = tgt[:,:-1]
                        tgt_expected = tgt[:,1:]

                        tgt_pad_mask = model.voc.compute_mask(tgt_input).to(device)
                        tgt_mask = get_tgt_mask(tgt_input.size(1)).to(device)

                        out = model(vec, tgt_input,
                                     tgt_mask=tgt_mask,
                                     tgt_pad_mask=tgt_pad_mask)

                        out = out.permute(1, 2, 0)

                        test_loss = loss_fn(out, tgt_expected)

                        break

                    metrics['test_loss'].append((step, test_loss.item()))
                    clear_output()

                    print(f'Epoch: {epoch}, step: {step}, loss: {loss.item()}, test_loss: {test_loss.item()}')
                    plot_metrics(metrics)

                    try:
                        sample(converter, model, test_dl,device=device, p_value=0.5)
                    except:
                        pass

                    torch.cuda.empty_cache()
