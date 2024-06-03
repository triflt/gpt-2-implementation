import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from gpt import GPTLanguageModel
from utils import log_gpu_usage, estimate_loss, get_batch

config = Config()

torch.manual_seed(1337)

with open('data/big_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

writer = SummaryWriter(log_dir='runs/GPT2')

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = GPTLanguageModel(vocab_size=vocab_size)
m = model.to(config.device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

if __name__ == '__main__':
    for iter in tqdm(range(config.max_iters)):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)
            log_gpu_usage(writer, iter)

        xb, yb = get_batch('train', train_data, val_data, config)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'models/model_{losses["val"]}.pth')

    writer.close()