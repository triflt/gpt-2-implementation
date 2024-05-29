import torch

def get_batch(split, train_data, val_data, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def log_gpu_usage(writer, step):
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9 
        writer.add_scalar('GPU/Memory Allocated', gpu_memory_allocated, step)
        writer.add_scalar('GPU/Memory Reserved', gpu_memory_reserved, step) 