import sys

import torch

from transformers import (
    CrossEntropyLoss,
    LabelSmoothingLoss,
    Transformer,
    get_subsequent_mask,
    position_encoding_sinusoid,
)


def train_one_step(model, optimizer, ques_b, ques_pos, ans_b, ans_pos, use_label_smoothing=False):
    model.train()
    optimizer.zero_grad()

    dec_out = model(ques_b, ques_pos, ans_b, ans_pos)
    pred = dec_out.reshape(-1, dec_out.shape[-1])

    if use_label_smoothing:
        loss = LabelSmoothingLoss(pred, ans_b[:, 1:])
    else:
        target = ans_b[:, 1:].reshape(-1)
        loss = CrossEntropyLoss(pred, target)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def greedy_decode(model, ques_b, ques_pos, bos_idx, eos_idx=None, max_len=10):
    model.eval()

    q_emb = model.emb_layer(ques_b)
    q_emb_inp = q_emb + ques_pos
    enc_out = model.encoder(q_emb_inp)

    generated = torch.full(
        (ques_b.shape[0], 1), bos_idx, dtype=torch.long, device=ques_b.device
    )

    for _ in range(max_len):
        pos = position_encoding_sinusoid(generated.shape[1], q_emb.shape[-1]).to(
            generated.device
        )
        pos = pos.expand(generated.shape[0], -1, -1)

        dec_inp = model.emb_layer(generated) + pos
        mask = get_subsequent_mask(generated)
        dec_out = model.decoder(dec_inp, enc_out, mask)
        next_token = dec_out[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if eos_idx is not None and torch.all(next_token.squeeze(1) == eos_idx):
            break

    return generated


def demo_shapes():
    vocab_len = 16
    num_heads = 2
    emb_dim = 8
    feedforward_dim = 16
    dropout = 0.1
    num_enc_layers = 2
    num_dec_layers = 2

    model = Transformer(
        num_heads=num_heads,
        emb_dim=emb_dim,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        vocab_len=vocab_len,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N, K, O = 2, 5, 6
    ques_b = torch.randint(0, vocab_len, (N, K))
    ans_b = torch.randint(0, vocab_len, (N, O))
    ques_pos = position_encoding_sinusoid(K, emb_dim).expand(N, -1, -1)
    ans_pos = position_encoding_sinusoid(O, emb_dim).expand(N, -1, -1)

    loss = train_one_step(model, optimizer, ques_b, ques_pos, ans_b, ans_pos)
    generated = greedy_decode(model, ques_b, ques_pos, bos_idx=0, eos_idx=1, max_len=O)

    print("train loss:", loss)
    print("generated shape:", tuple(generated.shape))
    print(generated)


if __name__ == "__main__":
    if "D:/CS/UMICH/A5" not in sys.path:
        sys.path.insert(0, "D:/CS/UMICH/A5")
    demo_shapes()

