import json
import random
import re
import math
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 1. 基本配置
# =========================
DATA_DIR = Path("data")
ROOT_OUT_DIR = Path("results_sweep")
ROOT_OUT_DIR.mkdir(exist_ok=True)

JSON_FILES = [
    "poet.song.40000.json",
    "poet.song.41000.json",
    "poet.song.42000.json",
    "poet.song.43000.json",
]

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
EMBED_DIM = 128
SEQ_LEN = 32
DROPOUT = 0.3
TOP_K = 20

# 固定 temperature = 1，不再作为扫参变量
TEMPERATURE = 1.0

# =========================
# 2. 扫参配置
# =========================
MODEL_TYPES = ["LSTM", "GRU", "RNN", "AttentionLSTM", "Transformer"]
NUM_LAYERS_LIST = [1, 2, 3]
ACTIVATION_LIST = ["Tanh", "ReLU", "GELU"]
HIDDEN_DIM_LIST = [128, 256]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 3. 固定随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("当前设备:", DEVICE)


# =========================
# 4. 数据预处理
# =========================
def clean_sentence(s):
    return re.sub(r"[^\u4e00-\u9fa5]", "", s)


def split_poem_lines(paragraphs):
    text = "".join(paragraphs)
    raw_lines = re.split(r"[，。！？；、]", text)

    lines = []
    for line in raw_lines:
        line = clean_sentence(line)
        if len(line) > 0:
            lines.append(line)

    return lines


def load_qiyan_jueju(data_dir):
    poems = []

    for file_name in JSON_FILES:
        file_path = data_dir / file_name

        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipped.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            paragraphs = item.get("paragraphs", [])

            if not paragraphs:
                continue

            lines = split_poem_lines(paragraphs)

            if len(lines) == 4 and all(len(line) == 7 for line in lines):
                poem = (
                    lines[0] + "，" + lines[1] + "。\n" +
                    lines[2] + "，" + lines[3] + "。"
                )
                poems.append(poem)

    return poems


poems = load_qiyan_jueju(DATA_DIR)

print(f"七言绝句数量: {len(poems)}")

if len(poems) == 0:
    raise ValueError("没有筛选到七言绝句，请检查 data 文件夹、json 文件名或筛选规则。")

text = "\n".join(poems)

chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

vocab_size = len(chars)

print(f"字符表大小: {vocab_size}")


# =========================
# 5. Dataset
# =========================
class PoetryDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        self.data = [char2idx[ch] for ch in text if ch in char2idx]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


dataset = PoetryDataset(text, char2idx, SEQ_LEN)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# =========================
# 6. 激活函数
# =========================
def get_activation(name):
    if name == "Tanh":
        return nn.Tanh()
    elif name == "ReLU":
        return nn.ReLU()
    elif name == "GELU":
        return nn.GELU()
    else:
        raise ValueError("ACTIVATION_NAME 必须是 Tanh, ReLU 或 GELU")


# =========================
# 7. 位置编码
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =========================
# 8. 通用模型
# =========================
class PoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        model_type,
        activation_name,
        dropout
    ):
        super().__init__()

        self.model_type = model_type
        self.activation_name = activation_name

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.activation = get_activation(activation_name)

        rnn_dropout = dropout if num_layers > 1 else 0

        if model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)

        elif model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)

        elif model_type == "RNN":
            nonlinearity = "relu" if activation_name == "ReLU" else "tanh"

            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                nonlinearity=nonlinearity
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)

        elif model_type == "AttentionLSTM":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4 if hidden_dim % 4 == 0 else 2,
                dropout=dropout,
                batch_first=True
            )

            self.norm = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        elif model_type == "Transformer":
            self.input_proj = nn.Linear(embed_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)

            nhead = 4 if hidden_dim % 4 == 0 else 2

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation=activation_name.lower() if activation_name in ["ReLU", "GELU"] else "gelu",
                batch_first=True
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )

            self.fc = nn.Linear(hidden_dim, vocab_size)

        else:
            raise ValueError("model_type 必须是 LSTM, GRU, RNN, AttentionLSTM 或 Transformer")

    def generate_square_subsequent_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

        return mask

    def forward(self, x, hidden=None):
        emb = self.embedding(x)

        if self.model_type in ["LSTM", "GRU", "RNN"]:
            out, hidden = self.rnn(emb, hidden)
            out = self.activation(out)
            logits = self.fc(out)
            return logits, hidden

        elif self.model_type == "AttentionLSTM":
            out, hidden = self.rnn(emb, hidden)

            seq_len = out.size(1)
            attn_mask = self.generate_square_subsequent_mask(seq_len, out.device)

            attn_out, _ = self.attention(
                out,
                out,
                out,
                attn_mask=attn_mask
            )

            out = self.norm(out + attn_out)
            out = self.activation(out)
            logits = self.fc(out)

            return logits, hidden

        elif self.model_type == "Transformer":
            out = self.input_proj(emb)
            out = self.pos_encoder(out)

            seq_len = out.size(1)
            mask = self.generate_square_subsequent_mask(seq_len, out.device)

            out = self.transformer(out, mask=mask)
            out = self.activation(out)
            logits = self.fc(out)

            return logits, None


# =========================
# 9. 古诗生成函数
# =========================
def generate_poem(
    model,
    start_text="明月",
    max_len=28,
    top_k=20
):
    model.eval()

    generated = list(start_text)

    input_ids = []
    for ch in start_text:
        if ch in char2idx:
            input_ids.append(char2idx[ch])

    if len(input_ids) == 0:
        raise ValueError("起始词中的字符不在词表中，请更换 start_text。")

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)

    hidden = None

    with torch.no_grad():
        while len(generated) < max_len:
            output, hidden = model(input_tensor, hidden)

            # temperature 固定为 1，因此不再除以 temperature
            logits = output[:, -1, :]

            for ch in generated[-10:]:
                if ch in char2idx:
                    logits[0, char2idx[ch]] /= 1.3

            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)

            sample_id = torch.multinomial(probs, num_samples=1).item()
            next_id = indices[0, sample_id].item()
            next_char = idx2char[next_id]

            next_char_clean = clean_sentence(next_char)

            if len(next_char_clean) == 0:
                if model.model_type == "Transformer":
                    input_tensor = torch.cat(
                        [
                            input_tensor,
                            torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)
                        ],
                        dim=1
                    )
                else:
                    input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)
                continue

            generated.append(next_char)

            if model.model_type == "Transformer":
                input_tensor = torch.cat(
                    [
                        input_tensor,
                        torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)
                    ],
                    dim=1
                )
            else:
                input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)

    content = "".join(generated)
    content = clean_sentence(content)
    content = content[:28]

    lines = [
        content[0:7],
        content[7:14],
        content[14:21],
        content[21:28],
    ]

    poem = lines[0] + "，" + lines[1] + "。\n" + lines[2] + "，" + lines[3] + "。"

    return poem


# =========================
# 10. 单组实验训练函数
# =========================
def run_experiment(
    model_type,
    num_layers,
    activation_name,
    hidden_dim
):
    set_seed(SEED)

    exp_name = f"{model_type}_layers{num_layers}_hidden{hidden_dim}_{activation_name}"
    out_dir = ROOT_OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("开始实验:", exp_name)
    print("=" * 80)

    model = PoetryModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        model_type=model_type,
        activation_name=activation_name,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = []

    epoch_poem_file = out_dir / "generated_poems_by_epoch.txt"
    loss_txt_file = out_dir / "loss_history.txt"
    loss_csv_file = out_dir / "loss_history.csv"
    config_file = out_dir / "experiment_config.txt"

    for file in [epoch_poem_file, loss_txt_file, loss_csv_file, config_file]:
        if file.exists():
            file.unlink()

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(f"model_type: {model_type}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"activation_name: {activation_name}\n")
        f.write(f"hidden_dim: {hidden_dim}\n")
        f.write(f"temperature: {TEMPERATURE}\n")
        f.write(f"embed_dim: {EMBED_DIM}\n")
        f.write(f"dropout: {DROPOUT}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"lr: {LR}\n")
        f.write(f"seq_len: {SEQ_LEN}\n")
        f.write(f"top_k: {TOP_K}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"poem_count: {len(poems)}\n")

    with open(loss_csv_file, "w", encoding="utf-8") as f:
        f.write("epoch,loss\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(
            loader,
            desc=f"{exp_name} | Epoch [{epoch}/{EPOCHS}]"
        )

        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            output, _ = model(x)

            loss = criterion(
                output.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"{exp_name} | Epoch [{epoch}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

        with open(loss_txt_file, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch}: {avg_loss:.6f}\n")

        with open(loss_csv_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                out_dir / f"poetry_{model_type}_epoch{epoch}.pt"
            )

            poem = generate_poem(
                model=model,
                start_text="明月",
                max_len=28,
                top_k=TOP_K
            )

            print(f"\n===== {exp_name} | Epoch {epoch} 生成古诗 =====")
            print(poem)

            with open(epoch_poem_file, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}:\n")
                f.write(poem + "\n\n")

    torch.save(
        model.state_dict(),
        out_dir / f"poetry_{model_type}_final.pt"
    )

    # Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(exp_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 最终生成 10 首
    generated_results = []

    for i in range(10):
        poem = generate_poem(
            model=model,
            start_text="明月",
            max_len=28,
            top_k=TOP_K
        )

        generated_results.append(poem)

    with open(out_dir / "generated_poems.txt", "w", encoding="utf-8") as f:
        for i, poem in enumerate(generated_results):
            f.write(f"生成古诗 {i + 1}：\n")
            f.write(poem + "\n\n")

    final_loss = loss_history[-1]
    best_loss = min(loss_history)

    return {
        "experiment": exp_name,
        "model_type": model_type,
        "num_layers": num_layers,
        "activation": activation_name,
        "hidden_dim": hidden_dim,
        "temperature": TEMPERATURE,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "out_dir": str(out_dir)
    }


# =========================
# 11. 自动扫参
# =========================
summary_file = ROOT_OUT_DIR / "summary.csv"

summary_results = []

for model_type in MODEL_TYPES:
    for num_layers in NUM_LAYERS_LIST:
        for activation_name in ACTIVATION_LIST:
            for hidden_dim in HIDDEN_DIM_LIST:

                # RNN 只支持 tanh / relu，跳过 GELU
                if model_type == "RNN" and activation_name == "GELU":
                    continue

                result = run_experiment(
                    model_type=model_type,
                    num_layers=num_layers,
                    activation_name=activation_name,
                    hidden_dim=hidden_dim
                )

                summary_results.append(result)

                with open(summary_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "experiment",
                            "model_type",
                            "num_layers",
                            "activation",
                            "hidden_dim",
                            "temperature",
                            "final_loss",
                            "best_loss",
                            "out_dir"
                        ]
                    )

                    writer.writeheader()

                    for row in summary_results:
                        writer.writerow(row)

print("\n全部实验完成！")
print(f"总结果表已保存到: {summary_file}")