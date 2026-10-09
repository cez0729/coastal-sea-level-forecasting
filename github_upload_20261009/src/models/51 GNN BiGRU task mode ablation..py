import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# 支持 direct 和 multi 两种预测任务
# ============================================================
class SeaLevelDataset(Dataset):
    def __init__(self, x, y, tide, window, horizon, start, end, mode="direct"):
        """
        x:    [T, N, F]
        y:    [T, N]
        tide: [T, N]

        mode="direct":
            只预测未来第 horizon 小时
            输出 y[t+horizon-1]
            shape: [N, 1]

        mode="multi":
            预测未来 1 到 horizon 小时整段
            输出 y[t:t+horizon]
            shape: [N, horizon]
        """

        self.x = x
        self.y = y
        self.tide = tide
        self.window = window
        self.horizon = horizon
        self.mode = mode

        self.indices = np.arange(start + window, end - horizon + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x_window = self.x[t - self.window:t]  # [window, N, F]

        if self.mode == "multi":
            y_target = self.y[t:t + self.horizon].T       # [N, horizon]
            tide_target = self.tide[t:t + self.horizon].T # [N, horizon]

        elif self.mode == "direct":
            y_target = self.y[t + self.horizon - 1]       # [N]
            tide_target = self.tide[t + self.horizon - 1] # [N]

            y_target = y_target[:, None]                  # [N, 1]
            tide_target = tide_target[:, None]            # [N, 1]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return (
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            torch.tensor(tide_target, dtype=torch.float32),
        )


# ============================================================
# Graph Convolution
# 修复版：正确处理 [B, T, N, F]
# ============================================================
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x:   [B, T, N, F]
        adj: [N, N]
        """

        if x.shape[2] != adj.shape[0]:
            raise ValueError(
                f"Node mismatch: x has {x.shape[2]} nodes, "
                f"but adj has {adj.shape[0]} nodes."
            )

        # 正确写法：
        # adj: i,j
        # x:   b,t,j,f
        # out: b,t,i,f
        x = torch.einsum("ij,btjf->btif", adj, x)

        return self.linear(x)


# ============================================================
# GNN-BiGRU 模型
# ============================================================
class GNNBiGRU(nn.Module):
    def __init__(self, input_dim, gnn_hidden, gru_hidden, output_steps, dropout=0.15):
        super().__init__()

        self.gcn1 = GraphConv(input_dim, gnn_hidden)
        self.gcn2 = GraphConv(gnn_hidden, gnn_hidden)

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, output_steps),
        )

    def forward(self, x, adj):
        """
        x: [B, T, N, F]
        """

        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(torch.relu(self.gcn2(h, adj)))

        b, t, n, f = h.shape

        # 每个站点单独进入 BiGRU
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, f)

        out, _ = self.gru(h)

        # 当前版本先用最后一个时间步
        last = out[:, -1, :]

        pred = self.head(last)

        return pred.reshape(b, n, -1)


# ============================================================
# 训练函数
# ============================================================
def train_one_epoch(model, loader, adj, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb, adj)
        loss = loss_fn(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# 验证函数
# ============================================================
@torch.no_grad()
def evaluate(model, loader, adj, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb, adj)
        loss = loss_fn(pred, yb)

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# 单个实验
# ============================================================
def run_experiment(
    x,
    y,
    tide,
    adj,
    horizon,
    mode,
    window,
    epochs,
    batch_size,
    lr,
    device,
):
    """
    x:    [T, N, F]
    y:    [T, N]
    tide: [T, N]
    adj:  [N, N]
    """

    print("=" * 70)
    print(f"Running experiment: mode={mode}, horizon={horizon}h")
    print(f"x shape    = {x.shape}")
    print(f"y shape    = {y.shape}")
    print(f"tide shape = {tide.shape}")
    print(f"adj shape  = {adj.shape}")
    print("=" * 70)

    if x.shape[1] != adj.shape[0]:
        raise ValueError(
            f"Node mismatch before training: x nodes={x.shape[1]}, adj nodes={adj.shape[0]}"
        )

    T = len(x)

    train_end = int(T * 0.7)
    val_end = int(T * 0.85)

    train_ds = SeaLevelDataset(
        x=x,
        y=y,
        tide=tide,
        window=window,
        horizon=horizon,
        start=0,
        end=train_end,
        mode=mode,
    )

    val_ds = SeaLevelDataset(
        x=x,
        y=y,
        tide=tide,
        window=window,
        horizon=horizon,
        start=train_end,
        end=val_end,
        mode=mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # direct 模式只输出 1 个未来点
    # multi 模式输出 horizon 个未来点
    output_steps = horizon if mode == "multi" else 1

    model = GNNBiGRU(
        input_dim=x.shape[-1],
        gnn_hidden=64,
        gru_hidden=64,
        output_steps=output_steps,
        dropout=0.15,
    ).to(device)

    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            adj=adj_tensor,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            adj=adj_tensor,
            loss_fn=loss_fn,
            device=device,
        )

        if val_loss < best_val:
            best_val = val_loss

        print(
            f"[{mode}-{horizon}h] "
            f"epoch={epoch:03d} "
            f"train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} "
            f"best_val_mse={best_val:.6f}"
        )

    return best_val


# ============================================================
# 构造假数据
# 你现在先用这个跑通 direct / multi 任务逻辑
# 后面再接入真实 NOAA 数据
# ============================================================
def build_fake_data():
    T = 2000
    N = 7
    F = 10

    x = np.random.randn(T, N, F).astype(np.float32)
    y = np.random.randn(T, N).astype(np.float32)
    tide = np.random.randn(T, N).astype(np.float32)

    # 先用 identity graph
    adj = np.eye(N, dtype=np.float32)

    return x, y, tide, adj


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["direct", "multi"],
        default="direct",
        help="direct: predict only t+h; multi: predict t+1 ... t+h",
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        help="run both direct and multi",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y, tide, adj = build_fake_data()

    print("Data loaded.")
    print(f"x shape    = {x.shape}")
    print(f"y shape    = {y.shape}")
    print(f"tide shape = {tide.shape}")
    print(f"adj shape  = {adj.shape}")

    results = []

    modes = ["direct", "multi"] if args.run_all else [args.mode]

    for mode in modes:
        for horizon in [1, 12, 24]:
            best_val = run_experiment(
                x=x,
                y=y,
                tide=tide,
                adj=adj,
                horizon=horizon,
                mode=mode,
                window=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )

            results.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "best_val_mse": best_val,
                }
            )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for r in results:
        print(
            f"mode={r['mode']}, "
            f"horizon={r['horizon']}h, "
            f"best_val_mse={r['best_val_mse']:.6f}"
        )


if __name__ == "__main__":
    main()