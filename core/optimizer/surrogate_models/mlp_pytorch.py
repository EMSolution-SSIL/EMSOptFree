"""
mlp_pytorch.py
The MIT License (MIT)
Copyright © 2026 Science Solutions International Laboratory, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 ; typical symbol
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.optim import lr_scheduler

from core.optimizer.surrogate_models.protocol import SurrogateProtocol

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, num_nodes: list[int], p_dropout: float = 0.2) -> None:
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(num_nodes[i], num_nodes[i + 1]) for i in range(len(num_nodes) - 2)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes[i + 1]) for i in range(len(num_nodes) - 2)])
        self.do = nn.ModuleList([nn.Dropout(p=p_dropout) for _ in range(len(num_nodes) - 2)])
        self.output_fc = nn.Linear(num_nodes[-2], num_nodes[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, bn, do in zip(self.fc, self.bn, self.do, strict=True):
            x = do(F.leaky_relu(bn(fc(x))))
        return self.output_fc(x)


class MLPMultiHead(nn.Module):
    def __init__(self, num_nodes: list[int], p_dropout: float = 0.2) -> None:
        super().__init__()
        # shared
        self.fc = nn.ModuleList([nn.Linear(num_nodes[i], num_nodes[i + 1]) for i in range(len(num_nodes) - 3)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes[i + 1]) for i in range(len(num_nodes) - 3)])
        self.do = nn.ModuleList([nn.Dropout(p=p_dropout) for _ in range(len(num_nodes) - 3)])
        # head
        self.fc_head = nn.ModuleList([nn.Linear(num_nodes[-3], num_nodes[-2]) for _ in range(num_nodes[-1])])
        self.bn_head = nn.ModuleList([nn.BatchNorm1d(num_nodes[-2]) for _ in range(num_nodes[-1])])
        self.do_head = nn.ModuleList([nn.Dropout(p=p_dropout) for _ in range(num_nodes[-1])])
        self.output_fc_head = nn.ModuleList([nn.Linear(num_nodes[-2], 1) for _ in range(num_nodes[-1])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, bn, do in zip(self.fc, self.bn, self.do, strict=True):
            x = do(F.leaky_relu(bn(fc(x))))
        x_heads = []
        for fc, bn, do, output_fc in zip(self.fc_head, self.bn_head, self.do_head, self.output_fc_head, strict=True):
            x_head = do(F.leaky_relu(bn(fc(x))))
            output_head = output_fc(x_head)
            x_heads.append(output_head)
        return torch.cat(x_heads, dim=1)


class MLPSurrogate(SurrogateProtocol):
    def __init__(
        self,
        num_nodes: list[int] | None,
        p_dropout: float = 0.2,
        lr: float = 1e-3,
        seed: int | None = None,
        device: str | None = None,
        *,
        use_multi_head: bool = True,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if use_multi_head:
            self.model = MLPMultiHead(num_nodes, p_dropout).to(self.device)
        else:
            self.model = MLP(num_nodes, p_dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.L1Loss()
        self._has_fit = False
        self.x_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler((0.0, 1.0))

    def is_ready(self) -> bool:
        return self._has_fit

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        # remove outlier using IQR method
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        Q1 = np.percentile(y_reshaped, 25, axis=0)
        Q3 = np.percentile(y_reshaped, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = np.all((y_reshaped >= lower_bound) & (y_reshaped <= upper_bound), axis=1)
        logger.debug("Data length before masking: %d, after masking: %d", len(y_reshaped), len(y_reshaped[mask]))
        x = x[mask]
        y_reshaped = y_reshaped[mask]
        # scale data and create loader
        x_norm = self.x_scaler.fit_transform(x)
        y_norm = self.y_scaler.fit_transform(y_reshaped)
        x_tensor = torch.from_numpy(x_norm).float().to(self.device)
        y_tensor = torch.from_numpy(y_norm).float().to(self.device)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs // 5, 20), eta_min=0)

        # training
        self.model.train()
        best_loss = float("inf")
        best_state = None
        avg_losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            logger.info("Epoch - %d Loss: %.6f", epoch + 1, avg_loss)
            avg_losses.append((epoch + 1, avg_loss))
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._has_fit = True
        pd.DataFrame(avg_losses).to_csv("losses.csv", index=False)

        # output true vs predicted side-by-side for inspection
        predicted, _ = self.predict(x)
        gt = y_reshaped if y_reshaped.ndim > 1 else y_reshaped.reshape(-1, 1)
        pred = predicted if predicted.ndim > 1 else predicted.reshape(-1, 1)
        n_targets = gt.shape[1]
        cols = [f"gt_{i + 1}" for i in range(n_targets)] + [f"predicted_{i + 1}" for i in range(n_targets)]
        df = pd.DataFrame(np.hstack([gt, pred]), columns=cols)
        df.to_csv("gt_vs_predicted.csv", index=False)

    def update(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        self.fit(x, y, epochs, batch_size)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        self.model.eval()
        x_norm = self.x_scaler.transform(x)
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).float().to(self.device)
            output = self.model(x_tensor)
            mu_norm = output.cpu().numpy()
            # Only reshape if output is 1D (N,), otherwise keep as is
            if mu_norm.ndim == 1:
                mu_norm = mu_norm.reshape(-1, 1)
            mu = self.y_scaler.inverse_transform(mu_norm)
            sigma = np.zeros_like(mu)
        return mu, sigma
