import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor, m: torch.Tensor):
        exp_z = torch.exp(z).sum(dim=1, keepdim=True)
        exp_m = torch.exp(m).sum(dim=1, keepdim=True)
        denominator = exp_z + exp_m

        prob_positive = exp_z / (denominator + 1e-10)
        loss = -torch.log(prob_positive + 1e-10).sum()

        labels = torch.tensor([1, 0], dtype=torch.float32, device=z.device).repeat(z.size(0), 1)
        return loss, prob_positive, labels


class SUGRLLoss_prototype(nn.Module):
    def __init__(self, my_margin: float, my_margin2: float) -> None:
        super().__init__()
        self.my_margin = my_margin
        self.my_margin_2 = my_margin + my_margin2

    def forward(self, h_p: torch.Tensor, h_n: torch.Tensor, h_a: torch.Tensor) -> torch.Tensor:
        margin_loss = torch.nn.MarginRankingLoss(margin=self.my_margin, reduce=False)
        s_p = F.pairwise_distance(h_a, h_p)
        s_n = F.pairwise_distance(h_a, h_n)

        margin_label = -1 * torch.ones_like(s_p)
        loss_mar = margin_loss(s_p, s_n, margin_label).sum()

        lbl_z = torch.tensor([0.0], device=h_a.device)
        mask_margin_n = torch.max((s_n - s_p.detach() - self.my_margin_2), lbl_z).sum()

        w_loss1 = 0.1
        w_loss3 = 0.1
        return loss_mar * w_loss1 + mask_margin_n * w_loss3
