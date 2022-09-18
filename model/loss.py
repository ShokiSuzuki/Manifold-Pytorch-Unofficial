# Based on https://github.com/facebookresearch/deit/blob/main/losses.py

import torch
from torch.nn import functional as F


class Loss(torch.nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                distillation_type: str, tau: float, output_dir: str,
                lamda_cls: float, lamda_distill: float, lamda_patch: float, hidden_stages, lambda_intra, lambda_inter, lambda_random):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.tau = tau
        self.output_dir = output_dir
        self.lamda_cls = lamda_cls
        self.lamda_distill = lamda_distill
        self.lamda_patch = lamda_patch

        self.hidden_stages = hidden_stages
        self.lambda_intra  = lambda_intra
        self.lambda_inter  = lambda_inter
        self.lambda_random = lambda_random

    def kl_div(self, output1, output2, T):
        if not isinstance(output2, torch.Tensor): # tuple to tensor
            output2 = (output2[0] + output2[1]) / 2

        loss = F.kl_div(
                    F.log_softmax(output1 / T, dim=1),
                    F.log_softmax(output2.detach() / T, dim=1),
                    reduction='batchmean',
                    log_target=True
                ) * (T * T)
        return loss

    def manifold(self, output1, output2):
        output1 = output1 / torch.norm(output1)
        output2 = output2 / torch.norm(output2)
        x1 = output1 @ output1.transpose(-2, -1)
        x2 = output2 @ output2.transpose(-2, -1)
        # L_intra = torch.norm(x1 - x2, dim=(1, 2)).mean()
        L_intra = ((x1 - x2) ** 2).sum() / output1.shape[0]

        x1 = output1.transpose(0, 1) @ output1.permute(1, 2, 0)
        x2 = output2.transpose(0, 1) @ output2.permute(1, 2, 0)
        # L_inter = torch.norm(x1 - x2, dim=(1, 2)).mean()
        L_inter = ((x1 - x2) ** 2).sum() / output1.shape[1]

        rand = torch.randperm(output1.shape[0] * output1.shape[1])
        x1 = output1.reshape(output1.shape[0] * output1.shape[1], output1.shape[2])[rand[:192]]
        x2 = output2.reshape(output2.shape[0] * output2.shape[1], output2.shape[2])[rand[:192]]
        # L_random = torch.norm(x1 - x2)
        L_random = ((x1 - x2) ** 2).sum()

        return self.lambda_intra * L_intra + self.lambda_inter * L_inter + self.lambda_random * L_random

    def forward(self, samples, model_id, logits, patches, labels):
        T = self.tau

        logits1    = logits[model_id]
        logits1_kd = None
        if not isinstance(logits1, torch.Tensor):
            logits1, logits1_kd = logits1
        patch1 = patches[model_id]

        base_loss = self.base_criterion(logits1, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_logits, teacher_patch = self.teacher_model(samples)

        distillation_loss = 0.0
        if logits1_kd is not None:
            if self.distillation_type == 'soft':
                distillation_loss += self.kl_div(logits1_kd, teacher_logits.detach(), T)
            else:
                distillation_loss += F.cross_entropy(logits1_kd, teacher_logits.detach().argmax(dim=1))
        else:
            if self.distillation_type == 'soft':
                distillation_loss += self.kl_div(logits1, teacher_logits.detach(), T)
            else:
                distillation_loss += F.cross_entropy(logits1, teacher_logits.detach().argmax(dim=1))

        patch_loss = 0.0
        for i, j in self.hidden_stages:
            patch_loss += self.manifold(patch1[i], teacher_patch[j].detach())

        loss = self.lamda_cls * base_loss + self.lamda_distill * distillation_loss + self.lamda_patch * patch_loss

        return loss, base_loss, distillation_loss, patch_loss
