"""Utility functions supporting the other modules."""

import torch


def norm_std(x):
    x = x - x.mean()
    s = x.std()
    if s == 0:
        return torch.ones_like(x) * 0.5
    else:
        return x / s


def norm_01(x):
    x = x - x.min()
    mx = x.max()
    if mx == 0:
        return torch.ones_like(x) * 0.5
    else:
        return x / mx


def norm_img_01(x):
    B, C, H, W = x.shape
    x = x - torch.min(x.reshape(B, C, H * W), dim=-1).values.reshape(B, C, 1, 1)
    x = x / torch.max(x.reshape(B, C, H * W), dim=-1).values.reshape(B, C, 1, 1)
    return x


def compute_cos_sim(
    vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False
) -> torch.FloatTensor:
    eps = 1e-8

    if vec1.ndim != 2:
        raise ValueError(f"vec1 must be 2D (N, C), got shape {tuple(vec1.shape)}")

    N, C1 = vec1.shape
    vec1_norm = vec1 / (vec1.norm(dim=1, keepdim=True) + eps)  # (N, C)

    if vec2.ndim == 2:
        M, C2 = vec2.shape
        if C1 != C2:
            raise ValueError(
                f"vec1 feature dimension '{C1}' does not match vec2 feature dimension '{C2}'"
            )
        vec2_norm = vec2 / (vec2.norm(dim=1, keepdim=True) + eps)  # (M, C)
        sim = vec2_norm @ vec1_norm.t()  # (M, N)
        return torch.softmax(100 * sim, dim=-1) if softmax else sim

    elif vec2.ndim == 4:
        B, C2, H, W = vec2.shape
        if C1 != C2:
            raise ValueError(
                f"vec1 feature dimension '{C1}' does not match vec2 feature dimension '{C2}'"
            )
        vec2_norm = vec2 / (vec2.norm(dim=1, keepdim=True) + eps)  # (B, C, H, W)
        HW = H * W
        vec2_perm = vec2_norm.view(B, C2, HW).permute(0, 2, 1)  # (B, HW, C)
        # batch matmul: (B, HW, C) @ (C, N) -> (B, HW, N)
        sim_b_hw_n = torch.matmul(vec2_perm, vec1_norm.t())  # (B, HW, N)
        sim = sim_b_hw_n.permute(0, 2, 1).view(B, N, H, W)  # (B, N, H, W)
        return torch.softmax(100 * sim, dim=1) if softmax else sim

    else:
        raise ValueError(
            f"vec2 must be 2D (M, C) or 4D (B, C, H, W), got shape {tuple(vec2.shape)}"
        )


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 3:
        hex_code = "".join([char * 2 for char in hex_code])
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
