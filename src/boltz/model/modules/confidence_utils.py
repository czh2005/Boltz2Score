import torch
from torch import nn

from boltz.data import const
from boltz.model.loss.confidence import compute_frame_pred


def compute_aggregated_metric(logits, end=1.0):
    """Compute the metric from the logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits of the metric
    end : float
        Max value of the metric, by default 1.0

    Returns
    -------
    Tensor
        The metric value

    """
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def tm_function(d, Nres):
    """Compute the rescaling function for pTM.

    Parameters
    ----------
    d : torch.Tensor
        The input
    Nres : torch.Tensor
        The number of residues

    Returns
    -------
    Tensor
        Output of the function

    """
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(logits, x_preds, feats, multiplicity):
    """Compute pTM and ipTM scores.

    Parameters
    ----------
    logits : torch.Tensor
        pae logits
    x_preds : torch.Tensor
        The predicted coordinates
    feats : Dict[str, torch.Tensor]
        The input features
    multiplicity : int
        The batch size of the diffusion roll-out

    Returns
    -------
    Tensor
        pTM score
    Tensor
        ipTM score
    Tensor
        ligand ipTM score
    Tensor
        protein ipTM score

    """
    # Compute mask for collinear and overlapping tokens
    _, mask_collinear_pred = compute_frame_pred(
        x_preds, feats["frames_idx"], feats, multiplicity, inference=True
    )
    mask_pad = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    
    is_multi_chain = (asym_id[:, None, :] != asym_id[:, :, None]).any()

    # Extract pae values
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    ).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)

    # compute pTM and ipTM
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)
    probs = nn.functional.softmax(logits, dim=-1)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )  # shape (B, N, N)
    
    if is_multi_chain:
        pair_mask_iptm = (
            maski[:, :, None]
            * (asym_id[:, None, :] != asym_id[:, :, None])
            * mask_pad[:, None, :]
            * mask_pad[:, :, None]
        )
        iptm = torch.max(
            torch.sum(tm_expected_value * pair_mask_iptm, dim=-1)
            / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
            dim=1,
        ).values
        ptm = torch.zeros_like(iptm)
    else:
        pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
        ptm = torch.max(
            torch.sum(tm_expected_value * pair_mask_ptm, dim=-1)
            / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
            dim=1,
        ).values
        iptm = torch.zeros_like(ptm)

    # compute ligand and protein ipTM
    ligand_iptm = torch.zeros_like(iptm)
    protein_iptm = torch.zeros_like(iptm)
    chain_pair_iptm = {}

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
