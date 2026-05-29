import numpy as np
import torch


def federated_averaging(global_model_state_dict, participant_models_state_dict):
    """
    Aggregates the state dict from all participant models using Federated Averaging.
    """
    num_participants = len(participant_models_state_dict)
    if num_participants == 0:
        raise ValueError("No participant models for averaging.")
    averaged_state_dict = {}
    for key in global_model_state_dict.keys():
        averaged_state_dict[key] = sum(
            p[key] for p in participant_models_state_dict
        ) / num_participants
    return averaged_state_dict


class FedYogi:
    """
    FedYogi server-side adaptive optimizer (Reddi et al., 2020).

    Key differences from plain Adam / the broken original:
    - Moment buffers persist across aggregation rounds (state is NOT reset per call).
    - A single global step counter drives bias correction uniformly for all layers.
    - The v update uses FedYogi's sign-gated rule:  v += (1-β₂)·sign(Δ²−v)·Δ²
      instead of Adam's exponential smoothing, which prevents v from shrinking
      back toward zero when the pseudo-gradient is small.
    """

    def __init__(self):
        self.moment_1 = None  # first moment (momentum)
        self.moment_2 = None  # second moment (adaptive)
        self.step = 0

    def __call__(self, global_model_state_dict, participant_models_state_dicts, **kwargs):
        learning_rate = kwargs.get('learning_rate', 0.01)
        beta1 = kwargs.get('yogi_beta1', 0.9)
        beta2 = kwargs.get('yogi_beta2', 0.999)
        epsilon = kwargs.get('yogi_epsilon', 1e-3)

        num_participants = len(participant_models_state_dicts)
        if num_participants == 0:
            raise ValueError("No participant models for FedYogi.")

        # Initialise moment buffers on first aggregation round
        if self.moment_1 is None:
            self.moment_1 = {k: torch.zeros_like(v) for k, v in global_model_state_dict.items()}
            self.moment_2 = {k: torch.zeros_like(v) for k, v in global_model_state_dict.items()}

        # Pseudo-gradient: mean of (local_weights - global_weights)
        delta = {}
        for key in global_model_state_dict:
            delta[key] = sum(
                p[key] - global_model_state_dict[key]
                for p in participant_models_state_dicts
            ) / num_participants

        self.step += 1

        new_state = {}
        for key, grad in delta.items():
            # First moment update (momentum)
            self.moment_1[key] = beta1 * self.moment_1[key] + (1 - beta1) * grad

            # Second moment update — FedYogi rule (sign-gated, not Adam)
            grad_sq = grad ** 2
            self.moment_2[key] = self.moment_2[key] + (1 - beta2) * torch.sign(
                grad_sq - self.moment_2[key]
            ) * grad_sq

            # Bias correction using the same global step for all layers
            m_hat = self.moment_1[key] / (1 - beta1 ** self.step)
            v_hat = self.moment_2[key] / (1 - beta2 ** self.step)

            new_state[key] = global_model_state_dict[key] + learning_rate * m_hat / (
                torch.sqrt(torch.clamp(v_hat, min=0.0)) + epsilon
            )

        return new_state

    def reset(self):
        """Call this if the global model is re-initialised mid-experiment."""
        self.moment_1 = None
        self.moment_2 = None
        self.step = 0


def fed_median(global_model_state_dict, participant_models_state_dicts, **kwargs):
    """
    Coordinate-wise median aggregation (Byzantine-robust).

    For each parameter tensor, takes the element-wise median across all
    participant models instead of the mean.  A single adversarial or
    corrupted client can shift the mean arbitrarily, but can only shift
    the median by at most one rank — making this robust to up to
    floor((n-1)/2) Byzantine participants out of n.

    This is particularly relevant for SereBench because the platform
    explicitly injects cyberattacks and can poison local model weights.
    """
    num_participants = len(participant_models_state_dicts)
    if num_participants == 0:
        raise ValueError("No participant models for FedMedian.")

    aggregated = {}
    for key in global_model_state_dict.keys():
        stacked = torch.stack([p[key].float() for p in participant_models_state_dicts], dim=0)
        aggregated[key] = torch.median(stacked, dim=0).values

    return aggregated
