"""Reusable custom layers for DA6401 Assignment 2."""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Inverted dropout — implemented from scratch without nn.Dropout or F.dropout.

    During training:
        1. Sample a binary mask from Bernoulli(1 - p) for each element.
        2. Zero out elements where mask == 0.
        3. Scale surviving elements by 1/(1-p) so the expected value is
           unchanged at test time (inverted / inference-time dropout).

    During eval (self.training = False):
        Pass the input through unchanged.

    Design note: CustomDropout is placed BEFORE each large FC layer in the
    classifier head. This forces the encoder to learn redundant feature
    representations rather than relying on specific neurons, which is the
    core regularisation effect. Placing it after FC (but before activation)
    is also valid, but pre-FC placement is empirically more effective for
    large linear layers (Srivastava et al., 2014).
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of zeroing an element. Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape [B, ...].
        Returns:
            Output tensor of same shape.
        """
        # Eval mode: pass through unchanged
        if not self.training:
            return x

        # p == 0 means keep everything; short-circuit for efficiency
        if self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p

        # Bernoulli mask: 1 = keep, 0 = drop
        # torch.full creates a tensor of keep_prob values, same shape as x
        # torch.bernoulli samples 1 with probability keep_prob for each element
        mask = torch.bernoulli(
            torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype)
        )

        # Inverted dropout: divide by keep_prob to preserve expected activation magnitude
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"

# """Reusable custom layers 
# """

# import torch
# import torch.nn as nn


# class CustomDropout(nn.Module):
#     """Custom Dropout layer.
#     """

#     def __init__(self, p: float = 0.5):
#         """
#         Initialize the CustomDropout layer.

#         Args:
#             p: Dropout probability.
#         """
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the CustomDropout layer.

#         Args:
#             x: Input tensor for shape [B, C, H, W].

#         Returns:
#             Output tensor.
#         """
#         # TODO: implement dropout.
#         raise NotImplementedError("Implement CustomDropout.forward")
