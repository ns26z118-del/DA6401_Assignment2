 
import torch
import torch.nn as nn


class CustomDropout(nn.Module):
  

    def __init__(self, p: float = 0.5):
      
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
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
 