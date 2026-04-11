"""Slice K=32 soft prompts/projected to first 128 for fair comparison with K=16/K=64."""
import torch

sp = torch.load("data/soft_prompts_K32.pt", weights_only=False)
sp128 = {
    "soft_prompts": sp["soft_prompts"][:128],
    "final_cos": sp["final_cos"][:128],
    "final_losses": sp["final_losses"][:128],
    "K": sp["K"],
    "loss_type": sp["loss_type"],
    "steps": sp["steps"],
    "lr": sp["lr"],
}
torch.save(sp128, "data/soft_prompts_K32_first128.pt")

pr = torch.load("data/projected_K32.pt", weights_only=False)
pr128 = {
    "projected_ids": pr["projected_ids"][:128],
    "decoded": pr["decoded"][:128],
    "K": pr["K"],
    "final_cos": pr["final_cos"][:128],
}
torch.save(pr128, "data/projected_K32_first128.pt")
print("sliced, K=32 first-128 stats: mean_cos =", sp128["final_cos"].mean().item())
