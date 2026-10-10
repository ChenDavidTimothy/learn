"""Cycle 02 neural controls: same hidden rules and same pre-change examples."""
import json
import math
import os
import statistics
import time
import numpy as np
import torch
from torch import nn
from cycle02_unknown_structure import (
    NOISE, PRE, POST, OUT, D, CHECKPOINTS, SEED0, XALL,
    ENTROPY_NOISE, one_trial, observe, rule_world
)
import random

torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEURAL_TRIALS = int(os.getenv("NEURAL_TRIALS", "12"))
PRE_STEPS = int(os.getenv("NEURAL_PRE_STEPS", "240"))
POST_STEPS = int(os.getenv("NEURAL_POST_STEPS", "125"))
LR = 0.003
METHODS = ("shared_replay", "shared_finetune",
           "modular_replay", "modular_finetune")

class ModularNet(nn.Module):
    """Output-wise independent trainable nonlinear functions."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(OUT, D, 48)*math.sqrt(2/D))
        self.b1 = nn.Parameter(torch.zeros(OUT, 48))
        self.w2 = nn.Parameter(torch.randn(OUT, 48)*math.sqrt(2/48))
        self.b2 = nn.Parameter(torch.zeros(OUT))

    def forward(self, x):
        h = torch.relu(torch.einsum("bd,odh->boh", x, self.w1) + self.b1)
        return (h*self.w2.unsqueeze(0)).sum(-1)+self.b2

def new_model(method):
    if method.startswith("shared"):
        net = nn.Sequential(
            nn.Linear(D, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, OUT))
    else:
        net = ModularNet()
    return net.to(DEVICE)

def fit_steps(model, optimizer, old_x, old_y, new_x, new_y,
              count, replay, rng):
    """Fixed, predeclared gradient budget for every method."""
    model.train()
    old_n, new_n = len(old_x), (len(new_x) if new_x is not None else 0)
    for _ in range(count):
        if new_n:
            npost = 48 if replay else 64
            j = rng.integers(0, new_n, npost)
            x = new_x[j]
            y = new_y[j]
            if replay:
                i = rng.integers(0, old_n, 16)
                x = torch.cat((x, old_x[i]), 0)
                y = torch.cat((y, old_y[i]), 0)
        else:
            i = rng.integers(0, old_n, 64)
            x, y = old_x[i], old_y[i]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def score_nn(net, world, changed, test_indices):
    net.eval()
    x = torch.from_numpy(XALL[test_indices]).to(DEVICE)
    p = net(x).sigmoid().clamp(min=1e-6, max=1-1e-6).T
    q = torch.from_numpy(
        (NOISE+(1-2*NOISE)*world[:, test_indices]).astype(np.float32)
    ).to(DEVICE)
    losses = -(q*torch.log(p)+(1-q)*torch.log1p(-p))
    per_output = (losses.mean(1)-ENTROPY_NOISE).cpu().numpy()
    return {
        "all": float(per_output.mean()),
        "changed": float(per_output[changed].mean()),
        "retained": float(np.delete(per_output, changed).mean()),
    }

def run_trial(seed, condition, models=METHODS):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    old, new, changed = rule_world(rng, condition)
    old_indices = np_rng.integers(0, len(XALL), size=PRE)
    old_labels = np.array([observe(old, int(x), np_rng)
                           for x in old_indices], dtype=np.float32)
    test_indices = np.random.default_rng(seed+998800).choice(
        len(XALL), size=512, replace=True)

    # Common post-change queries to every neural method, random policy.
    q_rng = np.random.default_rng(seed+221133)
    post_indices = q_rng.integers(0, len(XALL), size=POST)
    post_labels = []
    for t, ix in enumerate(post_indices):
        ob_rng = np.random.default_rng(
            seed*1000003 + (t+1)*10009 + int(ix)*71)
        post_labels.append(observe(new, int(ix), ob_rng))
    post_labels = np.asarray(post_labels, dtype=np.float32)

    old_x = torch.from_numpy(XALL[old_indices]).to(DEVICE)
    old_y = torch.from_numpy(old_labels).to(DEVICE)
    p_x = torch.from_numpy(XALL[post_indices]).to(DEVICE)
    p_y = torch.from_numpy(post_labels).to(DEVICE)

    result = {}
    for i, method in enumerate(models):
        torch.manual_seed(seed*13+123)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(seed*13+123)
        net = new_model(method)
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR,
                                      weight_decay=0.01)
        s_rng = np.random.default_rng(seed*917+i*7431+2001)
        fit_steps(net, optimizer, old_x, old_y, None, None,
                  PRE_STEPS, False, s_rng)
        history = {0: score_nn(net, new, changed, test_indices)}
        for t0, t1 in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:]):
            fit_steps(net, optimizer, old_x, old_y,
                      p_x[:t1], p_y[:t1], POST_STEPS,
                      method.endswith("replay"), s_rng)
            history[t1] = score_nn(net, new, changed, test_indices)
        result[method] = history
    return result

def main():
    start = time.perf_counter()
    data = {"meta": {"device": str(DEVICE), "torch": torch.__version__,
          "trials": NEURAL_TRIALS, "pre_steps": PRE_STEPS,
          "post_steps_per_checkpoint": POST_STEPS,
          "lr": LR, "batch_size": 64,
          "replay_pre_fraction": 0.25,
          "inputs": PRE, "post_queries": POST,
          "same_prechange_data_as_cycle02": True,
          "same_random_queries_across_neural_methods": True,
          "neural_query_policy": "random, not active"},
          "conditions": {}}
    for condition in ("in_class", "dense_threshold"):
        print("starting", condition, flush=True)
        runs = []
        for i in range(NEURAL_TRIALS):
            runs.append(run_trial(SEED0+i, condition))
            if (i+1)%4==0:
                print(" finished", i+1, "trials",
                      round(time.perf_counter()-start, 2),
                      "sec", flush=True)
        summaries = {}
        for method in METHODS:
            summaries[method] = {}
            for t in CHECKPOINTS:
                summaries[method][str(t)] = {
                    key: float(statistics.mean(
                        r[method][t][key] for r in runs))
                    for key in ("all", "changed", "retained")}
        data["conditions"][condition] = summaries
    data["runtime_seconds"] = time.perf_counter()-start
    out = "/home/dtdav/dev/learn/research/cycle02_neural_results.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print("saved:", out, "runtime:", data["runtime_seconds"])
    for condition, models in data["conditions"].items():
        print("CONDITION:", condition)
        for name, metrics in models.items():
            print(name, [round(metrics[str(t)]["all"], 5)
                         for t in CHECKPOINTS],
                  "changed:", round(metrics["48"]["changed"], 5),
                  "retained:", round(metrics["48"]["retained"], 5))

if __name__ == "__main__":
    main()
