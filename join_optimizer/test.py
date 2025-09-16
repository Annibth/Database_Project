"""
Compact starter: Join-order RL environment + PPO integration (stable-baselines3) with
behavior-cloning pretraining from left-deep labels.

File: join_order_rl_starter.py

How to use:
  1) pip install gym torch stable-baselines3
  2) python join_order_rl_starter.py

Notes:
 - This is a minimal, readable starting point. It flattens per-relation features into
   a fixed-size observation vector (padding to MAX_TABLES). Invalid actions are masked
   inside a custom policy so the agent cannot choose already-picked relations.
 - Behavior cloning pretraining runs using the left-deep order provided in the JSON.
 - After BC, PPO fine-tuning runs for a small number of timesteps. Adjust hyperparams
   for larger experiments.

"""

import json
import math
import random
from typing import List, Dict, Any

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# stable-baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import get_device
except Exception as e:
    raise ImportError("This script requires stable-baselines3. Install with: pip install stable-baselines3\nOriginal error: %s" % e)


# ------------------ Environment ------------------
class JoinOrderEnv(gym.Env):
    """Gym environment for join-order selection.

    Observation (flattened):
      For each slot i in 0..MAX_TABLES-1:
        [log_card, log_unfiltered_card, degree, has_unary_pred (0/1)]
      + mask vector of length MAX_TABLES (1=available, 0=already-chosen/pad)
      + prefix_length (scalar repeated MAX_TABLES times, optional)

    Action: Discrete(MAX_TABLES) -> index of relation chosen. If action points to
    a padded slot or already-chosen slot, environment will (optionally) reject it.
    But the policy provided will mask invalid actions; if not, environment returns
    a negative reward for invalid picks.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, query_json: Dict[str, Any], max_tables: int = 8, use_cost_model: bool = True):
        super().__init__()
        self.query = query_json
        self.max_tables = max_tables
        self.use_cost_model = use_cost_model

        # Build relation list and simple features
        self.relations = [r["name"] for r in query_json["relations"]]
        self.N = len(self.relations)
        if self.N > self.max_tables:
            raise ValueError(f"Number of relations {self.N} exceeds max_tables {self.max_tables}")

        # Precompute simple features per relation
        # features: [log(cardinality+1), log(unfilteredCard+1), degree(#joins), has_unary_pred]
        rel_by_name = {r["name"]: r for r in query_json["relations"]}
        degree = {r["name"]: 0 for r in query_json["relations"]}
        for j in query_json.get("joins", []):
            for a, b in zip(j["relations"][:-1], j["relations"][1:]):
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1
        # unary columns: we only check presence in query_json["unary columns"]
        unary_cols = set(query_json.get("unary columns", []))

        feats = []
        for r in self.relations:
            info = rel_by_name[r]
            card = float(info.get("cardinality", 0))
            unfiltered = float(info.get("unfilteredCardinality", card))
            has_unary = 0.0
            # crude heuristic: if any unary column references this relation name prefix
            for uc in unary_cols:
                if uc.startswith(r.split('.')[0]):
                    has_unary = 1.0
            feats.append([
                math.log1p(card),
                math.log1p(unfiltered),
                float(degree.get(r, 0)),
                has_unary,
            ])
        # pad to max_tables
        while len(feats) < self.max_tables:
            feats.append([0.0, 0.0, 0.0, 0.0])

        self.node_feats = np.array(feats, dtype=np.float32)  # shape (max_tables, F)

        # observation space: flattened nodes + mask + prefix_len
        F = self.node_feats.shape[1]
        obs_dim = self.max_tables * F + self.max_tables + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.max_tables)

        # ground truth left-deep order (as indices into relations)
        # parse order string like '((((it join mi_idx) join ct) join mc) join t)'
        self.gt_order = self._parse_left_deep_order(query_json.get("left deep tree min order", ""))

        # simple mock cost model: use provided sizes dict to get join cardinality sum as cost
        self.cost_lookup = {tuple(s["relations"]): s["cardinality"] for s in query_json.get("sizes", []) if "relations" in s}

        self.reset()

    def _parse_left_deep_order(self, order_str: str) -> List[int]:
        # fallback: use relations order if parse fails
        try:
            # a simple parser: extract tokens that match relation names
            tokens = []
            for r in self.relations:
                if r in order_str:
                    tokens.append(r)
            # ensure uniqueness and preserve occurrence order
            seen = set()
            idxs = []
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    idxs.append(self.relations.index(t))
            if len(idxs) == len(self.relations):
                return idxs
        except Exception:
            pass
        # default: return original relations order
        return list(range(len(self.relations)))

    def reset(self):
        self.chosen = []  # indices chosen so far
        # mask: 1=available, 0=chosen or padding slot
        mask = np.zeros(self.max_tables, dtype=np.float32)
        mask[: self.N] = 1.0
        self.mask = mask
        return self._get_obs()

    def _get_obs(self):
        # Flatten node feats
        flat_nodes = self.node_feats.reshape(-1)
        # mask as float vector + prefix_len
        prefix_len = float(len(self.chosen))
        obs = np.concatenate([flat_nodes, self.mask.astype(np.float32), np.array([prefix_len], dtype=np.float32)])
        return obs

    def step(self, action: int):
        done = False
        info = {}
        if action < 0 or action >= self.max_tables:
            # invalid action index
            reward = -100.0
            done = True
            return self._get_obs(), reward, done, info
        if self.mask[action] < 0.5:
            # invalid: choosing already-chosen or padded slot
            # return a negative reward and end episode to discourage invalids
            reward = -10.0
            done = True
            return self._get_obs(), reward, done, info

        # valid
        self.chosen.append(int(action))
        self.mask[action] = 0.0

        if len(self.chosen) == self.N:
            done = True
            # compute terminal cost (use simple heuristic lookup or fallback)
            plan = [self.relations[i] for i in self.chosen]
            # try to look up exact size for the full set
            key = tuple(self.relations)
            cost = None
            # try to find size entry matching full-relation set (order-insensitive)
            for s in self.query.get("sizes", []):
                if set(s.get("relations", [])) == set(self.relations):
                    cost = float(s.get("cardinality", 0))
                    break
            if cost is None:
                # fallback: sum of pairwise sizes if available
                cost = 0.0
                for i in range(len(self.chosen)):
                    for j in range(i + 1, len(self.chosen)):
                        a = self.relations[self.chosen[i]]
                        b = self.relations[self.chosen[j]]
                        # look for pair size
                        for s in self.query.get("sizes", []):
                            if set(s.get("relations", [])) == {a, b}:
                                cost += float(s.get("cardinality", 0))
                                break
            # reward is negative cost (we maximize reward)
            reward = -float(cost)
            info["cost"] = float(cost)
        else:
            reward = 0.0

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        print("chosen:", [self.relations[i] for i in self.chosen])


# ------------------ Behavior Cloning Dataset ------------------
class JoinOrderDataset(Dataset):
    def __init__(self, queries: List[Dict[str, Any]], max_tables: int = 8):
        self.queries = queries
        self.max_tables = max_tables

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        env = JoinOrderEnv(q, max_tables=self.max_tables)
        obs = env.reset()
        # target sequence: use env.gt_order padded to max_tables with -1
        tgt = env.gt_order + [-1] * (self.max_tables - len(env.gt_order))
        return obs, np.array(tgt, dtype=np.int64)


# ------------------ Custom Masking Policy for SB3 ------------------
class FlattenExtractor(BaseFeaturesExtractor):
    """A trivial extractor that returns input as features (identity)."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        # observation_space is expected to be Box
        super().__init__(observation_space, features_dim)
        # we'll not use a separate extractor network; the policy will handle it
        self._features_dim = observation_space.shape[0]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class MaskableActorCriticPolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy that reads a mask from the observation vector and
    applies it to the action logits so invalid actions have -inf probability.

    Observation format (as in env._get_obs):
      [flat_nodes (max_tables * F), mask (max_tables), prefix_len]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_mask_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, obs_dim)
        obs_dim = obs.shape[1]
        max_tables = (obs_dim - 1) - (obs_dim - 1)  # not used; we rely on policy attributes
        # Instead, we use self.observation_space to reconstruct sizes
        obs_space_dim = self.observation_space.shape[0]
        # We set these in policy initialization via policy.observation_space
        # Here we deduce max_tables by using a stored attribute
        max_tables = getattr(self, "max_tables", None)
        node_feat_dim = getattr(self, "node_feat_dim", None)
        if max_tables is None or node_feat_dim is None:
            raise RuntimeError("Policy missing max_tables/node_feat_dim attributes")
        start = max_tables * node_feat_dim
        mask = obs[:, start : start + max_tables]
        return mask

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        # obs tensor shape (batch, obs_dim)
        features = obs
        # standard feature extractor (identity)
        latent_pi = self.mlp_extractor.policy_net(features)
        latent_vf = self.mlp_extractor.value_net(features)

        # compute logits
        distribution = self._get_action_dist_from_latent(latent_pi)
        # distribution.distribution.probs shape: (batch, action_dim)
        # But we need to mask logits before sampling - unfortunately SB3's Categorical distribution
        # doesn't expose logits directly, so we re-compute logits via the last layer.

        # The ActorCriticPolicy keeps a 'action_net' which outputs logits for discrete actions
        logits = self.action_net(latent_pi)

        # apply mask: set logits of invalid actions to -1e9
        mask = self._get_mask_from_obs(obs)
        very_negative = -1e9
        logits = logits + (1.0 - mask) * very_negative

        # build new distribution from masked logits
        distributions = self.action_dist.proba_distribution.proba_distribution_from_latent(logits, None)
        # sample or choose action
        if deterministic:
            actions = distributions.mode()
        else:
            actions = distributions.sample()

        log_prob = distributions.log_prob(actions)
        values = self.value_net(latent_vf).flatten()
        return actions, values, log_prob

    def _predict(self, observation: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.as_tensor(observation[np.newaxis, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        return int(actions.cpu().numpy().squeeze())


# ------------------ Training utilities ------------------

def behavior_cloning_pretrain(policy_net: nn.Module, dataset: JoinOrderDataset, epochs: int = 5, batch_size: int = 32, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For BC we build a simple classifier that at each position predicts the next action
    # Here we reuse the policy_net by calling its internal nets, but for simplicity we
    # build a small MLP that outputs logits for each step.

    class BCModel(nn.Module):
        def __init__(self, input_dim, max_tables):
            super().__init__()
            self.max_tables = max_tables
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, max_tables * max_tables),  # predict full sequence as flattened logits
            )

        def forward(self, x):
            # x: (batch, obs_dim)
            out = self.net(x)
            # reshape to (batch, max_tables, max_tables) where out[:, t, k] is logit for choosing k at step t
            return out.view(-1, self.max_tables, self.max_tables)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset[0][0].shape[0]
    bc_model = BCModel(input_dim, dataset.max_tables).to(device)
    opt = torch.optim.Adam(bc_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    print("Starting behavior cloning pretraining for", epochs, "epochs")
    for epoch in range(epochs):
        total_loss = 0.0
        for obs, tgt in dataloader:
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
            tgt = torch.tensor(tgt, dtype=torch.long, device=device)  # shape (batch, max_tables)
            logits_seq = bc_model(obs)  # (batch, max_tables, max_tables)
            # compute CE per position
            loss = 0.0
            for t in range(dataset.max_tables):
                logits_t = logits_seq[:, t, :]
                tgt_t = tgt[:, t]
                loss = loss + loss_fn(logits_t, tgt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"BC Epoch {epoch+1}/{epochs} loss={total_loss:.4f}")

    print("BC pretraining finished.")
    return bc_model


# ------------------ Helper: build sample env from provided JSON ------------------
SAMPLE_JSON = {
 "name": "1 2",
 "relations": [
  {"name": "ct", "aliastable": "company_type", "basetable": "company_type", "cardinality": 3, "unfilteredCardinality": 4},
  {"name": "it", "aliastable": "info_type", "basetable": "info_type", "cardinality": 4, "unfilteredCardinality": 113},
  {"name": "mc", "aliastable": "movie_companies", "basetable": "movie_companies", "cardinality": 2609129, "unfilteredCardinality": 2609129},
  {"name": "mi_idx", "aliastable": "movie_info_idx", "basetable": "movie_info_idx", "cardinality": 1380035, "unfilteredCardinality": 1380035},
  {"name": "t", "aliastable": "title", "basetable": "title", "cardinality": 391666, "unfilteredCardinality": 2528312}
 ],
 "joins": [
  {"relations": ["ct", "mc"]},
  {"relations": ["t", "mc"]},
  {"relations": ["t", "mi_idx"]},
  {"relations": ["mc", "mi_idx"]},
  {"relations": ["it", "mi_idx"]}
 ],
 "sizes": [
  {"relations": ["ct", "mc"], "cardinality": 2609129},
  {"relations": ["it", "mi_idx"], "cardinality": 0},
  {"relations": ["mc", "mi_idx"], "cardinality": 4073078},
  {"relations": ["mc", "t"], "cardinality": 294635},
  {"relations": ["mi_idx", "t"], "cardinality": 131526},
  {"relations": ["ct", "mc", "mi_idx"], "cardinality": 4073078},
  {"relations": ["ct", "mc", "t"], "cardinality": 294635},
  {"relations": ["it", "mc", "mi_idx"], "cardinality": 0},
  {"relations": ["it", "mi_idx", "t"], "cardinality": 0},
  {"relations": ["mc", "mi_idx", "t"], "cardinality": 369049},
  {"relations": ["ct", "it", "mc", "mi_idx"], "cardinality": 0},
  {"relations": ["ct", "mc", "mi_idx", "t"], "cardinality": 369049},
  {"relations": ["it", "mc", "mi_idx", "t"], "cardinality": 0},
  {"relations": ["ct", "it", "mc", "mi_idx", "t"], "cardinality": 0}
 ],
 "query": "SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title, MIN(t.production_year) AS movie_year FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info_idx AS mi_idx, title AS t WHERE ct.id = mc.company_type_id AND t.id = mc.movie_id AND t.id = mi_idx.movie_id AND mc.movie_id = mi_idx.movie_id AND it.id = mi_idx.info_type_id AND ct.kind in ('special effects companies','distributors','production companies') AND it.info in ('LD supplement','adaption','birth notes','mini biography') AND t.production_year > 2010",
 "join columns": ["t.id", "mi_idx.movie_id", "mc.movie_id", "it.id", "mc.company_type_id", "ct.id", "mi_idx.info_type_id"],
 "unary columns": ["ct.kind", "t.production_year", "it.info"],
 "join expressions": [{"left": "ct.id", "right": "mc.company_type_id"}, {"left": "t.id", "right": "mc.movie_id"}, {"left": "t.id", "right": "mi_idx.movie_id"}, {"left": "mc.movie_id", "right": "mi_idx.movie_id"}, {"left": "it.id", "right": "mi_idx.info_type_id"}],
 "left deep tree min cost": "0",
 "left deep tree min order": "((((it join mi_idx) join ct) join mc) join t)",
 "bushy deep tree min cost": "0",
 "bushy deep tree min order": "((((it join mi_idx) join ct) join mc) join t)"
}


def main():
    max_tables = 8
    env = JoinOrderEnv(SAMPLE_JSON, max_tables=max_tables)

    # quick sanity rollout
    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Ground-truth left-deep order (indices):", env.gt_order)
    print("Relations:", env.relations)

    # Build dataset for BC
    dataset = JoinOrderDataset([SAMPLE_JSON], max_tables=max_tables)
    bc_model = behavior_cloning_pretrain(None, dataset, epochs=3, batch_size=1)

    # Now create SB3 env wrapper (vectorized single env)
    # SB3 expects observation_space and action_space to be set properly in env
    sb3_env = JoinOrderEnv(SAMPLE_JSON, max_tables=max_tables)

    # Custom policy requires knowing node_feat_dim and max_tables
    obs_dim = sb3_env.observation_space.shape[0]
    # infer node_feat_dim: we used F=4 in env
    node_feat_dim = 4

    policy_kwargs = dict(
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=dict(features_dim=obs_dim),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    )

    model = PPO(MaskableActorCriticPolicy, sb3_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=64, n_steps=128)

    # Attach attributes needed by policy to mask obs
    # Stable-baselines3 will create policy instance after model creation; we patch after
    model.policy.max_tables = max_tables
    model.policy.node_feat_dim = node_feat_dim

    # Behavior cloning warm start not integrated into SB3 policy here for brevity.
    # You can load bc_model weights into model.policy parameters if shapes are compatible.

    # Train for a small number of timesteps (demo)
    print("Starting PPO training (demo, small timesteps)...")
    model.learn(total_timesteps=2048)

    # Save model
    model.save("join_order_ppo_demo")
    print("Training complete. Model saved to join_order_ppo_demo.zip")


if __name__ == "__main__":
    main()
