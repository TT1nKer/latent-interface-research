"""Learn a demonstration-position metric from query action labels alone."""
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


ACTIONS = ('STOP', 'move', 'pickMarker', 'putMarker', 'turnLeft', 'turnRight')
ACTION_ID = {name: i for i, name in enumerate(ACTIONS)}
BOS = len(ACTIONS)
FEATURES = 4 + len(ACTIONS) + 1
SEEDS = (17, 29, 43)


def encode_trace(trace):
    inputs = torch.zeros((len(trace), FEATURES))
    labels = torch.empty(len(trace), dtype=torch.long)
    previous = BOS
    for i, (sensors, action) in enumerate(trace):
        inputs[i, :4] = torch.tensor(sensors, dtype=torch.float32)
        inputs[i, 4 + previous] = 1.0
        labels[i] = previous = ACTION_ID[action]
    if labels[-1].item() != ACTION_ID['STOP']:
        raise ValueError('Demonstration must contain a terminal STOP')
    return inputs, labels


def pack_episodes(episodes):
    """Pad within this batch only. The first trajectory in each episode is the query."""
    batch, count = len(episodes), len(episodes[0])
    length = max(len(y) for ep in episodes for _, y in ep)
    x = torch.zeros(batch, count, length, FEATURES)
    y = torch.full((batch, count, length), -1, dtype=torch.long)
    for i, ep in enumerate(episodes):
        for j, (trace_x, trace_y) in enumerate(ep):
            n = len(trace_y)
            x[i, j, :n] = trace_x
            y[i, j, :n] = trace_y
    return x, y


class MetricPolicy(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ('retrieval', 'centroid'):
            raise ValueError(variant)
        self.variant = variant
        self.gru = nn.GRU(FEATURES, 32, batch_first=True)
        self.project = nn.Linear(36, 24)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.0)))

    def encode(self, x, state=None):
        hidden, state = self.gru(x, state)
        vector = self.project(torch.cat((hidden, x[..., :4]), dim=-1))
        return F.normalize(vector, dim=-1), state

    def memory(self, keys, labels):
        mask = labels >= 0
        weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        keys = keys.flatten(1, 2)
        votes = F.one_hot(labels.clamp_min(0), num_classes=len(ACTIONS)).float().flatten(1, 2)
        weights = weights.flatten(1, 2)
        if self.variant == 'centroid':
            weighted_votes = votes * weights.unsqueeze(-1)
            frequency = weighted_votes.sum(dim=1)
            centers = weighted_votes.transpose(1, 2).bmm(keys)
            return F.normalize(centers, dim=-1), frequency.clamp_min(1e-12).log().masked_fill(frequency == 0, -torch.inf), None
        log_prior = weights.clamp_min(1e-12).log().masked_fill(weights == 0, -torch.inf)
        return keys, log_prior, votes

    def predict(self, query, memory):
        keys, log_prior, votes = memory
        scale = self.log_temperature.exp().clamp(min=1.0, max=60.0)
        weights = (scale * query.bmm(keys.transpose(1, 2)) + log_prior.unsqueeze(1)).softmax(-1)
        return weights if votes is None else weights.bmm(votes)

    def forward(self, x, y):
        b, n, t, _ = x.shape
        keys, _ = self.encode(x.flatten(0, 1))
        keys = keys.reshape(b, n, t, -1)
        return self.predict(keys[:, 0], self.memory(keys[:, 1:], y[:, 1:]))


class NeuralPolicy:
    """The executing query can access only its own observed prefix and five demonstrations."""
    @torch.inference_mode()
    def __init__(self, model, support, scrambled=False):
        self.model = model
        self.state = None
        self.pending_state = None
        self.previous = BOS
        episodes = [[encode_trace(trace) for trace in support]]
        x, labels = pack_episodes(episodes)
        if scrambled:
            for j, trace in enumerate(support):
                n = len(trace)
                labels[0, j, :n] = labels[0, j, :n].roll(max(1, n // 2))
        keys, _ = model.encode(x[0])
        self.memory = model.memory(keys.unsqueeze(0), labels)

    @torch.inference_mode()
    def act(self, obs):
        x = torch.zeros(1, 1, FEATURES)
        x[0, 0, :4] = torch.tensor(obs, dtype=torch.float32)
        x[0, 0, 4 + self.previous] = 1.0
        q, self.pending_state = self.model.encode(x, self.state)
        policy = self.model.predict(q, self.memory)
        return ACTIONS[int(policy.argmax(-1).item())]

    def advance(self, obs, action):
        if self.pending_state is None:
            raise ValueError('act must precede advance')
        self.state, self.pending_state = self.pending_state, None
        self.previous = ACTION_ID[action]


def read_checkpoint(path):
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def load_trained(path):
    checkpoint = read_checkpoint(path)
    models = {}
    for variant in ('retrieval', 'centroid'):
        for seed in SEEDS:
            model = MetricPolicy(variant)
            model.load_state_dict(checkpoint['models'][f'{variant}_{seed}'])
            models[variant, seed] = model.eval()
    torch.manual_seed(17)
    models['initial', 17] = MetricPolicy('retrieval').eval()
    return models
