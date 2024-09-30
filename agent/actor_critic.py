import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
        )
        self.actor = nn.Linear(int(hidden_dim/4), n_actions)
        self.critic = nn.Linear(int(hidden_dim/4), 1)

    def forward(self, x):
        shared_output = self.shared(x)
        action_logits = self.actor(shared_output)
        state_values = self.critic(shared_output)
        return action_logits, state_values