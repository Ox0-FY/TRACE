import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta

class TaskEncoder(nn.Module):
    def __init__(self, input_feature_dim=3, hidden_size=64, output_dim=32):
        super().__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, support_x, support_y):
        x_emb = self.x_encoder(support_x)
        y_emb = self.y_encoder(support_y)
        sample_emb = torch.cat([x_emb, y_emb], dim=1)
        return self.fusion_net(sample_emb.mean(0))

class FiLMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.film1, self.relu1 = FiLMLayer(), nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.film2, self.relu2 = FiLMLayer(), nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, gammas, betas):
        x = self.relu1(self.film1(self.fc1(x), gammas[0], betas[0]))
        x = self.relu2(self.film2(self.fc2(x), gammas[1], betas[1]))
        return self.fc3(x)

class FiLMGenerator(nn.Module):
    def __init__(self, task_embedding_dim, num_film_layers, hidden_size):
        super().__init__()
        self.num_film_layers, self.hidden_size = num_film_layers, hidden_size
        self.generator = nn.Sequential(
            nn.Linear(task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_film_layers * hidden_size * 2)
        )

    def forward(self, task_embedding):
        params = self.generator(task_embedding)
        gammas, betas = [], []
        chunks = torch.chunk(params, self.num_film_layers * 2, dim=0)
        for i in range(self.num_film_layers):
            gammas.append(chunks[2 * i].view(1, -1))
            betas.append(chunks[2 * i + 1].view(1, -1))
        return gammas, betas
