import torch
from torch import nn

# 1st model: (100, 64, 42, 18)->(4),    all hidden ReLU,      5  epoch early stopping
class ReviewNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ReviewNet, self).__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 42),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(42, 18),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.stars_output = nn.Linear(18, 6)  # 6 classes for 0 to 5 stars
        self.other_outputs = nn.Linear(18, 3)  # for useful, funny, cool

    def forward(self, x):
        common_features = self.common_layers(x)

        stars_logits = self.stars_output(common_features)
        stars_probs = nn.functional.softmax(stars_logits, dim=1)
        stars_output = torch.sum(stars_probs * torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device), dim=1)

        other_outputs = nn.functional.relu(self.other_outputs(common_features))  # Ensure non-negative outputs

        return torch.cat((stars_output.unsqueeze(1), other_outputs), dim=1)


# 2nd model: (100)->(128, 64, 32)->(4), all hidden LeakyReLU, 5 epoch early stopping
class ReviewNetLarge(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ReviewNetLarge, self).__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.stars_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 6)
        )
        self.other_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        common_features = self.common_layers(x)

        stars_logits = self.stars_layers(common_features)
        stars_probs = nn.functional.softmax(stars_logits, dim=1)
        stars_output = torch.sum(stars_probs * torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device), dim=1)

        other_outputs = nn.functional.relu(self.other_layers(common_features))

        return torch.cat((stars_output.unsqueeze(1), other_outputs), dim=1)


# 3rd model: (100)->(128, 64, 32)->(4), all hidden LeakyReLU, Batch Normalized, 5 epoch early stopping
class ReviewNetLargeNorm(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ReviewNetLargeNorm, self).__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(100, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.stars_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 6)
        )
        self.other_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        common_features = self.common_layers(x)

        stars_logits = self.stars_layers(common_features)
        stars_probs = nn.functional.softmax(stars_logits, dim=1)
        stars_output = torch.sum(stars_probs * torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device), dim=1)

        other_outputs = nn.functional.relu(self.other_layers(common_features))

        return torch.cat((stars_output.unsqueeze(1), other_outputs), dim=1)


# 4th model: (100)->(64)->(4), all hidden LeakyReLU, 5 epoch early stopping
class ReviewNetSmall(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ReviewNetSmall, self).__init__()
        self.common_layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.stars_output = nn.Linear(64, 6)  # 6 classes for 0 to 5 stars
        self.other_outputs = nn.Linear(64, 3)  # for useful, funny, cool

    def forward(self, x):
        common_features = self.common_layers(x)

        stars_logits = self.stars_output(common_features)
        stars_probs = nn.functional.softmax(stars_logits, dim=1)
        stars_output = torch.sum(stars_probs * torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device), dim=1)

        other_outputs = nn.functional.relu(self.other_outputs(common_features))  # Ensure non-negative outputs

        return torch.cat((stars_output.unsqueeze(1), other_outputs), dim=1)