import torch
from torch import nn


class MatchPredictor(nn.Module):

    def __init__(self, series_input_size, series_hidden_size, series_num_layers, static_data_size):
        super().__init__()
        self.hidden_size = 2 * series_hidden_size
        self.combined_size = 4 * self.hidden_size + static_data_size

        self.lstm_params = {
            'input_size' : series_input_size,
            'hidden_size' : self.hidden_size,
            'num_layers' : series_num_layers,
            'batch_first' : True,
            'bidirectional' : True,
            'dropout' : 0.2
            }
        
        self.home_lstm = nn.LSTM(**self.lstm_params)
        self.away_lstm = nn.LSTM(**self.lstm_params)
        self.fc_block = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
            )
        
    def forward(self, static, packed_home, packed_away):
        _, (h_home, _) = self.home_lstm(packed_home)
        _, (h_away, _) = self.away_lstm(packed_away)

        home_last = torch.cat((h_home[:, -2, :], h_home[:, -1, :]), dim=1)
        away_last = torch.cat((h_away[:, -2, :], h_away[:, -1, :]), dim=1)
        x = torch.cat((home_last, away_last, static), dim=1)

        return self.fc_block(x)