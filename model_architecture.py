import torch
from torch import nn


class MatchPredictor(nn.Module):

    def __init__(self, series_input_size, series_hidden_size, series_num_layers, static_data_size, device):
        super().__init__()
        self.lstm_params = {
            'input_size' : series_input_size,
            'num_layers' : series_num_layers,
            'batch_first' : True,
            'bidirectional' : False,
            'dropout' : 0.5,
            }
        
        self.device = device
        self.hidden_size = series_hidden_size + series_hidden_size * self.lstm_params['bidirectional']
        self.combined_size = 2 * self.hidden_size + static_data_size
        self.lstm_params['hidden_size'] = self.hidden_size
        
        self.home_lstm = nn.LSTM(**self.lstm_params)
        self.away_lstm = nn.LSTM(**self.lstm_params)
        self.fc = nn.Sequential(
            nn.Linear(self.combined_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 3)
            )
        
        #self.fc = nn.Linear(self.combined_size, 3)
        
    def forward(self, static, packed_home, packed_away):
        _, (h_home, _) = self.home_lstm(packed_home)
        _, (h_away, _) = self.away_lstm(packed_away)
        h_home, h_away = h_home.to(self.device), h_away.to(self.device)
        x = torch.cat((h_home[-1], h_away[-1], static), dim=1)
        return self.fc(x)