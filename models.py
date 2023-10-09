import torch
import torch.nn.functional as f
import audobject
import torch.nn as nn

class TransformerClassifier(torch.nn.Module):
    def __init__(self, 
        d_model: int, 
        output_dim: int,
        nhead: int = 8,
        sigmoid: bool = False,
        dropout: int = 0.5,
        ff_hidden_size: int = 512,
        num_encoder_layers: int = 3,
        embeddings_are_2D: bool = False
        ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.embeddings_are_2D = embeddings_are_2D

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        self.sigmoid = sigmoid
        self.dropout = dropout
        self.ff_hidden_size = ff_hidden_size
        self.classifier = nn.Sequential(
            torch.nn.Linear(self.d_model, self.ff_hidden_size),
            torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
            # torch.nn.Linear(self.ff_hidden_size, self.ff_hidden_size),
            # torch.nn.Dropout(self.dropout),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.ff_hidden_size, self.output_dim, bias=False)
        )

        self.sigmoid_layer = nn.Sigmoid()
        

    def forward(self, x):
        x = x.squeeze(1)
        # print("\nx1: ", x.shape)
        x = self.transformer_encoder(x)
        # print("x2: ", x.shape)
        if self.embeddings_are_2D:
            x = x.mean(dim=1)
        # print("x3: ", x.shape)
        x = self.classifier(x)
        # print("x4: ", x.shape)
        

        if self.sigmoid:
            # x = torch.nn.Sigmoid()(x)
            x = self.sigmoid_layer(x)
        return x
