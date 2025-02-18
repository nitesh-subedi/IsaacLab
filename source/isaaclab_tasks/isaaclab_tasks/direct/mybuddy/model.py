import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# class MultiInputSingleOutputLSTM(nn.Module):
#     def __init__(self, hidden_size=128, num_layers=4):
#         super(MultiInputSingleOutputLSTM, self).__init__()
        
#         # Pretrained CNN backbone (e.g., ResNet)
#         resnet = models.resnet18(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.feature_dim = resnet.fc.in_features  # Get feature dimension
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False

#         # LSTM for sequence modeling
#         self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, 
#                             num_layers=num_layers, batch_first=True)

#         # Fully connected layer for bbox prediction
#         self.fc = nn.Linear(hidden_size, 4)  # 4 for bbox coordinates

#     def forward(self, x):
#         batch_size, seq_len, _, _, _ = x.size()

#         # Extract features for each image in the sequence
#         x = x.view(batch_size * seq_len, 3, x.size(-2), x.size(-1))  # Flatten sequences
#         features = self.feature_extractor(x)  # Feature maps
#         features = self.pool(features).view(batch_size, seq_len, -1)  # (B, T, feature_dim)

#         # Pass through LSTM
#         lstm_out, _ = self.lstm(features)  # (B, T, hidden_size)

#         # Get output for the last time step
#         final_out = lstm_out[:, -1, :]  # (B, hidden_size)

#         # Predict bounding box
#         bbox = self.fc(final_out)  # (B, 4)
#         return bbox
class MultiInputSingleOutputLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=4):  # Change hidden_size to 256
        super(MultiInputSingleOutputLSTM, self).__init__()
        
        # Pretrained CNN backbone (e.g., ResNet)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = resnet.fc.in_features  # Get feature dimension (512 for ResNet18)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)

        # Fully connected layer for bbox prediction
        self.fc = nn.Linear(hidden_size, 4)  # 4 for bbox coordinates

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()

        # Extract features for each image in the sequence
        # x = x.view(batch_size * seq_len, 3, x.size(-2), x.size(-1))  # Flatten sequences
        x = x.reshape(batch_size * seq_len, 3, x.size(-2), x.size(-1))  # Flatten sequences
        features = self.feature_extractor(x)  # Feature maps
        features = self.pool(features).view(batch_size, seq_len, -1)  # (B, T, feature_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_size)

        # Get output for the last time step
        final_out = lstm_out[:, -1, :]  # (B, hidden_size)

        # Predict bounding box
        bbox = self.fc(final_out)  # (B, 4)
        return bbox


class HeatmapLSTM(nn.Module):
    def __init__(self, output_size=(256, 256), hidden_size=256, num_layers=5):
        super(HeatmapLSTM, self).__init__()
        
        self.output_size = output_size

        # Pretrained CNN backbone (e.g., ResNet)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = resnet.fc.in_features  # Get feature dimension
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer to output heatmap
        self.fc = nn.Linear(hidden_size, output_size[0] * output_size[1])
        
    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()

        # Extract features for each image in the sequence
        x = x.view(batch_size * seq_len, 3, x.size(-2), x.size(-1))  # Flatten sequences
        x = self.feature_extractor(x)  # Feature maps
        x = self.pool(x).view(batch_size, seq_len, -1)  # (B, T, feature_dim)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state to predict the heatmap
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the LSTM
        heatmap = self.fc(lstm_out)
        
        # Reshape the output to match the image size (height, width)
        heatmap = heatmap.view(batch_size, self.output_size[0], self.output_size[1])
        
        return heatmap


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_channels, h, w):
        super(Generator, self).__init__()
        self.h, self.w = h, w  # Store H and W for later use
        self.conv_lstm = nn.LSTM(input_size=input_channels * h * w, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128 * 128 * output_channels),
            nn.ReLU(),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output range [-1, 1] (to match image normalization)
        )

    def forward(self, x):
        # x: (batch_size, seq_length, input_channels, H, W)
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size, seq_length, -1)  # Flatten H and W for LSTM input
        lstm_out, _ = self.conv_lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the last timestep's output
        fc_out = self.fc(lstm_out).view(batch_size, c, 128, 128)
        out = self.conv_out(fc_out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(64, 1),  # Final layer outputs a single value
            nn.Sigmoid()  # Ensures output is in [0, 1]
        )

    def forward(self, x):
        return self.model(x)


class HeatmapTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, seq_length=10, in_channels=3, out_channels=1, d_model=128, nhead=4, num_layers=6):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # Embedding layers
        self.patch_embedding = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))  # Adjusted positional encoding

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output decoder
        # self.output_decoder = nn.Sequential(
        self.linearizer = nn.Linear(in_features=d_model, out_features=patch_size * patch_size * out_channels)
        self.unflattenizer = nn.Unflatten(dim=2, unflattened_size=(out_channels, patch_size, patch_size))
        # )

    def forward(self, input_sequence):
        batch_size, seq_length, _, H, W = input_sequence.size()
        assert seq_length == self.seq_length, f"Expected sequence length {self.seq_length}, got {seq_length}"

        # Flatten each image into patches and embed
        patches = self.patch_embedding(input_sequence.view(-1, 3, H, W))  # (batch_size * seq_length, d_model, H', W')
        patches = patches.flatten(2).permute(0, 2, 1)  # (batch_size * seq_length, num_patches, d_model)

        # Expand positional encoding to match the batch size and add
        positional_encoding = self.positional_encoding.expand(batch_size * seq_length, -1, -1)  # Match (batch_size * seq_length, num_patches, d_model)
        patches = patches + positional_encoding

        # Reshape for transformer input
        transformer_input = patches.reshape(batch_size, seq_length * self.num_patches, self.d_model)

        # Transformer encoding
        encoded = self.encoder(transformer_input)  # (batch_size, seq_length * num_patches, d_model)

        # Decode patches back to image space
        # decoded_patches = self.output_decoder(encoded)  # (batch_size, seq_length * num_patches, out_channels, patch_size, patch_size)
        decoded_patches = self.linearizer(encoded)
        decoded_patches = self.unflattenizer(decoded_patches)

        # Reshape into output images
        decoded_patches = decoded_patches.view(batch_size, seq_length, self.num_patches, 1, self.patch_size, self.patch_size)
        output_images = decoded_patches.permute(0, 1, 3, 4, 5, 2).reshape(batch_size, seq_length, 1, H, W)

        return output_images[:, -1, :, :, :].squeeze(1)  # Return only the last output image

