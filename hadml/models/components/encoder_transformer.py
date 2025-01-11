import torch, os, pickle


class Generator(torch.nn.Module):
    """ Generator implemented as a encoder-only transformer model """

    def __init__(
        self,
        noise_dim=4,            # Arbitrary number (noise dimensionality)
        cluster_kins_dim=8,     # Cluster four-momentum, two quark types, phi, theta
        n_quarks=2,             # Number of quark types in cluster_kins_dim
        quark_types=16,         # Quark PIDs: (+/-) 1-8 -> transformation -> 0-16
        hadron_kins_dim=4,      # Hadron four-momentum
        embedding_dim=32,       # Arbitrary number (but the same for the discriminator)
        quark_embedding_dim=4,  # Arbitrary number (quark embedding dimensionality)
        n_heads=8,              # Encoder architecture hyperparameter
        pid_map_filepath=None   # For getting information about the number of hadron most common IDs
    ):
        super().__init__()
        self.hadron_kins_dim = hadron_kins_dim
        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            n_hadron_types = len(pickle.load(f)) + 1
        self.quark_type_embedding_layer = torch.nn.Embedding(quark_types, quark_embedding_dim)
        self.input_embedding_layer = torch.nn.Linear(
            noise_dim + cluster_kins_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, hadron_kins_dim + n_hadron_types)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, noise, cluster_kins):
        quark_types = cluster_kins[:, :, 4:6].to(torch.int32)
        quark_types = torch.where(quark_types < 0, torch.abs(quark_types), quark_types + 8)
        embedded_quark_types = self.quark_type_embedding_layer(quark_types)
        embedded_quark_types = embedded_quark_types.reshape(*embedded_quark_types.size()[:2], -1)
        cluster_kins = torch.concatenate((cluster_kins[:, :, :4], embedded_quark_types, 
                                          cluster_kins[:, :, 6:]), dim=2)
        clusters_and_noise = torch.concatenate((cluster_kins, noise), dim=2)
        embedded_input = self.input_embedding_layer(clusters_and_noise)
        embedded_output = self.transformer_encoder(embedded_input)
        hadrons = self.output_embedding_layer(embedded_output)
        return hadrons
    

class Discriminator(torch.nn.Module):
    """ Discriminator implemented as a encoder-only transformer model """

    def __init__(
        self,
        hadron_kins_dim=4,      # Hadron four-momentum
        embedding_dim=32,       # Arbitrary number (but the same for the discriminator)
        n_heads=8,              # Encoder architecture hyperparameter
        pid_map_filepath=None   # For getting information about the number of hadron most common IDs
    ):
        super().__init__()
        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            n_hadron_types = len(pickle.load(f)) + 1
        self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + n_hadron_types, embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, hadrons):
        embedded_input = self.input_embedding_layer(hadrons)
        embedded_output = self.transformer_encoder(embedded_input)
        real_or_fake_response = self.output_embedding_layer(embedded_output)
        return real_or_fake_response.mean(dim=1)