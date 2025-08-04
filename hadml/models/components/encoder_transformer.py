import torch, os, pickle
from particle import Particle
from hadml.rambo.rambo_on_diet import RamboOnDiet
from hadml.rambo.additional_helpers import get_invariant_mass
import time


class Generator(torch.nn.Module):
    """ Generator implemented as a encoder-only transformer model """

    def __init__(
        self,
        noise_dim=8,            # Arbitrary number (noise dimensionality)
        cluster_data_dim=8,     # Cluster four-momentum, two quark types, phi, theta
        n_quarks=2,             # Number of quarks in cluster_data_dim
        quark_types=16,         # Quark types: 0-16
        hadron_kins_dim=4,      # Hadron four-momentum
        space_phase_dim=3,      # Phase space random variables dimensionality
        num_layers=2,           # Number of sub-encoder-layers in the encoder
        embedding_dim=128,      # Arbitrary number (but the same for the discriminator)
        dim_feedforward=128,    # Dimension of the feedforward network model used in the encoder
        quark_embedding_dim=2,  # Arbitrary number (quark embedding dimensionality)
        n_heads=4,              # Encoder architecture hyperparameter
        pid_map_filepath=None   # For getting information about the number of hadron most common IDs
    ):
        super().__init__()
        self.hadron_kins_dim = hadron_kins_dim
        self.space_phase_dim = space_phase_dim
        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            raw_pid_map = pickle.load(f)             # {pid: index}
        self.pid_map = {idx: float(pid) for pid, idx in raw_pid_map.items()}
        n_hadron_types = len(raw_pid_map) + 1
 
        self.quark_type_embedding_layer = torch.nn.Embedding(quark_types, quark_embedding_dim)
        self.input_embedding_layer = torch.nn.Linear(
            noise_dim + cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 3 + n_hadron_types)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rambo = RamboOnDiet()

    def forward(self, noise, cluster_kins):
        embedded_quark_types = self.quark_type_embedding_layer(cluster_kins[:, :, 4:6].to(torch.int32))
        embedded_quark_types = embedded_quark_types.reshape(*embedded_quark_types.size()[:2], -1)
        cluster_kins = torch.concatenate((cluster_kins[:, :, :4], embedded_quark_types, 
                                          cluster_kins[:, :, 6:]), dim=2)
        clusters_and_noise = torch.concatenate((cluster_kins, noise), dim=2)
        embedded_input = self.input_embedding_layer(clusters_and_noise)
        
        # Preparing the causal mask
        src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            embedded_input.size(1)).to(embedded_input.device)

        # Passing through the transformer encoder
        embedded_output = self.transformer_encoder(embedded_input, mask=src_mask, is_causal=True)
        
        # The first 3 dimensions are phase space random variables, the rest are hadron IDs  
        output = self.output_embedding_layer(embedded_output)
        
        # Clamp values to avoid numerical instability
        output = torch.clamp(output, min=-1e6, max=1e6)

        # Applying Gumbel-Softmax to the hadron IDs
        output[:, :, self.space_phase_dim:] = torch.nn.functional.gumbel_softmax(
            output[:, :, self.space_phase_dim:],
            self.current_gumbel_temp,
            hard=self.gumbel_softmax_hard
        )

        # The padding token is the one with the first dimension equal to 1.0
        # We want to zero out the padding tokens in the output
        pad_mask = output[:, :, self.space_phase_dim] == 1.0
        pure_padding = torch.zeros_like(output[pad_mask])
        pure_padding[:, self.space_phase_dim] = 1.0
        output[pad_mask] = pure_padding

        # Moving all padding tokens to end of each sequence, keep relative order of the rest
        pad_indicator = pad_mask.to(torch.int64)    # 1 for pad, 0 otherwise
        order = torch.argsort(pad_indicator, dim=1) # non‐pads (0) come first
        output = torch.gather(output, dim=1, index=order.unsqueeze(2).expand(-1, -1, output.size(2)))

        # Applying Rambo on diet to get the hadron four-momenta
        cluster_invariant_masses = get_invariant_mass(cluster_kins[:, 0, :4].reshape(-1, 4))
        new_output = []

        # TODO: is there a way to vectorize this?
        for seq, E_CM in zip(output, cluster_invariant_masses):
            # Separating phase space variables and hadron IDs
            pad_mask = seq[:, self.space_phase_dim] == 1.0
            phase_space_vars = seq[~pad_mask, :self.space_phase_dim]
            pad_tokens = seq[pad_mask]
            hadron_ids = seq[~pad_mask, self.space_phase_dim + 1:]
            hadron_ids = torch.argmax(hadron_ids, dim=1)
            
            # Getting masses based on hadron PIDs
            pids = torch.tensor([self.pid_map[hadron_id.item()] for hadron_id in hadron_ids])
            masses = torch.tensor([Particle.from_pdgid(pid).mass / 1000 for pid in pids], device=seq.device)
            n_particles = len(masses)

            if masses.sum() < E_CM and n_particles >= 2:
                # Normalise phase_space_vars to [0, 1] with improved handling
                eps = 1e-8
                min_val = phase_space_vars.min()
                max_val = phase_space_vars.max()
                denom = max_val - min_val
                if denom.abs() < eps:
                    phase_space_vars_norm = phase_space_vars
                else:
                    phase_space_vars_norm = (phase_space_vars - min_val) / (denom + eps)

                # Flatten and pad/truncate to correct length for RamboOnDiet
                required_len = 3 * n_particles - 4
                flat = phase_space_vars_norm.flatten()
                if flat.numel() < required_len:
                    pad = torch.zeros(required_len - flat.numel(), device=flat.device, dtype=flat.dtype)
                    phase_space_vars_norm = torch.cat([flat, pad], dim=0).reshape(1, -1)
                else:
                    phase_space_vars_norm = flat[:required_len].reshape(1, -1)
                
                # Applying Rambo on diet to get the hadron four-momenta
                device = phase_space_vars.device
                try:
                    # start_time = time.time()
                    (p,), _ = self.rambo.map(inputs=[phase_space_vars_norm.to(device), torch.tensor([E_CM]).to(device)],
                                        nparticles=n_particles, masses=masses)
                    # elapsed = time.time() - start_time
                    # print(f"Loop step took {elapsed:.6f} seconds")
                    # Getting rid of the first two zeroed particles (initial state)
                    p = p[:, 2:] 
                except Exception as e:
                    # If Rambo fails, we just return zero momenta
                    print(f"RamboOnDiet failed: {e}. Returning zero momenta.")
                    p = torch.zeros((1, n_particles, 4), device=seq.device)
            else:
                # If the invariant mass is too large, we just return zero momenta
                p = torch.zeros((1, n_particles, 4), device=seq.device)
            
            # Adding padding tokens back to the output
            new_output.append(
                torch.cat([
                    torch.cat([p[0], seq[~pad_mask, self.space_phase_dim:]], dim=1),
                    torch.cat([
                        torch.zeros(pad_tokens.size(0), 1, device=pad_tokens.device, dtype=pad_tokens.dtype),
                        pad_tokens], dim=1)
                ], dim=0)
            )
        
        output = torch.stack(new_output)
        return output


class Discriminator(torch.nn.Module):
    """ Discriminator implemented as a encoder-only transformer model """

    def __init__(
        self,
        hadron_kins_dim=4,      # Hadron four-momentum
        num_layers=2,           # Number of sub-encoder-layers in the encoder
        embedding_dim=128,      # Arbitrary number (but the same for the discriminator)
        dim_feedforward=128,    # Dimension of the feedforward network model used in the encoder
        n_heads=4,              # Encoder architecture hyperparameter
        pid_map_filepath=None   # For getting information about the number of hadron most common IDs
    ):
        super().__init__()
        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            n_hadron_types = len(pickle.load(f)) + 1
        self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + n_hadron_types, embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hadrons):
        embedded_input = self.input_embedding_layer(hadrons)
        
        # Preparing the causal mask
        src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            embedded_input.size(1)).to(embedded_input.device)
        
        embedded_output = self.transformer_encoder(embedded_input, mask=src_mask, is_causal=True)
        real_or_fake_response = self.output_embedding_layer(embedded_output)
        return real_or_fake_response.mean(dim=1)