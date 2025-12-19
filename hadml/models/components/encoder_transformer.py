import math
import torch, os, pickle
from particle import Particle
from hadml.rambo.rambo_on_diet import RamboOnDiet, NearlyRambo
from hadml.rambo.additional_helpers import get_invariant_mass, lorentz_boost, vectorized_boost
from hadml.rambo.solve_xi import solve_xi
import sys
import time
import numpy as np

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.n = 10_000

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        even_idx = torch.arange(0, d_model, 2).float()
        odd_idx  = torch.arange(1, d_model, 2).float()

        pe[:, 0::2] = torch.sin(position / torch.pow(self.n, (even_idx / d_model)))
        pe[:, 1::2] = torch.cos(position / torch.pow(self.n, (odd_idx  / d_model)))

        # pe: (1, max_len, d_model)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


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

        # create mass_tensor
        self.mass_tensor = torch.tensor([Particle.from_pdgid(pid).mass / 1000 for pid in self.pid_map.values()])

        n_hadron_types = len(raw_pid_map) + 1
 
        self.quark_type_embedding_layer = torch.nn.Embedding(quark_types, quark_embedding_dim)
        self.input_embedding_layer = torch.nn.Linear(
            noise_dim + cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 3 + n_hadron_types)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
      #  self.rambo = RamboOnDiet()
        #self.rambo = NearlyRambo()

    def forward(self, noise, cluster_kins):
      #  print(cluster_kins[5])
        embedded_quark_types = self.quark_type_embedding_layer(cluster_kins[:, :, 4:6].to(torch.int32))
        embedded_quark_types = embedded_quark_types.reshape(*embedded_quark_types.size()[:2], -1)
        cluster_kins = torch.concatenate((cluster_kins[:, :, :4], embedded_quark_types, 
                                          cluster_kins[:, :, 6:]), dim=2)

        clusters_and_noise = torch.concatenate((cluster_kins, noise), dim=2)
        embedded_input = self.input_embedding_layer(clusters_and_noise)
        embedded_input = self.positional_encoding(embedded_input)
        
        # Preparing the causal mask
       # src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        #    embedded_input.size(1)).to(embedded_input.device)

        # Passing through the transformer encoder
       # embedded_output = self.transformer_encoder(embedded_input, mask=src_mask, is_causal=True)
        embedded_output = self.transformer_encoder(embedded_input)
        # The first 3 dimensions are phase space random variables, the rest are hadron IDs  
        output = self.output_embedding_layer(embedded_output)
        # sys.exit()
        
        # Clamp values to avoid numerical instability
        output = torch.clamp(output, min=-1e6, max=1e6)
        # Applying Gumbel-Softmax to the hadron IDs
        output[:, :, self.space_phase_dim:] = torch.nn.functional.gumbel_softmax(
            output[:, :, self.space_phase_dim:],
            self.current_gumbel_temp,
            hard=self.gumbel_softmax_hard
        )

        #print('epoch: ', self.current_gumbel_temp)

        # The padding token is the one with the first dimension equal to 1.0
        # We want to zero out the padding tokens in the output
      #  print(output[:, :, self.space_phase_dim][0])
      #  sys.exit()
        pad_mask = output[:, :, self.space_phase_dim] == 1.0
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand_as(output)


        # Create the pure padding template
        pure_padding_template = torch.zeros_like(output)
        pure_padding_template[:, :, self.space_phase_dim] = 1.0

        # Replace using torch.where (preserves gradients!)
        output = torch.where(pad_mask_expanded, pure_padding_template, output)
       # print('after zeroing padding: ', output[:, :, self.space_phase_dim][0])

        # Moving all padding tokens to end of each sequence, keep relative order of the rest
        pad_indicator = pad_mask.to(torch.int64)    # 1 for pad, 0 otherwise
        order = torch.argsort(pad_indicator, dim=1) # non‐pads (0) come first
        output = torch.gather(output, dim=1, index=order.unsqueeze(2).expand(-1, -1, output.size(2)))

        # new pad_mask after reordering    
        pad_mask = output[:, :, self.space_phase_dim] == 1.0

        mass_table = self.mass_tensor.to(output.device)
        hadron_logits = output[:, :, self.space_phase_dim+1:]  # [B, N, 3]
        masses = hadron_logits @ mass_table                    # [B, N]

        all_paddings = pad_mask.all(dim=1)
        # Convert boolean to float for differentiable operations
        pad_mask_float = (~pad_mask).float()  # Invert: 1 for real, 0 for padding

        # Multiply instead of assign (preserves gradients)
        masses = masses * pad_mask_float  # padding -> 0, real -> keeps value
        momenta_3 = output[:, :, :3]


        # compute 3 momenta magnitudes
        momenta_mag = torch.norm(momenta_3, dim=2)  # (B, S)
        eps = 1e-12
        energies = torch.sqrt(torch.clamp(momenta_mag**2 + masses**2, min=eps)).unsqueeze(2)
        momenta_4 = torch.cat((energies, momenta_3), dim=2)  # (B, S, 4)
        total_4_momenta = momenta_4.sum(dim=1)  # (B, 4)

        # rest_frame boost
        momenta_4_rest_frame = vectorized_boost(momenta_4, total_4_momenta, inverse=False)
        momenta_4_rest_frame = momenta_4_rest_frame * (pad_mask_float.unsqueeze(2))  # zero out padding particles
        momenta_3_rest_frame = momenta_4_rest_frame[:, :, 1:]
        
        momenta_mag_rest_frame = torch.norm(momenta_3_rest_frame, dim=2)  # (B, S)
        cluster_invariant_masses = get_invariant_mass(cluster_kins[:, 0, :4].reshape(-1, 4))

        xi, valid, violation = solve_xi(momenta_mag_rest_frame, masses, cluster_invariant_masses, ~pad_mask, n_iter=12)
        xi = xi.unsqueeze(1).unsqueeze(2)

        # Scale momenta
        rescaled_momenta_3_rest_frame = momenta_3_rest_frame * xi          # (B,N,3)



        # Compute energies correctly: E_i = sqrt(px^2+py^2+pz^2 + m_i^2)
        p2 = (rescaled_momenta_3_rest_frame**2).sum(dim=2)                  # (B,N)
        eps = 1e-12
        E = torch.sqrt(torch.clamp(p2 + masses**2, min=eps))                                  # (B,N)

        # Build 4-vectors
        rescaled_4_momenta_rest_frame = torch.cat(
            (E.unsqueeze(2), rescaled_momenta_3_rest_frame), dim=2
        )
        # if valid == False, set to zero
        rescaled_4_momenta_rest_frame[~valid] = 0.0
        cluster_4_momenta = cluster_kins[:,0,:4]

        boosted_4_momenta = vectorized_boost(rescaled_4_momenta_rest_frame, 
                                             cluster_4_momenta, inverse=True)
    
        
      #  print('pad_mask shape', pad_mask.shape)
        boosted_4_momenta = boosted_4_momenta * pad_mask_float.unsqueeze(2)  # zero out padding particles

        # find the nans in boosted_4_momenta
        # shape N, 32, 8
        new_output = torch.cat(
            (boosted_4_momenta, output[:, :, self.space_phase_dim:]), dim=2
        )
        all_paddings = all_paddings.view(-1,1,1)


        new_output = torch.where(all_paddings, 
                         torch.zeros_like(new_output), 
                         new_output)
        #new_output[all_paddings] = torch.nan_to_num(new_output[all_paddings], nan=0.0)

        if torch.isnan(new_output).any():
            print('NaNs detected in output')
            # find which batch entries have nans
            nan_batches = torch.isnan(new_output).any(dim=(1,2)).nonzero(as_tuple=True)[0]

            # print the nan entries
            for batch_idx in nan_batches:
                print(f'NaNs in batch entry {batch_idx}:')
                print(new_output[batch_idx])


            sys.exit()





        padding_token = output[:,:, self.space_phase_dim]
        total_paddings = torch.sum(padding_token, dim=1)
        seq_length = padding_token.size(1)
        total_particles = seq_length - total_paddings
        # constrain relu on total_particles to be at least 2 particles
        non_pad_loss = torch.nn.ReLU()(2.0 - total_particles)

        is_odd = (total_particles % 2).float()  # 0 for even, 1 for odd
        #odd_penalty = torch.mean(is_odd)
        
     

        return new_output, violation**2, non_pad_loss**2, is_odd
        # print('boosted 4-momenta shape:', boosted_4_momenta.shape)
        # print('boosted 4-momenta shape:', boosted_4_momenta[0])
       # print('total 4-momenta shape:', total_4_momenta.shape)
       # print('cluster kins shape:', cluster_kins[:,0,:4].shape)

       # print(valid)

        # Total event 4-momentum
        # rescaled_total_4_momenta_rest_frame = rescaled_4_momenta_rest_frame.sum(dim=1)  # (B,4)

        # print('rescaled total 4-momenta rest frame shape:', rescaled_total_4_momenta_rest_frame[:3], 
        #       ' should be close to cluster invariant mass:', cluster_invariant_masses[:3])

        # boost out of cluster rest frame
        #boosted_4_momenta = vectorized_boost(rescaled_4_momenta_rest_frame

       # print('xi shape:', xi.shape)
       # sys.exit()

        # print('3-momenta mags rest frame shape:', momenta_mag_rest_frame[0])
        # print('masses rest frame shape:', masses[0])


       # energies_rest_frame = momenta_4_rest_frame[:, :, 0]  # (B, S)


        # check of E = sum sqrt(p^2 + m^2) in rest frame
        # energies_rest_frame = momenta_4_rest_frame[:,:,0]
        # momenta_rest_frame = momenta_4_rest_frame[:,:,1:]
        # momenta_mag_rest_frame = torch.norm(momenta_rest_frame, dim=2)
        # masses_rest_frame = torch.sqrt(energies_rest_frame**2 - momenta_mag_rest_frame**2)
       # print('masses rest frame shape:', masses_rest_frame[0], masses[0])

       # print('momenta 4 rest frame shape:', momenta_4_rest_frame[0])
        
        
        
        
        #total_4_momenta_rest_frame = momenta_4_rest_frame.sum(dim=1)
        #print('total 4-momenta rest frame shape:', total_4_momenta_rest_frame[0])
        


        #print('total 4-momenta shape:', total_4_momenta.shape)
        #print('total 4-momenta shape:', total_4_momenta[0])
       # sys.exit()
        # boost to the rest frame of total_4_momenta
    #     boosted_4_momenta = lorentz_boost(momenta_4, total_4_momenta)
    #     boosted_total_4_momenta = boosted_4_momenta.sum(dim=1)
    #     print('boosted total 4-momenta shape:', boosted_total_4_momenta[0])
    #     # compute the invariant mass of the cluster
        
    #    # print('total 4-momenta shape:', total_4_momenta[0])
    


    #     # boost moemnta_4 to it's rest frame
    #     #print(output[0])
    #     #print(momenta_4[0])
    #     print('masses shape:', masses[0])
    #     print('3-momenta mags shape:', momenta_mag[0])

    #     cluster_invariant_masses = get_invariant_mass(cluster_kins[:, 0, :4].reshape(-1, 4))
    #     print('cluster invariant masses:', cluster_invariant_masses.shape)
    
    #     sys.exit()
    #     new_output = []
    #     start_time = time.time()
    #     for seq, E_CM, cluster in zip(output, cluster_invariant_masses, cluster_kins[:, 0, :4]):
    #         # Separating phase space variables and hadron IDs
    #         pad_mask = seq[:, self.space_phase_dim] == 1.0
    #         phase_space_vars = seq[~pad_mask, :self.space_phase_dim]
    #         pad_tokens = seq[pad_mask]
    #         hadron_ids = seq[~pad_mask, self.space_phase_dim + 1:]
    #         hadron_ids = torch.argmax(hadron_ids, dim=1)
            
    #         # Getting masses based on hadron PIDs
    #         pids = torch.tensor([self.pid_map[hadron_id.item()] for hadron_id in hadron_ids])
    #         masses = torch.tensor([Particle.from_pdgid(pid).mass / 1000 for pid in pids], device=seq.device)
    #         n_particles = len(masses)

    #         if masses.sum() < E_CM and n_particles >= 2:
    #             # Normalise phase_space_vars to [0, 1] with improved handling
    #             eps = 1e-8
    #             min_val = phase_space_vars.min()
    #             max_val = phase_space_vars.max()
    #             denom = max_val - min_val
    #             if denom.abs() < eps:
    #                 phase_space_vars_norm = phase_space_vars
    #             else:
    #                 phase_space_vars_norm = (phase_space_vars - min_val) / (denom + eps)

    #             # Flatten and pad/truncate to correct length for RamboOnDiet
    #             required_len = 3 * n_particles - 4

    #             print('required n_particles:', n_particles)
    #             flat = phase_space_vars_norm.flatten()
    #             if flat.numel() < required_len:
    #                 pad = torch.zeros(required_len - flat.numel(), device=flat.device, dtype=flat.dtype)
    #                 phase_space_vars_norm = torch.cat([flat, pad], dim=0).reshape(1, -1)
    #             else:
    #                 phase_space_vars_norm = flat[:required_len].reshape(1, -1)
                
    #             # Applying Rambo on diet to get the hadron four-momenta
    #             device = phase_space_vars.device
    #             try:
    #                 (p,), _ = self.rambo.map(inputs=[phase_space_vars_norm.to(device), torch.tensor([E_CM]).to(device)],
    #                                     nparticles=n_particles, masses=masses)
    #                 # Getting rid of the first two zeroed particles (initial state)
    #                 p = p[:, 2:]
    #                 # Applying the inverse Lorentz transformation (rest frame to lab frame)
    #                 p = lorentz_boost(p[0], cluster, inverse=True)
    #             except Exception as e:
    #                 # If Rambo fails, we just return zero momenta
    #                 print(f"RamboOnDiet failed: {e}. Returning zero momenta.")
    #                 p = torch.zeros((n_particles, 4), device=seq.device)
    #         else:
    #             # If the invariant mass is too large, we just return zero momenta
    #             p = torch.zeros((n_particles, 4), device=seq.device)
            
    #         # Adding padding tokens back to the output
    #         new_output.append(
    #             torch.cat([
    #                 torch.cat([p, seq[~pad_mask, self.space_phase_dim:]], dim=1),
    #                 torch.cat([
    #                     torch.zeros(pad_tokens.size(0), 1, device=pad_tokens.device, dtype=pad_tokens.dtype),
    #                     pad_tokens], dim=1)
    #             ], dim=0)
    #         )
    #     end_time = time.time()
    #     print(f"Rambo processing time for batch: {end_time - start_time} seconds")
    #    #
    #   #  sys.exit()
    #     output = torch.stack(new_output)
    #     return output


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
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hadrons):
        embedded_input = self.input_embedding_layer(hadrons)
        embedded_input = self.positional_encoding(embedded_input)
        
        # Preparing the causal mask
        # src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        #     embedded_input.size(1)).to(embedded_input.device)
        
        # embedded_output = self.transformer_encoder(embedded_input, mask=src_mask, is_causal=True)
        embedded_output = self.transformer_encoder(embedded_input)

        real_or_fake_response = self.output_embedding_layer(embedded_output)
        return real_or_fake_response.mean(dim=1)