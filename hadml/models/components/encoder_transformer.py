import math
import torch, os, pickle
from particle import Particle
from hadml.rambo.rambo_on_diet import RamboOnDiet, NearlyRambo
from hadml.rambo.additional_helpers import get_invariant_mass, lorentz_boost, vectorized_boost
from hadml.rambo.solve_xi import solve_xi
import sys
import time
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from hadml.utils.utils import lorentz_to_kt_eta_phi_m, pid_map
import time

# set anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

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
        
        masses = pid_map(pid_map_filepath)
#         self.pid_map = {idx: (0.0 if pid == 'uncommon_pid' else float(pid))
#                 for pid, idx in raw_pid_map.items()}
#         # check if uncommon_pid is in pid_map
#         find_pid = False
#         for pid in self.pid_map.values():
#             if pid == 0.0:
#                 find_pid = True
#         print('----------------------------')
#         print('----------------------------')
#         print('Uncommon PID found in pid_map:', find_pid)

#  # Masses in GeV.  (Negative PDG IDs = antiparticles -> same mass.)
#         manual_mass_map = {5212.0: 5.8112082, 5214.0: 5.8325324, 
#                            5314.0: 5.967868, 5322.0: 5.897625, 
#                            10511.0: 5.726344, 10513.0: 5.7207456, 
#                            10521.0: 5.726035, 10523.0: 5.720276, 
#                            10531.0: 5.8176956, 10533.0: 5.8293395, 
#                            13322.0: 1.689997, 15122.0: 5.912, 
#                            15312.0: 5.8176956, 15322.0: 6.1110806, 
#                            20413.0: 2.4376314, 20513.0: 5.7615266, 
#                            20523.0: 5.762014, 20533.0: 5.829, 
#                            100311.0: 1.4600005, 100321.0: 1.4595373}
#        # manual_mass_map = {5212.0: 5.8112082}
#         # what is the mass of 15312 ? 


#         masses = []
#         not_found = []

#         for pid in self.pid_map.values():
#             if pid == 0.0:
#                 masses.append(0.0)
#                 continue

#             pid_int = abs(float(pid))
            
#             # Check manual overrides first
#             if pid_int in manual_mass_map:
#                 masses.append(manual_mass_map[pid_int])
#                 continue
            
#             pid_int = int(pid)
#             # Otherwise try PDG lookup
#             try:
#                 m = Particle.from_pdgid(pid_int).mass / 1000  # -> GeV
#                 masses.append(m)
#             except Exception:
#                 masses.append(0.0)
#                 not_found.append(pid_int)

#         # Print summary of missing masses
#         if not_found:
#             print("masses not found for PIDs:", sorted(set(not_found)))
#             for pid_u in sorted(set(not_found)):
#                 print("mass not found, set to 0 for PID:", pid_u)

        # Create final tensor safely
        self.mass_tensor = torch.tensor(masses, dtype=torch.float32)
        

        self.style_dim = 32
        self.num_layers = num_layers
        # create mass_tensor
       # self.mass_tensor = torch.tensor([Particle.from_pdgid(pid).mass / 1000 for pid in self.pid_map.values()])

        n_hadron_types = len(raw_pid_map) + 1
       # n_hadron_types = len(raw_pid_map)

        self.quark_type_embedding_layer = torch.nn.Embedding(quark_types, quark_embedding_dim)
        self.input_embedding_layer = torch.nn.Linear(
            noise_dim + cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )

        # Style mapping network
        self.noise_mapping = nn.Sequential(
            nn.Linear(noise_dim, noise_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(noise_dim, noise_dim),
        )

        self.style_gamma = nn.Linear(noise_dim, embedding_dim)
        self.style_beta  = nn.Linear(noise_dim, embedding_dim)




        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 3 + n_hadron_types)
        #self.output_embedding_layer = torch.nn.Linear(embedding_dim, 2*(3 + n_hadron_types)+1)
        self.encoder_layers = nn.ModuleList([
                                StyleTransformerEncoderLayer(
                                    nn.TransformerEncoderLayer(
                                        d_model=embedding_dim,
                                        nhead=n_heads,
                                        dim_feedforward=dim_feedforward,
                                        batch_first=True,
                                    ),
                                    embedding_dim,
                                    noise_dim
                                )
                                for _ in range(num_layers)
                            ])

        self.noise_strength = nn.Parameter(torch.zeros(embedding_dim))

        self.conditional_embedding_gamma = nn.Linear(cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim)
        self.conditional_embedding_beta  = nn.Linear(cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim)

       # self.pair_model = CrossAttentionPairing(feature_dim=3 + n_hadron_types, embed_dim=embedding_dim, temp=0.5)
        # encoder_layer = torch.nn.TransformerEncoderLayer(
        #     d_model=embedding_dim, nhead=n_heads, 
        #     dim_feedforward=dim_feedforward, norm_first=False,
        #     batch_first=True)
        # self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
      #  self.rambo = RamboOnDiet()
        #self.rambo = NearlyRambo()

    def forward(self, noise, cluster_kins, plot=False):

        w = self.noise_mapping(noise)     # (N,32,16)
        #w_global = w.mean(dim=1)          # (N,16)
        w_global = w.mean(dim=1)  # (N,1,16)
       # sys.exit()
        gamma = self.style_gamma(w_global)  # (N,E)
        beta  = self.style_beta(w_global)

        embedded_quark_types = self.quark_type_embedding_layer(cluster_kins[:, :, 4:6].to(torch.int32))
        embedded_quark_types = embedded_quark_types.reshape(*embedded_quark_types.size()[:2], -1)
        cluster_kins = torch.concatenate((cluster_kins[:, :, :4], embedded_quark_types, 
                                          cluster_kins[:, :, 6:]), dim=2)
    
        # cond_gamma = self.conditional_embedding_gamma(cluster_kins)
        # cond_beta  = self.conditional_embedding_beta(cluster_kins)

       # print('cluster_kins shape after embedding:', cluster_kins.shape)
       # print('noise shape:', noise.shape)
       # sys.exit()
        #noise = noise[:, :cluster_kins.size(1), :]
        #clusters_and_noise = torch.concatenate((cluster_kins, noise), dim=2)
        clusters_and_noise = torch.cat((cluster_kins, w), dim=2)
        #print('clusters_and_noise shape:', clusters_and_noise.shape)
       # sys.exit()
        embedded_input = self.input_embedding_layer(clusters_and_noise)
        gamma = gamma.unsqueeze(1)  # (N,1,E)
        beta  = beta.unsqueeze(1)
     #   gamma = self.style_to_gamma(w_global).unsqueeze(1)  # (N,1,E)
     #   beta  = self.style_to_beta(w_global).unsqueeze(1)
    #    sys.exit()
        embedded_input = gamma * embedded_input + beta
       # print('after adain:', embedded_input.shape)
        #sys.exit()
        embedded_input = self.positional_encoding(embedded_input)
        
        # Preparing the causal mask
        # src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        #     embedded_input.size(1)).to(embedded_input.device)

        x = embedded_input
        for i, layer in enumerate(self.encoder_layers):
          #  print('layer', i)
            style = w_global if i < self.num_layers // 2 else w
            #style = w_global
            x = layer(x, style)
            #x = cond_gamma.unsqueeze(1) * x + cond_beta.unsqueeze(1)

            # if i >= self.num_layers - 2:
            #     eps = torch.randn_like(x[:, :, :1])
            #     x = x + eps * self.noise_strength
            #x = layer(x, w_global)
        embedded_output = x

        output = self.output_embedding_layer(embedded_output)
        output = torch.clamp(output, min=-1e6, max=1e6)

        p = torch.nn.functional.gumbel_softmax(output[:, :, self.space_phase_dim+1:], self.current_gumbel_temp, hard=False)
        #batch_mean = p.mean(dim=[0,1])  # (290,)
        #coverage_entropy = -(batch_mean * (batch_mean + 1e-12).log()).sum()
        # paddinglogits = output[:, :, self.space_phase_dim]  # (256, 32)

        # # Get index of smallest padding logit (top-1 particle)
        # top2_idx = paddinglogits.topk(2, dim=1, largest=False).indices  # (256, 2)

        # seq_len = paddinglogits.size(1)
        # # Create soft, differentiable mask for those positions
        # seq = torch.arange(seq_len, device=output.device)[None, :, None]  # (1, 32, 1)
        # top2_mask = (seq == top2_idx[:, None, :]).any(dim=2).float()  # (256, 32)

        # # Extract the logits slice you want to sample
        # logits_slice = output[:, :, self.space_phase_dim:]  # (256, 32, remaining_dims)

        # # Force top-2 positions to 0, others unchanged (differentiable)
        # forced_logits = torch.where(
        #     top2_mask[:, :, None].bool(),
        #     torch.zeros_like(logits_slice),
        #     logits_slice
        # )

        output[:, :, self.space_phase_dim:] = torch.nn.functional.gumbel_softmax(
            output[:, :, self.space_phase_dim:],
            self.current_gumbel_temp,
            hard=self.gumbel_softmax_hard
        )

        # output[:, :, self.space_phase_dim:] = torch.nn.functional.gumbel_softmax(
        #     forced_logits,
        #     self.current_gumbel_temp,
        #     hard=self.gumbel_softmax_hard
        # )
        

        #print(output[0].shape)
        #sys.exit()

        padding_token = output[:,:, self.space_phase_dim]
        non_padding_token = 1.0 - padding_token
       # non_padding_token = non_padding_token * (~pad_mask).float()
        # try with mask
        total_particles = torch.sum(non_padding_token, dim=1)
        non_pad_loss = torch.nn.ReLU()(2.0 - total_particles)
        #is_odd = (total_particles % 2).float()  # 0 for even, 1 for odd


        # Create pad mask
        pad_mask = output[:, :, self.space_phase_dim] == 1.0
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand_as(output)


        # Create the pure padding template
        pure_padding_template = torch.zeros_like(output)
        pure_padding_template[:, :, self.space_phase_dim] = 1.0

        # Replace using torch.where (preserves gradients!)
        output = torch.where(pad_mask_expanded, pure_padding_template, output)

        valid_mask = (padding_token == 0)  # True only for particle positions
        total_valid = valid_mask.sum().clamp(min=1)

        # Aggregate histogram only over valid (particle) positions
        hist = (p * valid_mask.unsqueeze(-1)).sum(dim=[0, 1]) / total_valid  # (290,)

        # Minibatch coverage entropy on non-padding tokens
        coverage_entropy = -(hist * (hist + 1e-12).log()).sum()
        # Moving all padding tokens to end of each sequence, keep relative order of the rest
        pad_indicator = pad_mask.to(torch.int64)    # 1 for pad, 0 otherwise
        order = torch.argsort(pad_indicator, dim=1) # non‐pads (0) come first
        output = torch.gather(output, dim=1, index=order.unsqueeze(2).expand(-1, -1, output.size(2)))


        # new pad_mask after reordering    
        pad_mask = output[:, :, self.space_phase_dim] == 1.0

        mass_table = self.mass_tensor.to(output.device)
        hadron_logits = output[:, :, self.space_phase_dim+1:]  # [B, N, 3]
        #print('hadron_logits shape:', hadron_logits.shape)
        #print('mass_table shape:', mass_table.shape)
        masses = hadron_logits @ mass_table                    # [B, N]

        #print('masses before masking:', masses.shape)
        # sys.exit()

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

        #rescaled_4_momenta_rest_frame[~valid] = 0.0

       # rescaled_4_mo
        #output = output.clone()  # breaks view + resets versioning issues, fast on GPU
        # output[~valid] = 0.0 

        cluster_4_momenta = cluster_kins[:,0,:4]
        boosted_4_momenta = vectorized_boost(rescaled_4_momenta_rest_frame, 
                                             cluster_4_momenta, inverse=True,
                                             print_warnings=False)
    
        
        boosted_4_momenta = boosted_4_momenta * pad_mask_float.unsqueeze(2)  # zero out padding particles

        boosted_4_momenta[~valid] = 0.0

        # hadron_zeros = torch.zeros_like(output[:, :, self.space_phase_dim+1:])
        # # put output in new_output where valid

        # new_output = torch.where(
        #     valid.unsqueeze(-1).unsqueeze(-1).expand_as(output[:,:, self.space_phase_dim+1:]),
        #     output[:,:, self.space_phase_dim+1:],
        #     hadron_zeros)

            
        # new_output = torch.cat(
        #     (boosted_4_momenta, output[:, :, self.space_phase_dim].unsqueeze(-1) ,new_output), dim=2
        # )



        new_output = torch.cat(
            (boosted_4_momenta, output[:, :, self.space_phase_dim:]), dim=2
        )

        


        if plot==True:
            # new_output_invariant_mass = torch.sqrt(torch.clamp(
            #     boosted_4_momenta[:, :, 0]**2 - torch.sum(boosted_4_momenta[:, :, 1:]**2, dim=2),
            #     min=0.0
            # )) 

            print('rescaled_4_momenta_rest_frame:', rescaled_4_momenta_rest_frame[valid][0])
            print('boosted_4_momenta:', boosted_4_momenta[valid][0])
            print('cluster_4_momenta:', cluster_4_momenta[valid][0])

            new_output_invariant_mass = get_invariant_mass(boosted_4_momenta)  # [B, N]
            # print('new_output_invariant_mass before masking:', new_output_invariant_mass.shape)
            new_output_invariant_mass = new_output_invariant_mass[valid]
            print('new_output_invariant_mass:', new_output_invariant_mass[0])


            masses_new_output_pid = new_output[:, :, self.space_phase_dim+2:] @ mass_table  # [B, N]
            print('masses_new_output_pid:', masses_new_output_pid[valid][0])
            #masses_new_output_pid = masses_new_output_pid[pad_mask_float.bool()]
            masses_new_output_pid = masses_new_output_pid[valid]

            masses_original = masses[valid]
            #masses_original = masses[pad_mask_float.bool()] 

            plt.figure()
            plt.hist(masses_new_output_pid.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="pid mass")
            plt.hist(new_output_invariant_mass.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="invariant mass")
            plt.hist(masses_original.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="original mass")
            plt.xlabel("Mass from PID")
            plt.ylabel("Density")
            plt.legend()
            plt.xlim(0, 6)
            plt.yscale("log")
            plt.savefig("masses_from_pid_generator.png")
            plt.close()



        #sys.exit()
        #masses_new_output_pid = masses_new_output_pid * pad_mask_float  # padding -> 0, real -> keeps value

        

        if torch.isnan(new_output).any():
            print('NaNs detected in output')
            # find which batch entries have nans
            nan_batches = torch.isnan(new_output).any(dim=(1,2)).nonzero(as_tuple=True)[0]

            # print the nan entries
            for batch_idx in nan_batches:
                print(f'NaNs in batch entry {batch_idx}:')
                print(new_output[batch_idx])


            sys.exit()

        # padding_token = output[:,:, self.space_phase_dim]
        # non_padding_token = 1.0 - padding_token
        # total_particles = torch.sum(non_padding_token, dim=1)
        # non_pad_loss = torch.nn.ReLU()(2.0 - total_particles)
        # is_odd = (total_particles % 2).float()  # 0 for even, 1 for odd



        return new_output, violation**2, non_pad_loss, coverage_entropy, valid


class CachedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def _project_qkv(self, x):
        # x: [B,1,D]
        W = self.self_attn.in_proj_weight
        b = self.self_attn.in_proj_bias

        D = self.self_attn.embed_dim

        Wq, Wk, Wv = W[:D], W[D:2*D], W[2*D:]
        bq, bk, bv = b[:D], b[D:2*D], b[2*D:]

        q = torch.nn.functional.linear(x, Wq, bq)
        k = torch.nn.functional.linear(x, Wk, bk)
        v = torch.nn.functional.linear(x, Wv, bv)

        return q, k, v

    def forward(self, src):
        # src must be [B,1,D]

        q, k, v = self._project_qkv(src)

        B, S, D = q.shape
        H = self.self_attn.num_heads
        head_dim = D // H

        # -> [B,H,1,head_dim]
        q = q.view(B, S, H, head_dim).transpose(1,2)
        k = k.view(B, S, H, head_dim).transpose(1,2)
        v = v.view(B, S, H, head_dim).transpose(1,2)

        # ---- append cache ----
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)

        k_all = self.k_cache
        v_all = self.v_cache

        # attention: query only attends to past
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_all, v_all, is_causal=False
        )

        # back to [B,1,D]
        attn = attn.transpose(1,2).contiguous().view(B, S, D)
        src2 = self.self_attn.out_proj(attn)

        # identical residual + FFN as PyTorch
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src





class Generator2(torch.nn.Module):
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
        
        masses = pid_map(pid_map_filepath)

        self.mass_tensor = torch.tensor(masses, dtype=torch.float32)
        self.embedding_dim = embedding_dim

        self.style_dim = 32
        self.num_layers = num_layers
        # create mass_tensor
       # self.mass_tensor = torch.tensor([Particle.from_pdgid(pid).mass / 1000 for pid in self.pid_map.values()])

        n_hadron_types = len(raw_pid_map)
       # n_hadron_types = len(raw_pid_map)

        self.quark_type_embedding_layer = torch.nn.Embedding(quark_types, quark_embedding_dim)
        # self.input_embedding_layer = torch.nn.Linear(
        #     noise_dim + cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        # )
        self.noise_embedding_layer = torch.nn.Linear(noise_dim, embedding_dim)
        self.cluster_embedding_scale = torch.nn.Linear(
            1+cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )
        self.cluster_embedding_bias = torch.nn.Linear(
            1+cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim
        )


        self.input_embedding_layer = torch.nn.Linear(3 + n_hadron_types, embedding_dim)


        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 3 + n_hadron_types)



        self.rem_mass_prediction = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        self.rem_mass_embedding = nn.Linear(1 + cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim)
        # self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(
        #                                 d_model=embedding_dim,
        #                                 nhead=n_heads,
        #                                 dim_feedforward=dim_feedforward,
        #                                 batch_first=True,
        #                             )
        #                             for _ in range(num_layers)
        #                         ])
        self.encoder_layers = nn.ModuleList([
            CachedTransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # self.register_buffer("causal_mask", 
        #     torch.triu(torch.ones(30, 30) * float('-inf'), diagonal=1)
        # )
        
        
        #self.output_embedding_layer = torch.nn.Linear(embedding_dim, 2*(3 + n_hadron_types)+1)
        # self.encoder_layers = nn.ModuleList([
        #                         StyleTransformerEncoderLayer(
        #                             nn.TransformerEncoderLayer(
        #                                 d_model=embedding_dim,
        #                                 nhead=n_heads,
        #                                 dim_feedforward=dim_feedforward,
        #                                 batch_first=True,
        #                             ),
        #                             embedding_dim,
        #                             noise_dim
        #                         )
        #                         for _ in range(num_layers)
        #                     ])

        # self.noise_strength = nn.Parameter(torch.zeros(embedding_dim))

        # self.conditional_embedding_gamma = nn.Linear(cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim)
        # self.conditional_embedding_beta  = nn.Linear(cluster_data_dim - n_quarks + n_quarks * quark_embedding_dim, embedding_dim)



    def forward(self, noise, cluster_kins, plot=False):
        torch.autograd.set_detect_anomaly(True)
        for layer in self.encoder_layers:
            layer.reset_cache()

        embedded_quark_types = self.quark_type_embedding_layer(cluster_kins[:, :, 4:6].to(torch.int32))
        embedded_quark_types = embedded_quark_types.reshape(*embedded_quark_types.size()[:2], -1)
        cluster_kins = torch.concatenate((cluster_kins[:, :, :4], embedded_quark_types, 
                                          cluster_kins[:, :, 6:]), dim=2)

        # generate random noise for rem_mass_prediction
        rem_mass_noise = torch.randn_like(noise[:, 0, :1])
        cluster_and_rem_mass_noise = torch.cat((cluster_kins[:, 0, :], rem_mass_noise), dim=1)
        
        rem_mass_embedding = self.rem_mass_embedding(cluster_and_rem_mass_noise)
        rem_mass_fraction = self.rem_mass_prediction(rem_mass_embedding)  # (N,1)

        cluster_masses = get_invariant_mass(cluster_kins[:, 0, :4]).unsqueeze(1)  # (N,1)
        rem_masses = rem_mass_fraction * cluster_masses  # (N,1)

        temp_rem_masses = rem_masses

        summed_mass_tensor = self.mass_tensor.to(noise.device)

        #mask_allowed = torch.ones_like(summed_mass_tensor)
        B, T = noise.size(0), noise.size(1)
        D = 3 + len(self.mass_tensor)
        alive = torch.ones(B, dtype=torch.bool).to(noise.device)
        #x = torch.zeros(B,T,D, device=noise.device)  # placeholder for transformer input, will be filled iteratively
        _x  = torch.zeros(B,T,D, device=noise.device)
        #start = time.time()
        n_hadrons = torch.zeros(B, dtype=torch.int32).to(noise.device)

        # embedded_input_buffer
       # self.register_buffer("embedded_input", 
        #                     torch.zeros(B, T, self.embedding_dim, device=noise.device))

        x = []
        masses_x_sum = torch.zeros(B, 1).to(noise.device)
        for seq_idx in range(T):

            if alive.sum() == 0:
              #  end = time.time()
              #  print(f'Generation finished in {end-start:.2f} seconds')
                # print('All batches finished generation at seq idx', seq_idx)
                break

            cluster_kins_and_rem_mass = torch.cat((cluster_kins[:,0,:], temp_rem_masses), dim=1)
            
            if seq_idx == 0:
                noise = noise[:, :seq_idx+1, :]
            
            conditional_input = cluster_kins_and_rem_mass.unsqueeze(1).repeat(1, seq_idx+1, 1)
            conditional_input_scale = self.cluster_embedding_scale(conditional_input)
            conditional_input_bias  = self.cluster_embedding_bias(conditional_input)


            if seq_idx == 0:
               # print('self embedded_input shape:', self.embedded_input.shape)
                #print('noise shape:', noise.shape)
                #sys.exit()
                #self.embedded_input[:, :seq_idx+1, :] = self.noise_embedding_layer(noise) 
                embedded_input = self.noise_embedding_layer(noise)
                #embedded_noise = embedded_input
            else:
                embedded_input = self.input_embedding_layer(x[-1])

                #previous_output = self.input_embedding_layer(x[-1].squeeze(1))
                #embedded_input = torch.cat((embedded_input, previous_output.unsqueeze(1)), dim=1)

            #print('embedded_input shape at seq idx', seq_idx, ':', embedded_input.shape)
            input = conditional_input_scale * embedded_input + conditional_input_bias
            #input = conditional_input_scale * self.embedded_input[:, :seq_idx+1, :] + conditional_input_bias
            positional_encoded_input = self.positional_encoding(input)

            # mask = torch.triu(torch.ones(positional_encoded_input.size(1), 
            #                              positional_encoded_input.size(1)) * float('-inf'), 
            #                              diagonal=1).to(positional_encoded_input.device)
        


            if seq_idx == 0:
               # print('temp remaining masses at seq idx', seq_idx, ':', temp_rem_masses[0])
                masked_mass = self.mass_tensor.to(noise.device) < temp_rem_masses.unsqueeze(2)  # [B, 1]
   
            else:

                expanded_mass_tensor = summed_mass_tensor.unsqueeze(0).repeat(B, 1)  # [B, num_hadron_types]
                #expanded_mass_tensor = expanded_mass_tensor + temp_rem_masses

                masked_mass = expanded_mass_tensor < temp_rem_masses  # [B, 1]
              #  print('masked_mass.shape', masked_mass.shape)

                masked_mass = masked_mass.unsqueeze(1)  # [B, 1, num_hadron_types]


            next_token = positional_encoded_input
            for layer in self.encoder_layers:
                #next_token = layer(next_token, src_mask=mask)
                next_token = layer(next_token)


            next_token = self.output_embedding_layer(next_token)      
            
            next_token = next_token * alive[:, None, None]  # zero out output for dead batches
            logits = next_token[:, seq_idx:seq_idx+1, self.space_phase_dim:]

            masked_mass = masked_mass.bool()
            n_valid_hadrons = masked_mass.sum(dim=2) * alive[:, None]
           # print('number of valid hadrons at seq idx', seq_idx, ':', n_valid_hadrons[0])
           # print('logits before masking at seq idx', seq_idx, ':', logits[0])
            n_valid_hadrons = n_valid_hadrons > 0.0
            int_n_valid_hadrons = n_valid_hadrons.squeeze(1).to(torch.int32)
            n_hadrons = n_hadrons + int_n_valid_hadrons
            #masked_logits = logits
            masked_logits = logits.masked_fill(~masked_mass, -1e16)  # [B, 1, num_hadron_types]
           # print('masked_logits at seq idx', seq_idx, ':', masked_logits[0])
            #print('masked_logits at seq idx', seq_idx, ':', masked_logits[0])
            pid_next_token = torch.nn.functional.gumbel_softmax(
                masked_logits,
                tau=self.current_gumbel_temp,
                hard=self.gumbel_softmax_hard,
                dim=-1
            )
           # print('pid_next_token at seq idx', seq_idx, ':', pid_next_token[0])

            #x[:, seq_idx:seq_idx+1, :self.space_phase_dim] = next_token[:, seq_idx:seq_idx+1, :self.space_phase_dim] * n_valid_hadrons.unsqueeze(2)
            #x[:, seq_idx:seq_idx+1, self.space_phase_dim:] = pid_next_token * n_valid_hadrons.unsqueeze(2)
            x.append(torch.cat((next_token[:, seq_idx:seq_idx+1, :self.space_phase_dim], 
                                pid_next_token), dim=2) * n_valid_hadrons.unsqueeze(2))
           # x[:, seq_idx:seq_idx+1,:] = x[:, seq_idx:seq_idx+1,:] * n_valid_hadrons.unsqueeze(2)
            masses_x = x[-1][:, :, self.space_phase_dim:] @ self.mass_tensor.to(noise.device)  # [B, N]
            #print('masses_x at seq idx', seq_idx, ':', masses_x[0])
           # print('masses_x at seq idx', seq_idx, ':', masses_x[0])
            
            masses_x_sum += masses_x.sum(dim=1, keepdim=True)  # [B, 1]
            #print('masses_x_sum at seq idx', seq_idx, ':', masses_x_sum[0])
            #print('rem_masses at seq idx', seq_idx, ':', rem_masses[0])
            temp_rem_masses = rem_masses - masses_x_sum


            alive = n_valid_hadrons.squeeze(1)
            
            #previous_output = next_token[:, seq_idx:seq_idx+1, :]

        padding = torch.ones(B, T, 1, device=noise.device)
        # first n_hadrons positions are 0, rest are padding 1

        x = torch.cat(x, dim=1).to(noise.device)  # (B, T, D)
        

        for i in range(B):
            padding[i, :n_hadrons[i], 0] = 0.0
            zero_padding = torch.zeros_like(_x[i, n_hadrons[i]:, :])
            _x[i, :n_hadrons[i], :] = x[i, :n_hadrons[i], :]
            _x[i, n_hadrons[i]:, :] = zero_padding
            #x[i]
            # add zeroes to x
            #x[i, n_hadrons[i], :] = 0.0

        #print(x.)
        plt.figure()
        plt.hist(n_hadrons.detach().cpu().numpy().flatten(), bins=30,
                    histtype="step", linewidth=2)
        plt.xlabel("Number of hadrons")
        plt.ylabel("Counts")
        plt.yscale("log")
        plt.savefig("n_hadrons_generator.png")
        plt.close()
        #sys.exit()
        #padding = padding.detach()  # ensure no gradients through padding

        #print('n_hadrons:', n_hadrons[0])
        #print('padding after generation:', padding[0])
        #print('x shape before padding:', x.shape)
        #print('padding shape:', padding.shape)
        #print('x shape before padding:', _x.shape)
        #print('padding shape:', padding.shape)
        #sys.exit()

        output = torch.cat((_x[:, :, :self.space_phase_dim], padding, _x[:, :, self.space_phase_dim:]), dim=2)


        #output = torch.clamp(output, min=-1e6, max=1e6)

        p = torch.nn.functional.gumbel_softmax(output[:, :, self.space_phase_dim+1:], self.current_gumbel_temp, hard=False)


        padding_token = output[:,:, self.space_phase_dim]

        # Create pad mask
        valid_mask = (padding_token == 0)  # True only for particle positions
        total_valid = valid_mask.sum().clamp(min=1)

        # Aggregate histogram only over valid (particle) positions
        hist = (p * valid_mask.unsqueeze(-1)).sum(dim=[0, 1]) / total_valid  # (290,)

        # Minibatch coverage entropy on non-padding tokens
        coverage_entropy = -(hist * (hist + 1e-12).log()).sum()
        # Moving all padding tokens to end of each sequence, keep relative order of the rest

        # new pad_mask after reordering    
        pad_mask = output[:, :, self.space_phase_dim] == 1.0

        mass_table = self.mass_tensor.to(output.device)
        hadron_logits = output[:, :, self.space_phase_dim+1:]  # [B, N, 3]
        #print('hadron_logits shape:', hadron_logits.shape)
        #print('mass_table shape:', mass_table.shape)
        masses = hadron_logits @ mass_table                    # [B, N]

        #print('masses before masking:', masses.shape)
        # sys.exit()

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

        cluster_4_momenta = cluster_kins[:,0,:4]
        boosted_4_momenta = vectorized_boost(rescaled_4_momenta_rest_frame, 
                                             cluster_4_momenta, inverse=True,
                                             print_warnings=False)
    
        
        boosted_4_momenta = boosted_4_momenta * pad_mask_float.unsqueeze(2)  # zero out padding particles

        boosted_4_momenta[~valid] = 0.0

        # hadron_zeros = torch.zeros_like(output[:, :, self.space_phase_dim+1:])
        # # put output in new_output where valid

        # new_output = torch.where(
        #     valid.unsqueeze(-1).unsqueeze(-1).expand_as(output[:,:, self.space_phase_dim+1:]),
        #     output[:,:, self.space_phase_dim+1:],
        #     hadron_zeros)

            
        # new_output = torch.cat(
        #     (boosted_4_momenta, output[:, :, self.space_phase_dim].unsqueeze(-1) ,new_output), dim=2
        # )



        new_output = torch.cat(
            (boosted_4_momenta, output[:, :, self.space_phase_dim:]), dim=2
        )

        


        if plot==True:
            # new_output_invariant_mass = torch.sqrt(torch.clamp(
            #     boosted_4_momenta[:, :, 0]**2 - torch.sum(boosted_4_momenta[:, :, 1:]**2, dim=2),
            #     min=0.0
            # )) 

            print('rescaled_4_momenta_rest_frame:', rescaled_4_momenta_rest_frame[valid][0])
            print('boosted_4_momenta:', boosted_4_momenta[valid][0])
            print('cluster_4_momenta:', cluster_4_momenta[valid][0])

            new_output_invariant_mass = get_invariant_mass(boosted_4_momenta)  # [B, N]
            # print('new_output_invariant_mass before masking:', new_output_invariant_mass.shape)
            new_output_invariant_mass = new_output_invariant_mass[valid]
            print('new_output_invariant_mass:', new_output_invariant_mass[0])


            masses_new_output_pid = new_output[:, :, self.space_phase_dim+2:] @ mass_table  # [B, N]
            print('masses_new_output_pid:', masses_new_output_pid[valid][0])
            #masses_new_output_pid = masses_new_output_pid[pad_mask_float.bool()]
            masses_new_output_pid = masses_new_output_pid[valid]

            masses_original = masses[valid]
            #masses_original = masses[pad_mask_float.bool()] 

            plt.figure()
            plt.hist(masses_new_output_pid.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="pid mass")
            plt.hist(new_output_invariant_mass.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="invariant mass")
            plt.hist(masses_original.detach().cpu().numpy().flatten(), bins=100, density=False, 
                    histtype="step", linewidth=2, label="original mass")
            plt.xlabel("Mass from PID")
            plt.ylabel("Density")
            plt.legend()
            plt.xlim(0, 6)
            plt.yscale("log")
            plt.savefig("masses_from_pid_generator.png")
            plt.close()



        #sys.exit()
        #masses_new_output_pid = masses_new_output_pid * pad_mask_float  # padding -> 0, real -> keeps value

        

        if torch.isnan(new_output).any():
            print('NaNs detected in output')
            # find which batch entries have nans
            nan_batches = torch.isnan(new_output).any(dim=(1,2)).nonzero(as_tuple=True)[0]

            # print the nan entries
            for batch_idx in nan_batches:
                print(f'NaNs in batch entry {batch_idx}:')
                print(new_output[batch_idx])


            sys.exit()

        # padding_token = output[:,:, self.space_phase_dim]
        # non_padding_token = 1.0 - padding_token
        # total_particles = torch.sum(non_padding_token, dim=1)
        # non_pad_loss = torch.nn.ReLU()(2.0 - total_particles)
        # is_odd = (total_particles % 2).float()  # 0 for even, 1 for odd
        
       # print('output shape:', output.shape)
       # print('violation:', violation)

       # sys.exit()
        

        return new_output, violation**2, rem_masses, coverage_entropy, valid


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
        #self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + n_hadron_types, embedding_dim)
       # self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + 2, embedding_dim)

        self.hadron_embedding_layer = torch.nn.Linear(n_hadron_types, 32)
        self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + 5 + 32, embedding_dim)

        #self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + 5, embedding_dim)

        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            raw_pid_map = pickle.load(f)             # {pid: index}
        
        masses = pid_map(pid_map_filepath)


        # Create final tensor safely
        self.mass_tensor = torch.tensor(masses, dtype=torch.float32)


    def forward(self, hadrons):
        # Padding masks
        pad = (hadrons[:, :, 4] == 1)  # (batch, seq)
        fully_pad = pad.all(dim=1)    # (batch,)

        hadron_pids = hadrons[:, :, 4:]  # (batch, seq, n_hadron_types)
        hadron_pid_embeddings = self.hadron_embedding_layer(hadron_pids)  # (batch, seq, 32)


        #rem_mass_score = self.rem_mass_discriminator(rem_masses) if rem_masses is not None else None
        #hadron_input = hadrons
        hadron_4_momenta = hadrons[:, :, :4]  # (batch, seq, 4)
        hadron_pt_eta_phi = lorentz_to_kt_eta_phi_m(hadron_4_momenta)
        hadron_pt_eta_phi = hadron_pt_eta_phi * (~pad).unsqueeze(-1).float()  # zero out padding particles

        hadron_input = torch.cat((hadron_4_momenta, hadron_pt_eta_phi, hadrons[:, :, 4:5]), dim=2)  # (batch, seq, 9)
        hadron_input = torch.cat((hadron_input, hadron_pid_embeddings), dim=2)  # (batch, seq, 9+32)
       
        #print('Discriminator hadron_input shape:', hadron_input.shape)
       # print(hadrons[:, :, 4:5].shape)
       # print(hadron_pt_eta_phi.shape)
       # sys.exit()
        #hadron_input = torch.cat((hadron_pt_eta_phi, hadrons[:, :, 4:5]), dim=2)  # (batch, seq, 6)
        #print('Discriminator hadron_input shape:', hadron_input)
       # print('Discriminator hadron_input shape:', hadron_input)
        # hadron_pid_logits = hadrons[:, :, 5:]  # (batch, seq, n_hadron_types)
        # hadron_masses = hadron_pid_logits @ self.mass_tensor.to(hadrons.device)  # (batch, seq)
        # hadron_masses = hadron_masses * (~pad).float()
        # hadron_masses = hadron_masses.unsqueeze(-1)

        # hadron_input = torch.cat((hadrons[:, :, :5], hadron_masses), dim=2)  # (batch, seq, 6)
        #hadron_input = torch.cat((hadron_pt_eta_phi, hadrons[:, :, 4:6]), dim=2)  # (batch, seq, 6)

        

        # 1. Initialize pooled to all zeroes
        pooled = torch.zeros((hadrons.size(0),), device=hadrons.device, dtype=torch.float32) + 0.5

        # 2. Select non-fully-padded events
        valid = ~fully_pad  # (batch,)

        if valid.any():
            # Run operations only on valid subset
            x_valid = self.input_embedding_layer(hadron_input[valid])
           # print('Discriminator embedded input shape:', x_valid.shape)
            x_valid = self.positional_encoding(x_valid)

            out_valid = self.transformer_encoder(x_valid, src_key_padding_mask=pad[valid])

            logits_valid = self.output_embedding_layer(out_valid)
            logits_valid = torch.where(pad[valid].unsqueeze(-1), torch.zeros_like(logits_valid), logits_valid)

            mask_valid = (~pad[valid]).unsqueeze(-1).float()

            logits_sum_valid = logits_valid.sum(dim=1).squeeze(-1)  # (valid_batch,)
            mask_sum_valid = mask_valid.sum(dim=1).squeeze(-1).clamp(min=1.0)  # (valid_batch,)

            pooled_valid = logits_sum_valid / mask_sum_valid  # (valid_batch,)

            # 3. Scatter back into the original zero-initialized tensor
            pooled = pooled.masked_scatter(valid, pooled_valid)
        #print('Discriminator pooled output shape:', pooled.shape)
        # pooled remains zero for fully padded events automatically
        return pooled


class Discriminator2(torch.nn.Module):
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
        #self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + n_hadron_types, embedding_dim)
       # self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + 2, embedding_dim)
        self.input_embedding_layer = torch.nn.Linear(hadron_kins_dim + 5, embedding_dim)

        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.output_embedding_layer = torch.nn.Linear(embedding_dim, 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



        with open(os.path.normpath(pid_map_filepath), "rb") as f:
            raw_pid_map = pickle.load(f)             # {pid: index}
        
        masses = pid_map(pid_map_filepath)


        # Create final tensor safely
        self.mass_tensor = torch.tensor(masses, dtype=torch.float32)
        self.rem_mass_discriminator = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, hadrons, rem_masses=None):
        # Padding masks
        pad = (hadrons[:, :, 4] == 1)  # (batch, seq)
        fully_pad = pad.all(dim=1)    # (batch,)


        rem_mass_score = self.rem_mass_discriminator(rem_masses) if rem_masses is not None else None
        #hadron_input = hadrons
        hadron_4_momenta = hadrons[:, :, :4]  # (batch, seq, 4)
        hadron_pt_eta_phi = lorentz_to_kt_eta_phi_m(hadron_4_momenta)
        hadron_pt_eta_phi = hadron_pt_eta_phi * (~pad).unsqueeze(-1).float()  # zero out padding particles

        hadron_input = torch.cat((hadron_4_momenta, hadron_pt_eta_phi, hadrons[:, :, 4:5]), dim=2)  # (batch, seq, 9)
       # print(hadrons[:, :, 4:5].shape)
       # print(hadron_pt_eta_phi.shape)
       # sys.exit()
        #hadron_input = torch.cat((hadron_pt_eta_phi, hadrons[:, :, 4:5]), dim=2)  # (batch, seq, 6)
        #print('Discriminator hadron_input shape:', hadron_input)
       # print('Discriminator hadron_input shape:', hadron_input)
        # hadron_pid_logits = hadrons[:, :, 5:]  # (batch, seq, n_hadron_types)
        # hadron_masses = hadron_pid_logits @ self.mass_tensor.to(hadrons.device)  # (batch, seq)
        # hadron_masses = hadron_masses * (~pad).float()
        # hadron_masses = hadron_masses.unsqueeze(-1)

        # hadron_input = torch.cat((hadrons[:, :, :5], hadron_masses), dim=2)  # (batch, seq, 6)
        #hadron_input = torch.cat((hadron_pt_eta_phi, hadrons[:, :, 4:6]), dim=2)  # (batch, seq, 6)

        

        # 1. Initialize pooled to all zeroes
        pooled = torch.zeros((hadrons.size(0),), device=hadrons.device, dtype=torch.float32) + 0.5

        # 2. Select non-fully-padded events
        valid = ~fully_pad  # (batch,)

        if valid.any():
            # Run operations only on valid subset
            x_valid = self.input_embedding_layer(hadron_input[valid])
           # print('Discriminator embedded input shape:', x_valid.shape)
            x_valid = self.positional_encoding(x_valid)

            out_valid = self.transformer_encoder(x_valid, src_key_padding_mask=pad[valid])

            logits_valid = self.output_embedding_layer(out_valid)
            logits_valid = torch.where(pad[valid].unsqueeze(-1), torch.zeros_like(logits_valid), logits_valid)

            mask_valid = (~pad[valid]).unsqueeze(-1).float()

            logits_sum_valid = logits_valid.sum(dim=1).squeeze(-1)  # (valid_batch,)
            mask_sum_valid = mask_valid.sum(dim=1).squeeze(-1).clamp(min=1.0)  # (valid_batch,)

            pooled_valid = logits_sum_valid / mask_sum_valid  # (valid_batch,)

            # 3. Scatter back into the original zero-initialized tensor
            pooled = pooled.masked_scatter(valid, pooled_valid)
        #print('Discriminator pooled output shape:', pooled.shape)
        # pooled remains zero for fully padded events automatically
        return pooled, rem_mass_score




def discretize_padding(logits):
    y_soft = torch.sigmoid(logits)
    y_hard = (y_soft > 0.5).float()

    y = y_hard.detach() - y_soft.detach() + y_soft

    return y

def gumbel_sigmoid(logits, tau=1.0, hard=False):
    #g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    g = torch.distributions.Gumbel(0,1).sample(logits.shape).to(logits.device)
    y = torch.sigmoid((logits + g) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard.detach() - y.detach() + y
    return y


def share_padding_in_pairs(output):
    """
    output: (256, 36, 7) where feature at index 2 is padding
    Group particles into pairs: (0,1), (2,3), (4,5), etc.
    For each pair, compute mean padding and assign to both particles
    """
    batch_size, num_particles, num_features = output.shape
    
    # Reshape to group into pairs: (256, 18, 2, 7)
    paired = output.view(batch_size, num_particles // 2, 2, num_features)
    
    # Compute mean padding for each pair
    # paired[:, :, :, 2] has shape (256, 18, 2) - padding for both particles in each pair
    mean_padding = paired[:, :, :, 2].mean(dim=2, keepdim=True)  # (256, 18, 1)
    
    # Assign mean padding to both particles in the pair
    paired[:, :, 0, 3] = mean_padding.squeeze(-1)  # First particle in pair
    paired[:, :, 1, 3] = mean_padding.squeeze(-1)  # Second particle in pair
    
    # Reshape back to original shape
    result = paired.view(batch_size, num_particles, num_features)
    
    return result

def logits_to_binary_gumbel(logits, temperature=1.0, hard=True):
    """
    Convert logits to binary using Gumbel-Softmax trick
    Most common and effective method for training
    """
    # Add dimension for 2 classes: [class_1_logits, class_0_logits]
    binary_logits = torch.stack([logits, -logits], dim=-1)  # Shape: (256, 2)
    
    # Apply Gumbel-Softmax
    binary_probs = torch.nn.functional.gumbel_softmax(
        binary_logits,
        tau=temperature,
        hard=hard,
        dim=-1
    )
    
    # Take probability of class 1 (positive)
    binary_values = binary_probs[:, 0]  # Shape: (256,)
    
    return binary_values

def binarize_with_shared_padding_vectorized(
    output, padding_dim, temperature=1.0, hard=True, cross_attention=False, pair_model=None
):
    """
    If cross_attention=True, pair_model (CrossAttentionPairing) is used for learned pairing.
    Else, original adjacent pairing is used.
    """
    N, particles, features = output.shape
    assert particles % 2 == 0, "Need even number of particles for pairing"

    if not cross_attention:
        # --- Original fixed adjacent pairing ---
        n_pairs = particles // 2
        paired = output.view(N, n_pairs, 2, features)
        padding_logits_pair = paired[:, :, :, padding_dim]  # (N, n_pairs, 2)
        mean_padding_logits = padding_logits_pair.mean(dim=-1)  # (N, n_pairs)

    else:
        # --- Learned pairing via CrossAttentionPairing model ---
        assert pair_model is not None, "pair_model must be provided when cross_attention=True"

        # Use model to produce soft paired representations
        # Expected output: (N, 16, 2, 7) for 32 particles → 16 learned pairs
        paired = pair_model(output)  # (N, 16, 2, 7)

        n_pairs = paired.shape[1]  # 16
        # Compute pair-mean padding logits from model-formed pairs
        mean_padding_logits = paired[:, :, :, padding_dim].mean(dim=-1)  # (N, 16)

    # ---- Select top pair to always be real ----
    top_pair = mean_padding_logits.topk(1, dim=1).indices  # (N, 1)

    # Build mask marking the selected top pair
    is_top = torch.zeros((N, n_pairs), device=output.device, dtype=torch.bool)
    is_top.scatter_(1, top_pair, True)

    # ---- Binarize all other pairs using Gumbel-Sigmoid ----
    stochastic_logits = mean_padding_logits.masked_fill(is_top, -1e6)
    binary_pairs = gumbel_sigmoid(stochastic_logits, tau=temperature, hard=hard)  # (N, 16)

    # Force the top pair to be real (padding=0)
    binary_pairs = binary_pairs.scatter(1, top_pair, 0.0)

    # ---- Impose shared padding value on both particles inside each learned pair ----
    shared = binary_pairs.unsqueeze(-1).expand(N, n_pairs, 2)  # (N, 16, 2)
    # Top pair is 0, others come from Gumbel sampling
    pad_assign = shared[:, :, 0]  # same for both members of each pair

    paired[:, :, 0, padding_dim] = pad_assign
    paired[:, :, 1, padding_dim] = pad_assign

    # ---- If learned pairing was used, collapse back to (N, 32, 7) format ----
    if cross_attention:
        # paired is (N, 16, 2, 7) → reshape to (N, 32, 7)
        result = paired.reshape(N, particles, features)
    else:
        result = paired.view(N, particles, features)

    # Final GAN safety guard: finite + bounded
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clamp(-10.0, 10.0)

    return result



from torch import nn

class AdaptiveLayerNorm(nn.Module):
    """AdaIN that accepts per-layer style vectors"""
    def __init__(self, normalized_shape, style_dim):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.feature_dim = normalized_shape[0]  # 128
        self.style_dim = style_dim  # 32
        
        # Project style to feature dimension
        self.style_to_scale = nn.Linear(style_dim, self.feature_dim)
        self.style_to_bias = nn.Linear(style_dim, self.feature_dim)
        
    def forward(self, x, style_vectors, layer_idx=0):

        style_vector = style_vectors[:, layer_idx, :]   # (256, 32) presumably

        
        # Normalize
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        
        # Get scale and bias
        scale = self.style_to_scale(style_vector)
        bias = self.style_to_bias(style_vector)
        
        
        scale = scale.unsqueeze(1)
        bias = bias.unsqueeze(1)
        
        
        return x_normalized * scale + bias


class AdaINTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=256, 
                 dropout=0.1, style_dim=32, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, 
                        batch_first=batch_first)
        
        # AdaIN for this layer
        self.ada_norm1 = AdaptiveLayerNorm(d_model, style_dim)
        self.ada_norm2 = AdaptiveLayerNorm(d_model, style_dim)
        
    def forward(self, src, style_vector, src_mask=None, 
                src_key_padding_mask=None, is_causal=False):
        """
        src: (batch, seq_len, d_model)
        style_vector: (batch, style_dim) - SINGLE style for this layer
        """
        x = src
        
        # Self-attention block with AdaIN
        if self.norm_first:
            # Use the SINGLE style vector
            x_norm = self.ada_norm1(x, style_vector)
            x = x + self._sa_block(x_norm, src_mask, src_key_padding_mask, is_causal)
            x_norm = self.ada_norm2(x, style_vector)
            x = x + self._ff_block(x_norm)
        else:
            # Post-norm
            x = self.ada_norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal), 
                              style_vector)
            x = self.ada_norm2(x + self._ff_block(x), style_vector)
        
        return x


class NoisyTransformerEncoder(nn.Module):
    """Transformer that properly handles per-layer styles"""
    def __init__(self, d_model=128, nhead=2, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, style_dim=32):
        super().__init__()
        self.num_layers = num_layers
        
        # Create layers
        self.layers = nn.ModuleList([
            AdaINTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                style_dim=style_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Noise injection
        self.noise_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, style_vectors, mask=None):
        """
        src: (batch, seq_len, d_model) = (256, 32, 128)
        style_vectors: (batch, style_dim) = (256, 32) - NOT per-layer!
        """
        output = src
        
        # Generate noise for each layer
        noise_tensors = []
        if self.training:
            for i in range(self.num_layers):
                noise = torch.randn_like(output)
                noise_tensors.append(noise)
        
        # Process each layer with the SAME style
        for i, layer in enumerate(self.layers):
            # ALL layers use the SAME style_vectors (it's 2D!)
            output = layer(output, style_vectors, mask)  # Pass same style to all layers
            
            # Inject noise
            if self.training:
                output = output + self.noise_weights[i] * noise_tensors[i]
        
        output = self.norm(output)
        return output
    
class StyleTransformerEncoderLayer(nn.Module):
    def __init__(self, base_layer, embedding_dim, style_dim):
        super().__init__()
        self.layer = base_layer
        self.gamma = nn.Linear(style_dim, embedding_dim)
        self.beta  = nn.Linear(style_dim, embedding_dim)

    def forward(self, x, style, src_mask=None, 
                src_key_padding_mask=None, is_causal=False):
        x = self.layer(x, src_mask, src_key_padding_mask, is_causal)
        #print('style shape:', style.shape)
        if len(style.shape) == 3:
            g = self.gamma(style)
            b = self.beta(style)
        else:
            g = self.gamma(style).unsqueeze(1)
            b = self.beta(style).unsqueeze(1)
        # print('g shape:', g.shape)
        # print('b shape:', b.shape)
        # print('x shape:', x.shape)
        return g * x + b

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionPairing(nn.Module):
    def __init__(self, feature_dim=7, embed_dim=32, temp=0.5):
        super().__init__()
        self.temp = temp  # softmax temperature for pairing sharpness
        
        # Linear projections for attention
        self.q_proj = nn.Linear(feature_dim, embed_dim)
        self.k_proj = nn.Linear(feature_dim, embed_dim)
        self.v_proj = nn.Linear(feature_dim, embed_dim)

        # Optional output projection (not strictly required for matching)
        self.out_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):
        """
        x: (B, 32, 7)  → 32 particles, 7 features
        returns: (B, 16, 2, 7)  → 16 learned pairs, each with 2 particles
        """
        B, P, Fdim = x.shape  # P = 32
        
        # Project to attention space
        Q = self.q_proj(x)  # (B, 32, embed_dim)
        K = self.k_proj(x)  # (B, 32, embed_dim)
        V = self.v_proj(x)  # (B, 32, embed_dim)

        # Compute similarity / attention scores for matching
        # (B, 32, 32) — symmetric learned pairing logits
        scores = torch.einsum('b i d, b j d -> b i j', Q, K)

        # GAN safety: clamp logits to finite stable region
        scores = scores.clamp(min=-10.0, max=10.0)

        # Turn into soft pairing weights (divide by temperature to control sharpness)
        attn = F.softmax(scores / self.temp, dim=-1)  # (B, 32, 32)

        # Each particle gets a weighted "partner-aware" feature vector
        partner_repr = torch.bmm(attn, V)  # (B, 32, embed_dim)
        partner_repr = self.out_proj(partner_repr)  # (B, 32, 7)

        # Now form 16 pairs by taking top-2 partner weights per particle *softly*
        # We do not extract indices, we construct soft pairs via weighted grouping.

        # Compute "best 2 partner contributions" per particle (vectorized)
        top2_vals, _ = scores.topk(2, dim=-1)  # values only for sharpening, no gradients through idx
        sharpen_scores = scores.clone()
        threshold = top2_vals[:, :, -1].unsqueeze(-1)  # 2nd highest score
        pair_mask = (scores >= threshold).float()      # soft mask for top2 partners
        pair_mask = pair_mask / (pair_mask.sum(dim=-1, keepdim=True) + 1e-6)  # normalize

        # Build soft pair representations
        pairs = torch.bmm(pair_mask, x)  # (B, 32, 7) aggregated 2-body info

        # Reshape into (B, 16, 2, 7)
        # We merge particles into learned pairs by grouping every 2 top-partner contributions
        pairs = pairs.view(B, P//2, 2, 7)

        return pairs

