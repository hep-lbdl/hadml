import torch
from torch.optim import Optimizer
from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule
from utils.utils import get_r1_grad_penalty
from metrics.media_logger import log_images
from metrics.image_converter import fig_to_array
from collections import Counter
import numpy as np, matplotlib.pyplot as plt
from torch.nn.attention import SDPBackend, sdpa_kernel
import ot, os, pickle


class MultiHadronEventGANModule(LightningModule):
    def __init__(
        self,
        datamodule: torch.nn.Module,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: Optimizer,
        optimizer_discriminator: Optimizer,
        noise_dim: int,
        loss_type: str,
        r1_reg: float,
        target_gumbel_temp: float = 0.3,
        gumbel_softmax_hard: bool = False,
        deviation_coeff: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator
        self.train_gen_loss = MeanMetric()
        self.train_disc_loss = MeanMetric()
        self.val_gen_loss = MeanMetric()
        self.val_disc_loss = MeanMetric()
        self.current_gumbel_temp = 1.0
        self.val_swd_token = MeanMetric()
        self.val_swd_sentence = MeanMetric()
        self.val_swd_hadron_multiplicity = MeanMetric()
        self.hadron_kins_dim = self.generator.hadron_kins_dim
        self.hadron_stats = None
        self.datamodule = datamodule
        self.training_stats_filename = datamodule.training_stats_filename
        with open(self.training_stats_filename, "rb") as f:
            stats = np.load(f, allow_pickle=True).item()
        self.hadron_stats = {
            "momentum_mean" : stats["hadron_momentum_mean"], "momentum_std" : stats["hadron_momentum_std"],
            "energy_mean" : stats["hadron_energy_mean"], "energy_std" : stats["hadron_energy_std"], 
        }
        # Register stats as buffers so Lightning keeps them on the module device.
        self.register_buffer("hadron_momentum_mean", torch.as_tensor(stats["hadron_momentum_mean"]))
        self.register_buffer("hadron_momentum_std", torch.as_tensor(stats["hadron_momentum_std"]))
        self.register_buffer("hadron_energy_mean", torch.as_tensor(stats["hadron_energy_mean"]))
        self.register_buffer("hadron_energy_std", torch.as_tensor(stats["hadron_energy_std"]))

    def _sanitise_tensor(self, x, name="tensor", clamp_val=50.0):
        if not torch.isfinite(x).all():
            import logging
            logging.warning(
                f"Sanitising {name} (contains NaN/Inf). Max abs was:" + \
                f" {x.abs().max().item() if not torch.isnan(x).all() else 'NaN'}")
            x = torch.nan_to_num(x, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        if x.abs().max() > clamp_val:
            import logging
            logging.warning(f"Clipping {name} (values > {clamp_val}). " + \
                            f"Max abs was: {x.abs().max().item()}")
            x = torch.clamp(x, min=-clamp_val, max=clamp_val)
        return x

    def forward(self, clusters):
        noise = self._generate_noise(*clusters.size()[:2])
        generated_hadrons = self.generator(noise.to(clusters.device), clusters)
        
        raw_kinematics = generated_hadrons[:, :, :self.hadron_kins_dim]
        raw_types_logits = generated_hadrons[:, :, self.hadron_kins_dim:]
        
        clean_kinematics = self._sanitise_tensor(
            raw_kinematics, "generated_kinematics", 50.0)
        clean_types_logits = self._sanitise_tensor(
            raw_types_logits, "generated_logits", 10.0)
        
        orig_dtype = clean_types_logits.dtype
        safe_tau = max(self.current_gumbel_temp, 1e-4) # Prevent division by zero
        
        sampled_types = torch.nn.functional.gumbel_softmax(
            clean_types_logits.float(), # Force fp32 execution
            tau=safe_tau, 
            hard=self.hparams.gumbel_softmax_hard
        ).to(orig_dtype) # Return to original dtype
        
        fake_hadrons = torch.cat([clean_kinematics, sampled_types], dim=-1)
        
        is_padding = (fake_hadrons[:, :, self.hadron_kins_dim] == 1.0).unsqueeze(-1)
        pure_padding_tokens = torch.zeros_like(fake_hadrons)
        pure_padding_tokens[:, :, self.hadron_kins_dim] = 1.0
        fake_hadrons = torch.where(is_padding, pure_padding_tokens, fake_hadrons)
        
        return fake_hadrons

    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Updating the Gumbel Softmax temperature
        self._update_gumbel_temp()
        
        gen_input, real_hadrons = batch

        if optimizer_idx == 0:
            # Generator turn
            fake_hadrons = self(gen_input)
            score_for_fake = self.discriminator(fake_hadrons)
            score_for_fake = self._sanitise_tensor(score_for_fake, "score_for_fake", 50.0)

            generator_loss = self._generator_loss(score_for_fake)

            if self.hparams.deviation_coeff > 0:
                valid_mask = (fake_hadrons[:, :, self.hadron_kins_dim:self.hadron_kins_dim+1] == 0.0).float()

                energy_std = self.hadron_energy_std.to(dtype=fake_hadrons.dtype)
                energy_mean = self.hadron_energy_mean.to(dtype=fake_hadrons.dtype)
                momentum_std = self.hadron_momentum_std.to(dtype=fake_hadrons.dtype)
                momentum_mean = self.hadron_momentum_mean.to(dtype=fake_hadrons.dtype)

                destandardised_energy = fake_hadrons[:, :, 0:1] * \
                    energy_std + energy_mean
                destandardised_momentum = fake_hadrons[:, :, 1:4] * \
                    momentum_std + momentum_mean
                
                destandardised_kin = torch.cat([destandardised_energy, destandardised_momentum], dim=2)
                destandardised_kin = destandardised_kin * valid_mask
                
                actual_momentum_sum = destandardised_kin.sum(axis=1)
                expected_momentum_sum = torch.zeros_like(actual_momentum_sum)
                expected_momentum_sum[:, 0] = 1.0  # Target energy = 1.0, Target px, py, pz = 0.0
                deviation = (expected_momentum_sum - actual_momentum_sum).abs().sum(axis=1).mean()
                self.log("deviation_from_conservation_law", deviation, prog_bar=True)

                generator_loss += self.hparams.deviation_coeff * deviation

            self.train_gen_loss(generator_loss)
            self.log("generator_loss", generator_loss, prog_bar=True)
            loss = generator_loss
        
        else:
            # Discriminator turn
            fake_hadrons = self(gen_input).detach()
            score_for_fake = self.discriminator(fake_hadrons)
            score_for_fake = self._sanitise_tensor(score_for_fake, "score_for_fake", 50.0)
            score_for_real = self.discriminator(real_hadrons)
            score_for_real = self._sanitise_tensor(score_for_real, "score_for_real", 50.0)

            discriminator_loss = self._discriminator_loss(score_for_real, score_for_fake)
            self.log("discriminator_loss", discriminator_loss, prog_bar=True)
        
            # Computing the R1 gradient penalty
            r1_grad_penalty = 0.0
            if self.hparams.r1_reg > 0:
                # Force FP32 context for the second-order derivative stability
                with torch.cuda.amp.autocast(enabled=False):
                    with sdpa_kernel(SDPBackend.MATH):
                        # Ensure inputs are explicitly float32
                        real_hadrons_fp32 = real_hadrons.detach().float().requires_grad_(True)                
                        r1_grad_penalty = (
                            get_r1_grad_penalty(self.discriminator, [real_hadrons_fp32]) * self.hparams.r1_reg
                        )
                
                self.log("r1_grad_penalty", r1_grad_penalty)
            
            loss = discriminator_loss + r1_grad_penalty

        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        import logging
                
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
            logging.warning("NaNs or Infs found in gradients. Sanitising to valid numbers.")
            for p in self.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            
        clip_val = getattr(self.hparams, 'gradient_clip_val', 50.0) 
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)
        if total_norm > clip_val:
            logging.warning(f"Gradient norm {total_norm:.4f} exceeded clip value {clip_val}. Clipped.")

    def _generator_loss(self, score):
        loss_type = self.hparams.loss_type
        if loss_type == "wasserstein":
            loss_gen = -score.mean(0).view(1)
        elif loss_type == "bce":
            loss_gen = torch.nn.functional.binary_cross_entropy_with_logits(
                score, torch.ones_like(score))
        elif loss_type == "ls":
            loss_gen = 0.5 * ((score - 1) ** 2).mean(0).view(1)
        return loss_gen

    def _discriminator_loss(self, score_for_real, score_for_fake):
        loss_type = self.hparams.loss_type
        if loss_type == "wasserstein":
            loss_disc = score_for_fake.mean(0).view(1) - score_for_real.mean(0).view(1)
        elif loss_type == "bce":
            loss_disc = torch.nn.functional.binary_cross_entropy_with_logits(
                score_for_real, torch.ones_like(score_for_real)) + \
                torch.nn.functional.binary_cross_entropy_with_logits(
                    score_for_fake, torch.zeros_like(score_for_fake))
        elif loss_type == "ls":
            loss_disc = 0.5 * ((score_for_real - 1)**2).mean(0).view(1) + \
                0.5 * (score_for_fake**2).mean(0).view(1)
        return loss_disc
    
    def validation_step(self, batch, batch_idx):
        gen_input, real_hadrons = batch
        if self.trainer.state.stage == "validate":
            fake_hadrons = self(gen_input)
            swd_shape = fake_hadrons.shape
            
            # Wasserstein (reshaping: [batch_size * seq_len, features])
            swd_token = ot.sliced_wasserstein_distance(
                fake_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]), 
                real_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]),
                n_projections=10*swd_shape[2])
            self.val_swd_token(swd_token)

            # Wasserstein (reshaping: [batch_size, seq_len * features])
            swd_sentence = ot.sliced_wasserstein_distance(
            fake_hadrons.cpu().detach().numpy().reshape(swd_shape[0], swd_shape[1] * swd_shape[2]), 
            real_hadrons.cpu().detach().numpy().reshape(swd_shape[0], swd_shape[1] * swd_shape[2]),
            n_projections=10*swd_shape[2])
            self.val_swd_sentence(swd_sentence)
            
            # Wassestein (comparing the number of non-padding tokens in the whole batch)
            counter_for_fake_hadrons = np.zeros((1, swd_shape[1] + 1))
            num_non_padding_tokens = [len(sequence[sequence[:, self.hadron_kins_dim] == 0.0]) 
                                      for sequence in fake_hadrons]
            for n in num_non_padding_tokens:
                counter_for_fake_hadrons[0, n] += 1

            counter_for_real_hadrons = np.zeros((1, swd_shape[1] + 1))
            num_non_padding_tokens = [len(sequence[sequence[:, self.hadron_kins_dim] == 0.0]) 
                                      for sequence in real_hadrons]
            for n in num_non_padding_tokens:
                counter_for_real_hadrons[0, n] += 1
            swd_hadron_multiplicity = ot.sliced_wasserstein_distance(
                counter_for_fake_hadrons,
                counter_for_real_hadrons,
                n_projections=10*(swd_shape[1]+1)
            )

            self.val_swd_hadron_multiplicity(swd_hadron_multiplicity)            

            return {"gen_output": fake_hadrons.cpu().detach(), 
                    "disc_input": real_hadrons.cpu().detach(),
                    "swd_token": swd_token, "swd_sentence": swd_sentence, 
                    "swd_hadron_multiplicity": swd_hadron_multiplicity,
                    "clusters": gen_input[:, 0, :].cpu().detach()}
        
        elif self.trainer.state.stage == "sanity_check":
            return {"gen_input": gen_input[:, 0, :].cpu().detach(),
                    "disc_input": real_hadrons.cpu().detach()}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        generator_opt = self.hparams.optimizer_generator(params=self.generator.parameters())
        disriminator_opt = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())
        return generator_opt, disriminator_opt
    
    def _generate_noise(self, batch_size, n_tokens):
        return torch.randn(batch_size, n_tokens, self.hparams.noise_dim)
    
    def _update_gumbel_temp(self):
        progress = self.trainer.global_step / self.trainer.max_steps
        progress = 1 - (1 - progress)**2
        self.current_gumbel_temp = 1.0 - (1 - self.hparams.target_gumbel_temp) * progress
        self.log("gumbel", self.current_gumbel_temp)

    def _compare(self, predictions, truths):
        images = self._prepare_plots(predictions.cpu().detach(), truths.cpu().detach())
        # Attributes self.logger and self.logger.experiment are defined by the logger passed
        # to the trainer which in turn uses an object of this model class: 
        if self.logger and self.logger.experiment is not None:
            log_images(logger=self.logger, key="MultiHadronEvent GAN",
                       images=list(images.values()), caption=list(images.keys()))

    def validation_epoch_end(self, validation_step_outputs):
        truths_batches = [d["disc_input"] for d in validation_step_outputs]
        truths_events = [event for batch in truths_batches for event in batch]

        if self.trainer.state.stage == "validate":
            sentence_stats = {}
            
            # Extract and flatten predictions
            preds_batches = [d["gen_output"] for d in validation_step_outputs]
            preds_events = [event for batch in preds_batches for event in batch]      

            # ==================================================================
            # ================= FOR SAVING PREDICTIONS AND TRUTHS ==============
            # ==================================================================
            # clusters = [d["clusters"] for d in validation_step_outputs]
            # save_dir = os.path.join(
            #     self.datamodule.data_dir,
            #     "plots",
            #     self.datamodule.raw_processed_filename.split(".")[0],
            #     "validation_batches",
            # )
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"{self.trainer.global_step}.pt")
            # torch.save(
            #     {
            #         "preds_batches": preds_batches,
            #         "truths_batches": truths_batches,
            #         "clusters": clusters,
            #     },
            #     save_path,
            # )
            # ==================================================================
            
            # Compute multiplicity stats safely using the unstacked event lists
            if not self.hparams.gumbel_softmax_hard:
                threshold = 0.5
                sentence_stats["pred_n_hads_per_cluster"] = \
                    [len(d[d[:, self.hadron_kins_dim] <= threshold]) for d in preds_events]
            else:
                sentence_stats["pred_n_hads_per_cluster"] = \
                    [len(d[d[:, self.hadron_kins_dim] == 0.0]) for d in preds_events]
            
            sentence_stats["pred_n_pad_hads_per_cluster"] = [len(preds_events[0]) - n for n in 
                                                                sentence_stats["pred_n_hads_per_cluster"]]
            
            sentence_stats["true_n_hads_per_cluster"] = \
                [len(d[d[:, self.hadron_kins_dim] == 0.0]) for d in truths_events]
            sentence_stats["true_n_pad_hads_per_cluster"] = [len(truths_events[0]) - n for n in 
                                                             sentence_stats["true_n_hads_per_cluster"]]
            
            # Stack into massive 2D matrices (total_hadrons, features) without double-flattening loops
            preds = torch.cat(preds_events, dim=0)         
            truths = torch.cat(truths_events, dim=0)         

            # Preparing diagrams
            images = self._prepare_plots(predictions=preds.cpu().detach(), truths=truths.cpu().detach(),
                                         sentence_stats=sentence_stats)

            # Computing the Wasserstein distance and sending it to the logger
            swd_token_distance = self.val_swd_token.compute()
            self.log("val/swd_token", swd_token_distance, sync_dist=True)
            self.val_swd_token.reset()

            swd_sentence_distance = self.val_swd_sentence.compute()
            self.log("val/swd_sentence", swd_sentence_distance, sync_dist=True)
            self.val_swd_sentence.reset()

            swd_hadron_multiplicity_distance = self.val_swd_hadron_multiplicity.compute()
            self.log("val/swd_hadron_multiplicity", swd_hadron_multiplicity_distance, sync_dist=True)
            self.val_swd_hadron_multiplicity.reset()

        elif self.trainer.state.stage == "sanity_check":
            gen_input = [d["gen_input"] for d in validation_step_outputs]
            gen_input = [d for gen_in in gen_input for d in gen_in] # [total_n_clusters, features]
            gen_input = torch.stack(gen_input)

            truths = torch.cat(truths_events, dim=0)
            images = self._prepare_plots(clusters=gen_input.cpu().detach(), truths=truths.cpu().detach())

        # Sending the diagrams to the logger
        if self.logger and self.logger.experiment is not None:
            log_images(
                self.logger,
                "MultiHadronEvent GAN",
                images=list(images.values()),
                caption=list(images.keys()),
            )

    def _prepare_plots(self, predictions=None, truths=None, sentence_stats=None, clusters=None):
        """ Prepare histograms and other charts using the data received from validation_epoch_end().
        Diagrams for the sanity check (clusters) are prepared once only before training. All the 
        other ones (hadrons) are drawn each time validation_epoch_end() is called. """
        diagrams = {}
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
        })

        if predictions is not None and truths is not None:
            predictions = predictions.clone()
            truths = truths.clone()

            # Getting rid of the padding token kinematics
            preds_kin = predictions[predictions[:, self.hadron_kins_dim] != 1.0][:, :self.hadron_kins_dim]
            preds_types = torch.argmax(predictions[:, self.hadron_kins_dim:], dim=1) - 1
            truths_kin = truths[truths[:, self.hadron_kins_dim] != 1.0][:, :self.hadron_kins_dim]
            truths_types = torch.argmax(truths[:, self.hadron_kins_dim:], dim=1) - 1

            # Destandardising the kinematics of the hadrons
            # Out-of-place calculation using standard operations
            m_mean, m_std = self.hadron_stats["momentum_mean"], self.hadron_stats["momentum_std"]
            e_mean, e_std = self.hadron_stats["energy_mean"], self.hadron_stats["energy_std"]
            # Compute directly without assigning back to slices in-place
            truth_energy = truths_kin[:, 0] * e_std + e_mean
            truth_momenta = truths_kin[:, 1:4] * m_std + m_mean
            preds_energy = preds_kin[:, 0] * e_std + e_mean
            preds_momenta = preds_kin[:, 1:4] * m_std + m_mean

            # ==================================================================
            # ======================= Hadron type histogram ====================
            # ==================================================================
            sample_range = [0, truths_types.max()]
            bins = np.linspace(
                start=sample_range[0] - 0.5, 
                stop=sample_range[1] + 0.5, 
                num=sample_range[1] - sample_range[0] + 2, 
                retstep=0.5)[0]
            n_types = truths_types.max() + 1
            density = n_types // 25 if n_types // 25 > 0 else 1
            fig = plt.figure(figsize=(9, 6))
            plt.title("Hadron Type Distribution")
            plt.hist(truths_types, bins=bins, color="maroon", label="True", rwidth=0.7, density=True)
            plt.hist(preds_types, bins=bins, color="black", label="Generated", 
                     rwidth=0.5, density=True)
            plt.ylabel("Hadrons", labelpad=12)
            plt.xlabel("Hadron Most Common ID\n(mapped from PIDs)", labelpad=20)
            xticks = np.arange(start=sample_range[0] - 1, stop=sample_range[1] + 1, step=density)[1:]
            plt.xticks(xticks, rotation=90)
            plt.legend(loc="upper right")
            plt.tight_layout()
            diagrams["hadron_type_hist"] = fig_to_array(fig, tight_layout=False)
            
            # =========== Saving the hadron type histogram to a file ===========
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0], 
                "hadron_type_histograms"
            )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                f"{self.trainer.global_step + 1}.pdf"
            )
            plt.savefig(filepath)

            # ==================================================================
            # ============= Hadron energy and momentum histogram ===============
            # ==================================================================
            fig, axs = plt.subplots(2, 2, figsize=(12, 9))
            fig.subplots_adjust(wspace=0.35, hspace=0.35)        
            labels = ["Generated", "True"]

            bins = np.histogram_bin_edges(truth_energy, bins="auto")
            records, bins, _ = axs[0][0].hist(
                truth_energy, bins=bins, 
                color="maroon", 
                label="True",
                density=True
            )
            axs[0][0].hist(
                preds_energy, 
                bins=bins, 
                color="black", 
                alpha=0.7, 
                label="Generated",
                density=True
            )
            axs[0][0].set_xlim(truth_energy.min().item(), 
                               truth_energy.max().item())
            axs[0][0].set_ylim(0, records.max() * 1.15)
            axs[0][0].set_xlabel("Energy [GeV]", labelpad=15)

            # Flattening the axis labels for a 1D mapping
            axis_names = ['x', 'y', 'z']
            # Flattening the 2x2 grid to a 1D array of 4 subplots
            axs_flat = axs.flatten() 
            # Looping through the 3 momentum columns (indices 0, 1, 2)
            for feature in range(3):
                # Energy is at index 0, so momentum plots occupy indices 1, 2, and 3
                ax = axs_flat[feature + 1] 
                ax.set_xlabel(f"Momentum ({axis_names[feature].upper()})", labelpad=15)                
                (records, bins, _) = ax.hist(
                    truth_momenta[:, feature], bins="auto", rwidth=0.9, color="maroon", 
                    label=labels[1], density=True
                )
                ax.hist(
                    preds_momenta[:, feature], bins=bins, color="black", 
                    rwidth=0.8, density=True,
                    label=labels[0], alpha=0.7
                )
                
                max_y_value = max(records)
                ax.set_ylim((0, max_y_value + max_y_value * 0.15))
                mean_val = truth_momenta[:, feature].mean().item()
                std_val = truth_momenta[:, feature].std().item()
                ax.set_xlim((mean_val - 3 * std_val, mean_val + 3 * std_val))
            
            # Global adjustments for all 4 subplots
            for ax in axs_flat:
                ax.set_ylabel("Hadrons", labelpad=12)
                ax.legend(loc='upper right')
                
            fig.suptitle("Hadron Kinematics Distribution (Laboratory Frame).\n" + \
                         "\"True\" defines the scale and limits.")
            diagrams["hadron_kinematics_hist"] = fig_to_array(fig, tight_layout=False)

            # === Saving the hadron energy and momentum histogram to a file ====
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0], 
                "hadron_energy_momentum_histograms"
            )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                f"{self.trainer.global_step + 1}.pdf"
            )
            plt.savefig(filepath)

            # ==================================================================
            # =============== Hadron and padding token multiplicity ============
            # ==================================================================
            n_max_hads = sentence_stats["true_n_hads_per_cluster"][0] + \
                            sentence_stats["true_n_pad_hads_per_cluster"][0]
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            bins = np.linspace(start=-0.5, stop=n_max_hads+0.5, num=n_max_hads+2, 
                               retstep=0.5)[0]
            datatype = ["true", "pred"]
            labels = ["True", "Generated"]
            colours = ["maroon", "black"]
            rwidth = [0.9, 0.8]
            for col in range(0, 2):
                for i in range(0, 2):
                    if col == 0:
                        axs[col].hist(
                            sentence_stats[f"{datatype[i]}_n_hads_per_cluster"], 
                            bins=bins,
                            color=colours[i], 
                            label=labels[i], 
                            rwidth=rwidth[i],
                            density=True
                        )
                        axs[col].set_xlabel("Number of hadrons", labelpad=15)
                    else:
                        axs[col].hist(
                            sentence_stats[f"{datatype[i]}_n_pad_hads_per_cluster"], 
                            bins=bins, color=colours[i], label=labels[i], 
                            rwidth=rwidth[i], density=True
                        )
                        axs[col].set_xlabel("Number of padding tokens", labelpad=15)
                axs[col].legend(loc="upper right")
                axs[col].set_ylabel("Sentences", labelpad=12)
            fig.suptitle("Sentence Statistics")
            plt.tight_layout()
            diagrams["sentence_statistics_hist"] = fig_to_array(
                fig, tight_layout=False)
            
            # ============= Saving sentence statistics histograms ==============
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0], 
                "sentence_statistics_histograms"
            )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                f"{self.trainer.global_step + 1}.pdf"
            )
            plt.savefig(filepath)

        elif clusters is not None and truths is not None:
            # Initial plots when there is no training yet (sanity check)
            diagrams = {}
            kinematics = clusters[:, :4]
            quark_types = clusters[:, 4:6]
            quark_angles = clusters[:, 6:]
            hadron_types = truths[:, self.hadron_kins_dim:]
            
            # ==================================================================
            # ====================== Cluster statistics ========================
            # ==================================================================
            fig, axs = plt.subplots(1, 3, figsize=(15, 6))
            axis = ['x', 'y', 'z']
            for col in range(0, 3):
                axs[col].hist(kinematics[:, col], bins="auto", color="black", 
                              rwidth=0.9)
                axs[col].set_xlabel(f"Momentum ({axis[col].capitalize()})", 
                                    labelpad=15)
                axs[col].set_ylabel("Clusters", labelpad=12)
            fig.suptitle("Cluster Momentum Distribution" + \
                         f"\n(validation data, {len(kinematics[:, col])} clusters)")
            plt.tight_layout()
            diagrams["cluster_kinematics_hist"] = fig_to_array(
                fig, tight_layout=False)

            # ============= Saving cluster kinematics histograms ==============
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0]
            )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                "cluster_kinematics_histograms.pdf"
            )
            plt.savefig(filepath)

            # Quark types and angles
            quark_types = torch.where(quark_types <= 8, -quark_types, quark_types - 8)
            count = Counter(quark_types.flatten().tolist())
            quark_pids = list(map(lambda x: x[0], count.most_common()))
            pids_to_idx = {pids: i for i, pids in enumerate(quark_pids)}
            n_idx = len(pids_to_idx)
            bins = np.linspace(start=-0.5, stop=n_idx+0.5, num=n_idx+2, retstep=0.5)[0]
            fig, axs = plt.subplots(2, 2, figsize=(9, 9))
            angles = ["phi", "theta"]
            for row in range(0, 2):
                for col in range(0, 2):
                    if row == 0:
                        quark_idx = [pids_to_idx[t.item()] for t in quark_types[:, col]]
                        axs[row][col].hist(quark_idx, bins=bins, rwidth=0.8, color="black")
                        axs[row][col].set_xlabel("Particle ID (PID)", labelpad=15)
                        axs[row][col].title.set_text("Type") 
                        axs[row][col].set_xticks([int(id) for id in pids_to_idx.values()])
                        axs[row][col].set_xticklabels([int(pid) for pid in pids_to_idx.keys()], 
                                                      rotation=90)
                    else:
                        axs[row][col].hist(quark_angles[:, col], bins="scott", rwidth=0.8, color="black")
                        axs[row][col].set_xlabel("Angle", labelpad=15)
                        axs[row][col].title.set_text(f"Kinematics ({angles[col]})") 
                    axs[row][col].set_ylabel("Quarks", labelpad=12)
            fig.suptitle("Quark Type and Momentum Distribution" + \
                         f"\n(validation data, {len(quark_types[:, 0])} quark pairs)")
            plt.tight_layout()
            diagrams["quarks_features_hist"] = fig_to_array(fig, tight_layout=False) 

            # =============== Saving quark features histograms =================
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0]
            )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                "quark_features_histograms.pdf"
            )
            plt.savefig(filepath)
    
            # ==================================================================
            # ==================== Hadron type histogram =======================
            # ==================================================================
            hadron_types = torch.argmax(hadron_types, dim=1)
            hadron_types = hadron_types[hadron_types != 0] - 1
            with open(os.path.join(os.path.normpath(self.hparams.datamodule.data_dir),
                                   "processed", self.hparams.datamodule.pid_map_file), "rb") as f:
                pids_to_idx = pickle.load(f)
            fig = plt.figure(figsize=(11.2, 6.3))
            n_idx = len(pids_to_idx)
            bins = np.linspace(start=-0.5, stop=n_idx+0.5, num=n_idx+2, retstep=0.5)[0]
            plt.hist(hadron_types, bins=bins, color="black", rwidth=0.8)
            x_ticks = [int(id) for id in pids_to_idx.values()]
            x_labels = [pid for pid in pids_to_idx.keys()]
            if "uncommon_pid" in pids_to_idx:
                plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
            else:
                plt.xticks(ticks=[x for x in x_ticks[::5 if len(x_ticks) > 50 else 1]], 
                           labels=[x for x in x_labels[::5 if len(x_labels) > 50 else 1]], 
                           rotation=90)
            plt.title(f"Hadron Type Distribution\n(validation data, {len(hadron_types)} hadrons, " + \
                      f"{len(pids_to_idx)} types)")
            plt.xlabel("Particle ID (PID)", labelpad=15)
            plt.ylabel("Hadrons", labelpad=12)
            plt.tight_layout()
            diagrams["hadron_initial_type_hist"] = fig_to_array(fig, tight_layout=False)    
            
            # ================= Saving hadron type histogram ===================
            dirname = os.path.join(
                self.datamodule.data_dir, "plots",
                self.datamodule.raw_processed_filename.split(".")[0]
        )
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(
                dirname,
                "hadron_initial_type_histograms.pdf"
            )
            plt.savefig(filepath)

        plt.close('all')
        return diagrams