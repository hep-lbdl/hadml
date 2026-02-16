import torch
from torch.optim import Optimizer
from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule
from utils.utils import get_r1_grad_penalty, lorentz_to_kt_eta_phi_m, get_r1_grad_penalty_2
from metrics.media_logger import log_images
from metrics.image_converter import fig_to_array
from collections import Counter
import numpy as np, matplotlib.pyplot as plt
from torch.nn.attention import SDPBackend, sdpa_kernel
import ot, os, pickle
import sys
torch.autograd.set_detect_anomaly(True)
from hadml.rambo.additional_helpers import get_invariant_mass, lorentz_boost, vectorized_boost


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
        deviation_coeff: float = 0.0,
        violation_loss_coeff: float = 0.0,
        non_pad_loss_coeff: float = 0.0,
        is_odd_loss: float = 0.0,
        entropy_coeff: float = 0.0,
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
        self.training_stats_filename = datamodule.training_stats_filename
        with open(self.training_stats_filename, "rb") as f:
            stats = np.load(f, allow_pickle=True).item()
        self.hadron_stats = {
            "momentum_mean" : stats["hadron_momentum_mean"], "momentum_std" : stats["hadron_momentum_std"],
            "energy_mean" : stats["hadron_energy_mean"], "energy_std" : stats["hadron_energy_std"], 
        }

    def forward(self, clusters):
        noise = self._generate_noise(*clusters.size()[:2])
        self.generator.hadron_kins_dim = self.hadron_kins_dim
        self.generator.current_gumbel_temp = self.current_gumbel_temp
        self.generator.gumbel_softmax_hard = self.hparams.gumbel_softmax_hard


        if self.trainer.state.stage == "validate":
            generated_hadrons = self.generator(noise.to(clusters.device), clusters)
        else:
            generated_hadrons = self.generator(noise.to(clusters.device), clusters)

        return generated_hadrons
    
    def setup(self, stage=None):
       # print(print(self.hparams))
        self.base_lr_g = self.hparams.optimizer_generator.keywords["lr"]
        self.base_lr_d = self.hparams.optimizer_discriminator.keywords["lr"]
        #pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Updating the Gumbel Softmax temperature
        self._update_gumbel_temp()
        
        gen_input, real_hadrons = batch
       # np.save(f"real_hadrons_batch_{batch_idx}.npy", real_hadrons.cpu().detach().numpy())

       # print(gen_input[:,0,:4])
       # sys.exit()
        fake_hadrons, violation, non_pad_loss, coverage_entropy, _ = self(gen_input)
        n_valid_fake_hadrons = len(fake_hadrons[violation == 0.0])
        fake_hadrons = fake_hadrons[violation == 0.0]
        
        if torch.isnan(fake_hadrons).any():
            raise ValueError("NaN detected in fake_hadrons")

        #score_for_fake, score_for_fake_rem = self.discriminator(fake_hadrons, rem_masses=non_pad_loss.mean(dim=1).unsqueeze(1))
        score_for_fake = self.discriminator(fake_hadrons)

        # Check for NaN values in score_for_fake
        if torch.isnan(score_for_fake).any():
            raise ValueError("NaN detected in score_for_fake")
        #print(violation)
        #print(len(violation[[violation==0.0]]))
        #sys.exit()

        
        violation_loss = self.hparams.violation_loss_coeff*violation.mean()
        current_epoch = self.trainer.current_epoch


        #print(violation_loss)
        #print(sys.exit())

    # Block optimizer 1 until epoch 20 is complete
        # if optimizer_idx == 1 and current_epoch < 50:
        #     return None
        
        # # block optimizer 1 every 2 steps
        # if optimizer_idx == 1 and current_epoch % 2 == 1:
        #     return None
       # real_hadrons_mass = get_invariant_mass(real_hadrons[:, :, :self.hadron_kins_dim])
        #real_hadrons_mass_sum = real_hadrons_mass.sum(dim=1).unsqueeze(1)  # [batch_size]

        if optimizer_idx == 0:
            # Training the generator
            
            # if violation_loss.item() < 0.1:
            non_pad_loss = self.hparams.non_pad_loss_coeff*non_pad_loss.mean()
            # else:
            #     non_pad_loss = 0.0

            # if violation_loss < 0.1:
            #     entropy_loss = -self.hparams.entropy_coeff * coverage_entropy
            # else:
            #     entropy_loss = 0.0
            entropy_loss = -self.hparams.entropy_coeff * coverage_entropy

            
            is_odd_loss = 0.0


            #is_odd_loss = self.hparams.is_odd_loss*is_odd.mean()
            # entropy_loss = - (pad_logit.sigmoid() * torch.log(pad_logit.sigmoid() + 1e-8) +
            #        (1 - pad_logit.sigmoid()) * torch.log(1 - pad_logit.sigmoid() + 1e-8))
            # # hard zeroes where the mask is False in a non-differentiable way
            # entropy_loss = entropy_loss * mask_all
            # entropy_loss = entropy_loss.mean()
            #entropy_loss = 0.0
            #entropy_loss = entropy_loss
          #  entropy_loss = self.hparams.entropy_coeff * entropy_loss
            generator_loss = self._generator_loss(score_for_fake) + violation_loss + non_pad_loss + is_odd_loss + entropy_loss
            # generator_loss = self._generator_loss(score_for_fake) + \
            # violation_loss + non_pad_loss + is_odd_loss + entropy_loss

            self.train_gen_loss(generator_loss)
            self.log("total_generator_loss", generator_loss, prog_bar=True,
                     on_step=False, on_epoch=True)
            if self.hparams.violation_loss_coeff > 0:
                self.log("violation_loss", violation_loss, prog_bar=True, on_step=False, on_epoch=True)
            if self.hparams.non_pad_loss_coeff > 0:
                self.log("non_pad_loss", non_pad_loss, prog_bar=True, on_step=False, on_epoch=True)
            if self.hparams.is_odd_loss > 0:
                self.log("is_odd_loss", is_odd_loss, prog_bar=True, on_step=False, on_epoch=True)
            if self.hparams.entropy_coeff > 0:
                self.log("entropy_loss", entropy_loss, prog_bar=True, on_step=False, on_epoch=True)
            

            self.log("generator_loss", generator_loss - entropy_loss -
                     violation_loss - non_pad_loss - is_odd_loss, prog_bar=True, on_step=False, on_epoch=True)
            loss = generator_loss

            return {"loss": loss}

        
        else:
            # Training the discriminator
            # invariant masses of all real hadrons in the batch
            n_valid_real_hadrons = len(real_hadrons)
            #score_for_real, score_for_real_rem = self.discriminator(real_hadrons, real_hadrons_mass_sum)
            if violation_loss.item() < 1.0:

                wr = 0.5/n_valid_real_hadrons
                wf = 0.5/(n_valid_fake_hadrons+1e-8)

                score_for_real = self.discriminator(real_hadrons)
                discriminator_loss = self._discriminator_loss(score_for_real, score_for_fake, wr, wf)
                #discriminator_loss_rem = self._discriminator_loss(score_for_real_rem, score_for_fake_rem)
                
                self.log("discriminator_loss", discriminator_loss, prog_bar=True, on_step=False, on_epoch=True)
            
                # Computing the R1 gradient penalty
                r1_grad_penalty = 0.0
                if self.hparams.r1_reg > 0:
                    with sdpa_kernel(SDPBackend.MATH):
                        r1_grad_penalty = (
                            get_r1_grad_penalty(self.discriminator, [real_hadrons]) * self.hparams.r1_reg)
                

                    self.log("r1_grad_penalty", r1_grad_penalty, on_step=False, on_epoch=True)
                #loss = discriminator_loss + r1_grad_penalty + self.hparams.violation_loss_coeff*discriminator_loss_rem
                
                loss = discriminator_loss + r1_grad_penalty
                
                return {"loss": loss}
            
            else:
                return None

            #else:
             #   return None
                


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
    
    # def check_gradients(self, model):
    #     for name, p in model.named_parameters():
    #         if p.grad is None:
    #             print(f"{name}: no grad")
    #         else:
    #             print(f"{name}: NaN={torch.isnan(p.grad).any()}, Inf={torch.isinf(p.grad).any()}")

    # def on_after_backward(self):
    #     self.check_gradients(self.discriminator)
    #    # sys.exit()


    def _discriminator_loss(self, score_for_real, score_for_fake, wr, wf):
        loss_type = self.hparams.loss_type
        if loss_type == "wasserstein":
            loss_disc = score_for_fake.mean(0).view(1) - score_for_real.mean(0).view(1)
        elif loss_type == "bce":



            loss_real = torch.nn.functional.binary_cross_entropy_with_logits(score_for_real,
                                                                                torch.ones_like(score_for_real),
                                                                                reduction='sum')
            loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(score_for_fake,
                                                                                torch.zeros_like(score_for_fake),
                                                                                reduction='sum')


            loss_disc = wr*loss_real + wf*loss_fake
            # loss_disc = torch.nn.functional.binary_cross_entropy_with_logits(
            #     score_for_real, torch.ones_like(score_for_real),
            #     reduction='sum') + \
            #     torch.nn.functional.binary_cross_entropy_with_logits(
            #         score_for_fake, torch.zeros_like(score_for_fake),
            #         reduction='sum')

            with torch.no_grad():
                pred_real = torch.sigmoid(score_for_real)
                pred_fake = torch.sigmoid(score_for_fake)
                labels = torch.cat([torch.ones_like(pred_real), torch.zeros_like(pred_fake)], dim=0)
                preds = torch.cat([pred_real, pred_fake], dim=0)
                preds_cls = (preds >= 0.5).float()

                accuracy = (preds_cls == labels).float().mean()
                self.log("disc_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

                
                

        elif loss_type == "ls":
            loss_disc = 0.5 * ((score_for_real - 1)**2).mean(0).view(1) + \
                0.5 * (score_for_fake**2).mean(0).view(1)
        return loss_disc
    
    def validation_step(self, batch, batch_idx):
        gen_input, real_hadrons = batch

        #print(gen_input.shape)
        #sys.exit()

        if self.trainer.state.stage == "validate":
            with torch.no_grad():
                #fake_hadrons, _, _, _ = self(gen_input)
                fake_hadrons, _, _, _, valid = self(gen_input)
                fake_hadrons = fake_hadrons[valid]
            swd_shape = fake_hadrons.shape
            
            # Wasserstein (reshaping: [batch_size * seq_len, features])
            # swd_token = ot.sliced_wasserstein_distance(
            #     fake_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]), 
            #     real_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]),
            #     n_projections=10*swd_shape[2])
            # self.val_swd_token(swd_token)

            # # Wasserstein (reshaping: [batch_size, seq_len * features])
            # swd_sentence = ot.sliced_wasserstein_distance(
            # fake_hadrons.cpu().detach().numpy().reshape(swd_shape[0], swd_shape[1] * swd_shape[2]), 
            # real_hadrons.cpu().detach().numpy().reshape(swd_shape[0], swd_shape[1] * swd_shape[2]),
            # n_projections=10*swd_shape[2])
            # self.val_swd_sentence(swd_sentence)
            
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
                    #"swd_token": swd_token, "swd_sentence": swd_sentence, 
                    "swd_hadron_multiplicity": swd_hadron_multiplicity}
        
        elif self.trainer.state.stage == "sanity_check":
            return {"gen_input": gen_input[:, 0, :].cpu().detach(),
                    "disc_input": real_hadrons.cpu().detach()}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        generator_opt = self.hparams.optimizer_generator(params=self.generator.parameters())
        disriminator_opt = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())



        return generator_opt, disriminator_opt

    # def automatic_optimization(self) -> bool:
    #     return False  
    
    def _generate_noise(self, batch_size, n_tokens):
        return torch.randn(batch_size, n_tokens, self.hparams.noise_dim)
    
    # def _update_gumbel_temp(self):
    #     progress = self.trainer.global_step / self.trainer.max_steps
    #     progress = 1 - (1 - progress)**2
    #     self.current_gumbel_temp = 1.0 - (1 - self.hparams.target_gumbel_temp) * progress
    #     self.log("gumbel", self.current_gumbel_temp)
       # print(print(f"progress: {progress}, max_steps: {self.trainer.max_steps}"))

    def _update_gumbel_temp(self):
        # Compute progress safely
        if hasattr(self.trainer, "max_steps") and self.trainer.max_steps > 0:
            progress = self.trainer.global_step / self.trainer.max_steps
        elif hasattr(self.trainer, "max_epochs") and self.trainer.max_epochs > 0:
            progress = self.trainer.current_epoch / self.trainer.max_epochs
        else:
            progress = self.trainer.global_step / 10000

        progress = max(0.0, min(1.0, progress))

        # GAN-friendly annealing: fast early, flat late
        # sqrt slows decay near the end
        progress = progress**0.5

        tau_start = 1.0
        tau_min   = self.hparams.target_gumbel_temp

        self.current_gumbel_temp = tau_start - (tau_start - tau_min) * progress

        self.log("gumbel_temp", self.current_gumbel_temp, prog_bar=True,
                 on_step=True, on_epoch=False)

    
    # # Optional: Debug logging (disable in production)
    # if self.trainer.global_step % 100 == 0:
    #     self.log_dict({
    #         "gumbel_progress": progress,
    #         "gumbel_target": target_temp,
    #     })


    def _compare(self, predictions, truths):
        images = self._prepare_plots(predictions, truths)
        # Attributes self.logger and self.logger.experiment are defined by the logger passed
        # to the trainer which in turn uses an object of this model class: 
        if self.logger and self.logger.experiment is not None:
            log_images(logger=self.logger, key="MultiHadronEvent GAN",
                       images=list(images.values()), caption=list(images.keys()))

    def validation_epoch_end(self, validation_step_outputs):


        truths = [d["disc_input"] for d in validation_step_outputs]
        truths_numpy = [d.cpu().detach().numpy() for d in truths]
        truths_numpy = np.concatenate(truths_numpy, axis=0)
        #np.save("truths.npy", truths_numpy)
        #print(truths_numpy.shape)
 
        # [n_hadron_sets, max_n_hadrons, features]
        truths = [d for truth in truths for d in truth]
        
        if self.trainer.state.stage == "validate":
            # Handling the validation output list
            # Shape of validation_step_outputs: [n_batches, dict_key, batch_size, max_n_hadrons, features]
            sentence_stats = {}
            
            preds = [d["gen_output"] for d in validation_step_outputs]
            preds_numpy = [d.cpu().detach().numpy() for d in preds]
            preds_numpy = np.concatenate(preds_numpy, axis=0)
           # np.save("preds.npy", preds_numpy)
            #np.save("preds.npy", preds)
            # [n_hadron_sets, max_n_hadrons, features]
            preds = [d for pred in preds for d in pred]      
            sentence_stats["pred_n_hads_per_cluster"] = \
                [len(d[d[:, self.hadron_kins_dim] == 0.0]) for d in preds]
            sentence_stats["pred_n_pad_hads_per_cluster"] = [len(preds[0]) - n for n in 
                                                             sentence_stats["pred_n_hads_per_cluster"]]
            # [total_n_hadrons, features]
            preds = [d for pred in preds for d in pred]      
            preds = torch.stack(preds) 

            #print(f"Preds: {preds}")
            #sys.exit()        
              
            sentence_stats["true_n_hads_per_cluster"] = \
                [len(d[d[:, self.hadron_kins_dim] == 0.0]) for d in truths]
            sentence_stats["true_n_pad_hads_per_cluster"] = [len(truths[0]) - n for n in 
                                                             sentence_stats["true_n_hads_per_cluster"]]
            # [total_n_hadrons, features]
            truths = [d for truth in truths for d in truth]  
            truths = torch.stack(truths)         

            # Preparing diagrams
            images = self._prepare_plots(predictions=preds.cpu(), truths=truths.cpu(),
                                         sentence_stats=sentence_stats)

            # Computing the Wasserstein distance and sending it to the logger
            # swd_token_distance = self.val_swd_token.compute()
            # self.log("val/swd_token", swd_token_distance, sync_dist=True, on_step=False, on_epoch=True)
            # self.val_swd_token.reset()

            # swd_sentence_distance = self.val_swd_sentence.compute()
            # self.log("val/swd_sentence", swd_sentence_distance, sync_dist=True, on_step=False, on_epoch=True)
            # self.val_swd_sentence.reset()

            swd_hadron_multiplicity_distance = self.val_swd_hadron_multiplicity.compute()
            self.log("val/swd_hadron_multiplicity", swd_hadron_multiplicity_distance, sync_dist=True, on_step=False, on_epoch=True)
            self.val_swd_hadron_multiplicity.reset()

        elif self.trainer.state.stage == "sanity_check":
            gen_input = [d["gen_input"] for d in validation_step_outputs]
            gen_input = [d for gen_in in gen_input for d in gen_in] # [total_n_clusters, features]
            gen_input = torch.stack(gen_input)

            # [total_n_hadrons, features]
            truths = [d for truth in truths for d in truth]  
            truths = torch.stack(truths)

            # Preparing diagrams
            images = self._prepare_plots(clusters=gen_input.cpu(), truths=truths.cpu())

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

        if predictions is not None and truths is not None:

            print(f"Predictions shape: {predictions.shape}")
            print('hadron kin dimension: ', self.hadron_kins_dim)
           # sys.exit()
            # Getting rid of the padding token kinematics
            preds_kin = predictions[predictions[:, self.hadron_kins_dim] != 1.0][:, :self.hadron_kins_dim]
            preds_types = torch.argmax(predictions[:, self.hadron_kins_dim:], dim=1) - 1
            truths_kin = truths[truths[:, self.hadron_kins_dim] != 1.0][:, :self.hadron_kins_dim]
            truths_types = torch.argmax(truths[:, self.hadron_kins_dim:], dim=1) - 1 

            print(f"Preds kinematics shape: {preds_kin.shape}")
            # sys.exit()
            preds_pteta_phi_m = lorentz_to_kt_eta_phi_m(preds_kin)
            truths_pteta_phi_m = lorentz_to_kt_eta_phi_m(truths_kin) 
            
            # plt.figure()
            # plt.hist(truths_pteta_phi_m[:, 3], bins=50, alpha=0.5, label='True Mass', color='red')
            # plt.yscale('log')
            # plt.savefig('true_mass_distribution.png')
            # plt.close()

            # plt.figure()
            # plt.hist(truths_pteta_phi_m[:, 0], bins=50, alpha=0.5, label='Generated pt', color='black')
            # plt.yscale('log')
            # plt.savefig('generated_pt_distribution.png')
            # plt.close()

            # plt.figure()
            # plt.hist(truths_pteta_phi_m[:, 1], bins=50, alpha=0.5, label='Generated eta', color='black')
            # plt.yscale('log')
            # plt.savefig('generated_eta_distribution.png')
            # plt.close()

            # plt.figure()
            # plt.hist(truths_pteta_phi_m[:, 2], bins=50, alpha=0.5, label='Generated phi', color='black')
            # plt.yscale('log')
            # plt.savefig('generated_phi_distribution.png')
            # plt.close()

            # Hadron type histogram
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
            plt.hist(truths_types, bins=bins, color="red", histtype="step", label="True", rwidth=0.9)
            plt.hist(preds_types, bins=bins, color="black", label="Generated", rwidth=0.8)
            plt.ylabel("Hadrons")
            plt.xlabel("Hadron Most Common ID\n(mapped from PIDs)", labelpad=20)
            xticks = np.arange(start=sample_range[0] - 1, stop=sample_range[1] + 1, step=density)[1:]
            plt.xticks(xticks, rotation=90)
           # plt.yscale("log")
            plt.legend(loc="upper right")
            plt.tight_layout()
            diagrams["hadron_type_hist"] = fig_to_array(fig, tight_layout=False)

            # Hadron energy and momentum histogram 
            fig, axs = plt.subplots(2, 2, figsize=(12, 9))
            fig.subplots_adjust(wspace=0.2, hspace=0.35)        
            axs[0][0].set_title("Hadron Energy Distribution")
            labels = ["Generated", "True"]
            (records, bins, _) = axs[0][0].hist(truths_kin[:, 0], bins="auto", color="red", 
                                        label=labels[1], alpha=0.7)
            min_x_value, max_x_value = min(bins), max(bins)
            axs[0][0].set_xlim((min_x_value, max_x_value))
            axs[0][0].hist(preds_kin[:, 0], bins=bins, color="black", label=labels[0], alpha=0.7)
            axs[0][0].set_xlabel("Energy")
            axis = ['E', 'x', 'y', 'z']
            for row in range(0, 2):
                for col in range(0, 2):
                    if row == 0 and col == 0:
                        continue
                    feature = 2*row + col
                    axs[row][col].set_xlabel(f"Momentum ({axis[feature].capitalize()})")
                    axs[row][col].title.set_text("Hadron Momentum Distribution")
                    (records, bins, _) = axs[row][col].hist(
                        truths_kin[:, feature], bins="auto", rwidth=0.9, color="red", 
                        label=labels[1], alpha=0.7)
                    axs[row][col].hist(
                        preds_kin[:, feature], bins=bins, color="black", rwidth=0.8,
                        label=labels[0], alpha=0.7)
                    min_x_value, max_x_value = min(bins), max(bins)
                    axs[row][col].set_xlim((min_x_value, max_x_value))
            for row in range(0, 2):
                for col in range(0, 2):
                    axs[row][col].set_ylabel("Hadrons")
                    axs[row][col].legend(loc='upper right')
            fig.suptitle("Hadron Kinematics Distribution (Cluster Rest Frame).\n" + \
                         "\"True\" defines the scale and limits.")
            diagrams["hadron_kinematics_hist"] = fig_to_array(fig, tight_layout=False)


            # Hadron energy and momentum histogram 
            fig, axs = plt.subplots(2, 2, figsize=(12, 9))
            fig.subplots_adjust(wspace=0.2, hspace=0.35)        
            axs[0][0].set_title("Hadron pt eta phi m")
            labels = ["Generated", "True"]
            (records, bins, _) = axs[0][0].hist(truths_pteta_phi_m[:, 0], bins="auto", color="red", 
                                        label=labels[1], alpha=0.7)
            min_x_value, max_x_value = min(bins), max(bins)
            axs[0][0].set_xlim((min_x_value, max_x_value))
            axs[0][0].hist(preds_pteta_phi_m[:, 0], bins=bins, color="black", label=labels[0], alpha=0.7)
            axs[0][0].set_xlabel("pt")
            axis = ['pt', 'eta', 'phi', 'm']
            for row in range(0, 2):
                for col in range(0, 2):
                    if row == 0 and col == 0:
                        continue
                    feature = 2*row + col
                    axs[row][col].set_xlabel(f"Momentum ({axis[feature].capitalize()})")
                    axs[row][col].title.set_text("Hadron Momentum Distribution")
                    (records, bins, _) = axs[row][col].hist(
                        truths_pteta_phi_m[:, feature], bins="auto", rwidth=0.9, color="red", 
                        label=labels[1], alpha=0.7)
                    axs[row][col].hist(
                        preds_pteta_phi_m[:, feature], bins=bins, color="black", rwidth=0.8,
                        label=labels[0], alpha=0.7)
                    min_x_value, max_x_value = min(bins), max(bins)
                    axs[row][col].set_xlim((min_x_value, max_x_value))
            for row in range(0, 2):
                for col in range(0, 2):
                    axs[row][col].set_ylabel("Hadrons")
                    axs[row][col].legend(loc='upper right')
            fig.suptitle("pt eta phi m Distribution")
            diagrams["pt_eta_phi_m"] = fig_to_array(fig, tight_layout=False)



            # Hadron and padding token multiplicity
            n_max_hads = sentence_stats["true_n_hads_per_cluster"][0] + \
                            sentence_stats["true_n_pad_hads_per_cluster"][0]
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            bins = np.linspace(start=-0.5, stop=n_max_hads+0.5, num=n_max_hads+2, retstep=0.5)[0]
            datatype = ["true", "pred"]
            labels = ["True", "Generated"]
            colours = ["red", "black"]
            rwidth = [0.9, 0.8]
            for col in range(0, 2):
                for i in range(0, 2):
                    if col == 0:
                        axs[col].hist(sentence_stats[f"{datatype[i]}_n_hads_per_cluster"], bins=bins,
                                      color=colours[i], label=labels[i], rwidth=rwidth[i])
                        axs[col].set_xlabel("Number of hadrons")
                    else:
                        axs[col].hist(sentence_stats[f"{datatype[i]}_n_pad_hads_per_cluster"], 
                                            bins=bins, color=colours[i], label=labels[i], rwidth=rwidth[i])
                        axs[col].set_xlabel("Number of padding tokens")
                axs[col].legend(loc="upper right")
                axs[col].set_ylabel("Sentences")
            fig.suptitle("Sentence Statistics")
            plt.tight_layout()
            diagrams["sentence_statistics_hist"] = fig_to_array(fig, tight_layout=False)
            # plt.show()

        elif clusters is not None and truths is not None:
            diagrams = {}
            kinematics = clusters[:, :4]
            quark_types = clusters[:, 4:6]
            quark_angles = clusters[:, 6:]
            hadron_types = truths[:, self.hadron_kins_dim:]

            # Cluster kinematics histograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 6))
            axis = ['x', 'y', 'z']
            for col in range(0, 3):
                axs[col].hist(kinematics[:, col], bins="auto", color="black", rwidth=0.9)
                axs[col].set_xlabel(f"Momentum ({axis[col].capitalize()})")
                axs[col].set_ylabel("Clusters")
            fig.suptitle("Cluster Momentum Distribution" + \
                         f"\n(validation data, {len(kinematics[:, col])} clusters)")
            plt.tight_layout()
            diagrams["cluster_kinematics_hist"] = fig_to_array(fig, tight_layout=False)

            # Quark types and angles
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
                        axs[row][col].set_xlabel("Particle ID (PID)")
                        axs[row][col].title.set_text("Type") 
                        axs[row][col].set_xticks([int(id) for id in pids_to_idx.values()])
                        axs[row][col].set_xticklabels([int(pid) for pid in pids_to_idx.keys()], 
                                                      rotation=90)
                    else:
                        axs[row][col].hist(quark_angles[:, col], bins="scott", rwidth=0.8, color="black")
                        axs[row][col].set_xlabel("Angle") 
                        axs[row][col].title.set_text(f"Kinematics ({angles[col]})") 
                    axs[row][col].set_ylabel("Quarks")
            fig.suptitle("Quark Type and Momentum Distribution" + \
                         f"\n(validation data, {len(quark_idx)} quark pairs)")
            plt.tight_layout()
            diagrams["quarks_features_hist"] = fig_to_array(fig, tight_layout=False)    
            
            # Hadron type histogram
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
                plt.xticks(ticks=[x for x in x_ticks[::5]], 
                           labels=[x for x in x_labels[::5]], rotation=90)
            plt.title(f"Hadron Type Distribution\n(validation data, {len(hadron_types)} hadrons, " + \
                      f"{len(pids_to_idx)} types)")
            plt.xlabel("Particle ID (PID)")
            plt.ylabel("Hadrons")
            plt.tight_layout()
            diagrams["hadron_initial_type_hist"] = fig_to_array(fig, tight_layout=False)    

        return diagrams