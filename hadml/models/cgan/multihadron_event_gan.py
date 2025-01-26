import torch
from torch.optim import Optimizer
from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule
from utils.utils import conditional_cat, get_r1_grad_penalty
from metrics.media_logger import log_images
from metrics.image_converter import fig_to_array
from collections import Counter
import numpy as np, matplotlib.pyplot as plt
from torch.nn.attention import SDPBackend, sdpa_kernel
import ot


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
        gumbel_softmax_hard: bool = False
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
        self.val_swd = MeanMetric()

    def forward(self, clusters):
        noise = self._generate_noise(*clusters.size()[:2])
        generated_hadrons = self.generator(noise.to(clusters.device), clusters)
        hadron_kins_dim = self.generator.hadron_kins_dim
        generated_hadrons[:, :, hadron_kins_dim:] = torch.nn.functional.gumbel_softmax(
            generated_hadrons[:, :, hadron_kins_dim:], self.current_gumbel_temp, 
            hard=self.hparams.gumbel_softmax_hard)
        return generated_hadrons
    
    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Updating the Gumbel Softmax temperature
        self._update_gumbel_temp()
        
        gen_input, real_hadrons = batch
        fake_hadrons = self(gen_input)
        score_for_fake = self.discriminator(fake_hadrons)
        if optimizer_idx == 0:
            # Training the generator
            generator_loss = self._generator_loss(score_for_fake)
            self.train_gen_loss(generator_loss)
            self.log("generator_loss", generator_loss, prog_bar=True)
            loss = generator_loss
        else:
            # Training the discriminator
            score_for_real = self.discriminator(real_hadrons)
            discriminator_loss = self._discriminator_loss(score_for_real, score_for_fake)
            self.log("discriminator_loss", discriminator_loss, prog_bar=True)
            # Computing the R1 gradient penalty
            r1_grad_penalty = 0.0
            if self.hparams.r1_reg > 0:
                with sdpa_kernel(SDPBackend.MATH):
                    r1_grad_penalty = (
                        get_r1_grad_penalty(self.discriminator, [real_hadrons]) * self.hparams.r1_reg)
                self.log("r1_grad_penalty", r1_grad_penalty)
            loss = discriminator_loss + r1_grad_penalty
        return {"loss": loss}

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
            swd = ot.sliced_wasserstein_distance(
                fake_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]), 
                real_hadrons.cpu().detach().numpy().reshape(swd_shape[0] * swd_shape[1], swd_shape[2]),
                n_projections=2*swd_shape[2])
            self.val_swd(swd)
            return {"gen_output": fake_hadrons.cpu().detach(), 
                    "disc_input": real_hadrons.cpu().detach(),
                    "swd": swd}
        elif self.trainer.state.stage == "sanity_check":
            return {"gen_input": gen_input[:, 0, :].cpu().detach()}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        generator_opt = self.hparams.optimizer_generator(params=self.generator.parameters())
        disriminator_opt = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())
        return generator_opt, disriminator_opt
    
    def _generate_noise(self, batch_size, n_tokens):
        return torch.randn(batch_size, n_tokens, self.hparams.noise_dim)
    
    def _update_gumbel_temp(self):
        progress = self.trainer.global_step / (0.8 * self.trainer.max_epochs * \
            len(self.hparams.datamodule.train_dataloader()))
        progress = 1 - (1 - progress)**2
        self.current_gumbel_temp = 1.0 - (1 - self.hparams.target_gumbel_temp) * progress
        self.log("gumbel", self.current_gumbel_temp)

    def _compare(self, predictions, truths):
        images = self._prepare_plots(predictions, truths)
        # Attributes self.logger and self.logger.experiment are defined by the logger passed
        # to the trainer which in turn uses an object of this model class: 
        if self.logger and self.logger.experiment is not None:
            log_images(logger=self.logger, key="MultiHadronEvent GAN",
                       images=list(images.values()), caption=list(images.keys()))

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.state.stage == "validate":
            # Handling the validation output list
            # Shape of validation_step_outputs: [n_batches, dict_key, batch_size, max_n_hadrons, features]
            sentence_stats = {}
            
            preds = [d["gen_output"] for d in validation_step_outputs]
            # [n_hadron_sets, max_n_hadrons, features]
            preds = [d for pred in preds for d in pred]      
            sentence_stats["pred_n_hads_per_cluster"] = [len(d[d[:, 4] == 0.0]) for d in preds]
            sentence_stats["pred_n_pad_hads_per_cluster"] = [len(preds[0]) - n for n in 
                                                             sentence_stats["pred_n_hads_per_cluster"]]
            # [total_n_hadrons, features]
            preds = [d for pred in preds for d in pred]      
            preds = torch.stack(preds)         
            
            truths = [d["disc_input"] for d in validation_step_outputs]
            # [n_hadron_sets, max_n_hadrons, features]
            truths = [d for truth in truths for d in truth]  
            sentence_stats["true_n_hads_per_cluster"] = [len(d[d[:, 4] == 0.0]) for d in truths]
            sentence_stats["true_n_pad_hads_per_cluster"] = [len(truths[0]) - n for n in 
                                                             sentence_stats["true_n_hads_per_cluster"]]
            # [total_n_hadrons, features]
            truths = [d for truth in truths for d in truth]  
            truths = torch.stack(truths)         

            # Preparing diagrams
            images = self._prepare_plots(predictions=preds.cpu(), truths=truths.cpu(),
                                         sentence_stats=sentence_stats)
            
            # Computing the Wasserstein distance and sending it to the logger
            swd_distance = self.val_swd.compute()
            self.log("val/swd", swd_distance)
            self.val_swd.reset()

        elif self.trainer.state.stage == "sanity_check":
            gen_input = [d["gen_input"] for d in validation_step_outputs]
            gen_input = [d for gen_in in gen_input for d in gen_in] # [total_n_clusters, features]
            gen_input = torch.stack(gen_input)

            # Preparing diagrams
            images = self._prepare_plots(clusters=gen_input.cpu())

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
            preds_kin, preds_types = predictions[:, :4], torch.argmax(predictions[:, 4:], dim=1) - 1
            truths_kin, truths_types = truths[:, :4], torch.argmax(truths[:, 4:], dim=1) - 1

            # Hadron type histogram
            sample_range = [1, preds_types.max()]
            bins = np.linspace(
                start=sample_range[0] - 0.5, 
                stop=sample_range[1] + 0.5, 
                num=sample_range[1] - sample_range[0] + 2, 
                retstep=0.5)[0]
            n_types = truths_types.max() + 1
            density = n_types // 25 if n_types // 25 > 0 else 1
            fig = plt.figure(figsize=(9, 6))
            plt.title("Hadron Type Distribution")
            plt.hist(truths_types, bins=bins, color="black", histtype="step", label="True")
            plt.hist(preds_types, bins=bins, color="#AABA9E", label="Generated")
            plt.ylabel("Hadrons")
            plt.xlabel("Hadron Most Common ID\n(mapped from PIDs)", labelpad=20)
            xticks = np.arange(start=sample_range[0] - 1, stop=sample_range[1] + 1, step=density)[1:]
            plt.xticks(xticks, rotation=90)
            plt.legend()
            plt.tight_layout()
            diagrams["hadron_type_hist"] = fig_to_array(fig, tight_layout=False)

            # Hadron energy and momentum histogram 
            fig, axs = plt.subplots(2, 2, figsize=(12, 9))
            fig.subplots_adjust(wspace=0.2, hspace=0.35)        
            axs[0][0].set_title("Hadron Energy Distribution")
            labels = ["Generated", "True"]
            (_, bins, _) = axs[0][0].hist(truths_kin[:, 0], bins="scott", color="black", 
                                        label=labels[1], histtype="step")
            axs[0][0].hist(preds_kin[:, 0], bins=bins, color="#AEC5EB", label=labels[0])
            axs[0][0].set_xlabel("Energy (Cluster Rest Frame)")
            axis = ['x', 'y', 'z']
            for row in range(0, 2):
                for col in range(0, 2):
                    if row == 0 and col == 0:
                        continue
                    feature = row + col + 1
                    axs[row][col].set_xlabel(f"Momentum ({axis[feature - 1].capitalize()})")
                    axs[row][col].title.set_text("Hadron Momentum Distribution")
                    (_, bins, _) = axs[row][col].hist(truths_kin[:, feature], bins="auto", 
                                                    color="black", label=labels[1], histtype="step")
                    axs[row][col].hist(preds_kin[:, feature], bins=bins, color="#F9DEC9", 
                                    label=labels[0])
            for row in range(0, 2):
                for col in range(0, 2):
                    axs[row][col].set_ylabel("Hadrons (Log Scale)")
                    axs[row][col].set_yscale("log")
                    axs[row][col].legend(loc='upper right')
            fig.suptitle("Hadron Kinematics Distribution")
            diagrams["hadron_kinematics_hist"] = fig_to_array(fig, tight_layout=False)

            # Hadron and padding token multiplicity
            n_max_hads = sentence_stats["true_n_hads_per_cluster"][0] + \
                            sentence_stats["true_n_pad_hads_per_cluster"][0]
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            bins = np.linspace(start=-0.5, stop=n_max_hads+0.5, num=n_max_hads+2, retstep=0.5)[0]
            datatype = ["true", "pred"]
            labels = ["True", "Generated"]
            colours = ["black", "#DE3C4B"]
            for col in range(0, 2):
                for i in range(0, 2):
                    if col == 0:
                        axs[col].hist(sentence_stats[f"{datatype[i]}_n_hads_per_cluster"], 
                                            bins=bins, color=colours[i], label=labels[i], rwidth=0.9)
                        axs[col].set_xlabel("Number of hadrons")
                    else:
                        axs[col].hist(sentence_stats[f"{datatype[i]}_n_pad_hads_per_cluster"], 
                                            bins=bins, color=colours[i], label=labels[i], rwidth=0.9)
                        axs[col].set_xlabel("Number of padding tokens")
                axs[col].legend()
                axs[col].set_ylabel("Sentences")
            fig.suptitle("Sentence Statistics")
            plt.tight_layout()
            diagrams["sentence_statistics_hist"] = fig_to_array(fig, tight_layout=False)
        
        elif clusters is not None:
            diagrams = {}
            kinematics = clusters[:, :4]
            quark_types = clusters[:, 4:6]
            quark_angles = clusters[:, 6:]

            # Cluster kinematics histograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axis = ['x', 'y', 'z']
            for col in range(0, 3):
                axs[col].hist(kinematics[:, col], bins="auto", color="#AB92BF")
                axs[col].set_xlabel(f"Momentum ({axis[col].capitalize()})")
                axs[col].set_ylabel("Clusters")
            fig.suptitle("Cluster Momentum Distribution")
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
                        axs[row][col].hist(quark_idx, bins=bins, rwidth=0.8, color="#4C1E4F")
                        axs[row][col].set_xlabel("PID")
                        axs[row][col].title.set_text("Type") 
                        axs[row][col].set_xticks([int(pid) for pid in pids_to_idx.values()])
                        axs[row][col].set_xticklabels([int(pid) for pid in pids_to_idx.keys()], 
                                                      rotation=90)
                    else:
                        axs[row][col].hist(quark_angles[:, col], bins="scott", rwidth=0.8, 
                                           color="#B5A886")
                        axs[row][col].set_xlabel("Angle") 
                        axs[row][col].title.set_text(f"Kinematics ({angles[col]})") 
                    axs[row][col].set_ylabel("Quarks")
            fig.suptitle("Quark Type and Momentum Distribution")
            plt.tight_layout()
            diagrams["quarks_features_hist"] = fig_to_array(fig, tight_layout=False)    
            
        return diagrams