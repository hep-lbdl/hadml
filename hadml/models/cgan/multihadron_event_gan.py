import torch
from pytorch_lightning import LightningModule
from utils.utils import conditional_cat
from typing import Optional, Callable
from metrics.media_logger import log_images
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np
from metrics.image_converter import fig_to_array
from collections import Counter


class MultiHadronEventGANModule(LightningModule):
    def __init__(
        self,
        datamodule: torch.nn.Module,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: Optimizer,
        optimizer_discriminator: Optimizer,
        noise_dim: int
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, clusters):
        generator_input = conditional_cat(clusters, self._generate_noise(len(clusters)))
        generated_hadrons = self.hparams.generator(generator_input)
        return generated_hadrons
    
    def setup(self, stage=None):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Debugging: batch_size=1, num_workers=0:
        # batch[0] - generator input
        # batch[1] - ideal generator output (discriminator input)
        return None

    def validation_step(self, batch, batch_idx):
        gen_input, disc_input = batch
        device = gen_input.device
        if self.trainer.state.stage == "validate":
            # ==========================================================================================
            # Simulating model predictions on validation data by providing some noise to the true values
            # ==========================================================================================
            batch_size = len(disc_input)
            n_types = len(disc_input[0, 0, 4:])
            max_n_hadrons = len(disc_input[0])
            true_types = torch.stack([torch.argmax(d[:, 4:], dim=1) for d in disc_input])
            distortion_noise =  torch.randint(0, n_types, (batch_size, max_n_hadrons))
            distorted_types = (true_types + distortion_noise.to(device)) % n_types
            distorted_disc_input = torch.stack([
                torch.column_stack([
                    d[:, :4] * torch.rand(1).to(device),           # scaling kinematics (Gaussian noise)
                    torch.nn.functional.one_hot(d_type,            # randomly shifted types
                                                num_classes=n_types)            
                ]) for d, d_type in zip(disc_input, distorted_types)
            ])
            # ==========================================================================================
            return {"distorted_disc_input": distorted_disc_input, "disc_input": disc_input}
        
        elif self.trainer.state.stage == "sanity_check":
            return {"gen_input": gen_input[:, 0, :]}

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        generator_opt = self.hparams.optimizer_generator(params=self.generator.parameters())
        disriminator_opt = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())
        return generator_opt, disriminator_opt
    
    def _generate_noise(self, n_tokens):
        return torch.randn(n_tokens, self.hparams.noise_dim)
    
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
            preds = [d["distorted_disc_input"] for d in validation_step_outputs]
            preds = [d for pred in preds for d in pred]      # [n_hadron_sets, max_n_hadrons, features]
            preds = [d for pred in preds for d in pred]      # [total_n_hadrons, features]
            preds = torch.stack(preds)         
            truths = [d["disc_input"] for d in validation_step_outputs]
            truths = [d for truth in truths for d in truth]  # [n_hadron_sets, max_n_hadrons, features]
            truths = [d for truth in truths for d in truth]  # [total_n_hadrons, features]
            truths = torch.stack(truths)         

            # Preparing diagrams
            images = self._prepare_plots(predictions=preds.cpu(), truths=truths.cpu())
        
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

    def _prepare_plots(self, predictions=None, truths=None, clusters=None):
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
            bins = np.linspace(
                start=sample_range[0] - 0.5, 
                stop=sample_range[1] + 0.5, 
                num=sample_range[1] - sample_range[0] + 2, 
                retstep=0.5)[0]
            plt.hist(truths_types, bins=bins, color="black", histtype="step", label="True")
            plt.hist(preds_types, bins=bins, color="#AABA9E", label="Generated")
            plt.ylabel("Hadrons")
            plt.xlabel("Hadron Most Common ID\n(mapped from PIDs)", labelpad=20)
            xticks = np.arange(start=sample_range[0] - 1, stop=sample_range[1] + 1, step=density)[1:]
            plt.xticks(xticks, rotation=90)
            plt.legend()
            plt.tight_layout()
            diagrams["hadron_type_hist"] = fig_to_array(fig, tight_layout=False)
            plt.show()

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
            plt.show()
            
            diagrams["hadron_kinematics_hist"] = fig_to_array(fig, tight_layout=False)
        
        elif clusters is not None:
            diagrams = {}
            kinematics = clusters[:, :4]
            quark_types = clusters[:, 4:6]
            quark_angles = clusters[:, 6:]

            # Cluster kinematics histograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axis = ['x', 'y', 'z']
            for row in range(0, 3):
                axs[row].hist(kinematics[:, row], bins="auto", color="black")
                axs[row].set_xlabel(f"Momentum ({axis[row].capitalize()})")
                axs[row].set_ylabel("Clusters")
            fig.suptitle("Cluster Momentum Distribution")
            plt.tight_layout()
            diagrams["cluster_kinematics_hist"] = fig_to_array(fig, tight_layout=False)
            plt.show()

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
            plt.show()
            
        return diagrams