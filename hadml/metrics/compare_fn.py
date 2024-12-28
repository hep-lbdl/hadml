import math
import os
from typing import List, Tuple, Optional, Any, Dict
from matplotlib import ticker
from pytorch_lightning.core.mixins import HyperparametersMixin
import numpy as np
import matplotlib.pyplot as plt
from .image_converter import fig_to_array
from torch import Tensor


def create_plots(nrows, ncols):
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=False
    )
    axs = axs.flatten()
    return fig, axs


class CompareParticles(HyperparametersMixin):
    def __init__(
        self,
        xlabels: List[str],
        num_kinematics: int,
        num_particles: int,
        num_particle_ids: int,
        outdir: Optional[str] = None,
        xranges: Optional[List[Tuple[float, float]]] = None,
        xbins: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        super().__init__()
        self.save_hyperparameters()

    def __call__(
        self, predictions: np.ndarray, truths: np.ndarray, tags: Optional[str] = None
    ) -> Dict[str, Any]:
        """Expect predictions = [batch_size, num_kinematics + num_particle_type_indices]."""
        out_images = {}

        _, num_dims = truths.shape
        assert num_dims == (self.hparams.num_kinematics + self.hparams.num_particles)

        xranges = self.hparams.xranges
        xbins = self.hparams.xbins
        xlabels = self.hparams.xlabels

        outname = "dummy" if tags is None else tags
        if self.hparams.outdir is not None:
            os.makedirs(self.hparams.outdir, exist_ok=True)
            outname = os.path.join(self.hparams.outdir, outname)
        else:
            outname = None

        needed_plot_count = self.hparams.num_kinematics * (1 + self.hparams.num_particle_ids)
        plot_row_count = math.ceil(needed_plot_count ** 0.5)
        plot_col_count = math.ceil(needed_plot_count / plot_row_count)
        fig, axs = create_plots(plot_row_count, plot_col_count)
        axs = axs.reshape(-1, 2)
        self._plot_kinematics(predictions, truths, xbins, xlabels, xranges, fig, axs[0, :])

        for i in range(self.hparams.num_particle_ids):
            sim_particle_types = (predictions[:, self.hparams.num_kinematics:] == i).sum(-1) > 0
            true_particle_types = (truths[:, self.hparams.num_kinematics:] == i).sum(-1) > 0
            predictions_i = predictions[sim_particle_types]
            truths_i = truths[true_particle_types]
            xlabels_i = [l + f" [pid={i}]" for l in xlabels]
            self._plot_kinematics(predictions_i, truths_i, xbins,
                                  xlabels_i, xranges, fig, axs[1 + i, :])

        if outname is not None:
            plt.savefig(outname + "-angles.png")
            plt.savefig(outname + "-angles.pdf")

        # convert the image to a numpy array
        out_images[f"particle kinematics"] = fig_to_array(fig)
        plt.close("all")

        config = dict(alpha=0.5, lw=2, density=True)

        # figure out predicted particle type
        num_particles = self.hparams.num_particles
        if num_particles > 0:
            fig, axs = create_plots(1, num_particles)
            ranges = (-0.5, self.hparams.num_particle_ids + 0.5)
            bins = self.hparams.num_particle_ids + 1

            for idx in range(num_particles):
                sim_particle_types = predictions[:, self.hparams.num_kinematics + idx]
                true_particle_types = truths[:, self.hparams.num_kinematics + idx]

                ax = axs[idx]
                bin_heights_true, _, patches_true = ax.hist(
                    true_particle_types,
                    bins=bins,
                    range=ranges,
                    label="Truth",
                    **config,
                )
                bin_heights_false, _, patches_false = ax.hist(
                    sim_particle_types,
                    bins=bins,
                    range=ranges,
                    label="Generator",
                    **config,
                )
                ax.set_xlabel(r"{}".format(f"{idx}th particle type"))
                bin_heights = np.concatenate([bin_heights_true, bin_heights_false])
                patches = np.concatenate([patches_true, patches_false])

                self.set_hist_log_scale(ax, patches, bin_heights)
                ax.legend()

            if outname is not None:
                plt.savefig(outname + "-types.png")
                plt.savefig(outname + "-types.pdf")

            # convert the image to a numpy array
            out_images["particle type"] = fig_to_array(fig)
            plt.close("all")

        return out_images

    def _plot_kinematics(self, predictions, truths, xbins, xlabels, xranges, fig, axs):
        config = dict(alpha=0.5, lw=2, density=True)
        for idx in range(self.hparams.num_kinematics):
            xrange = xranges[idx] if xranges else (-1, 1)
            xbin = xbins[idx] if xbins else 40

            if len(truths[:, idx]) == 0:
                continue
            ax = axs[idx]
            bin_heights, _, _ = ax.hist(
                truths[:, idx], bins=xbin, range=xrange, label="Truth", **config
            )
            max_y = np.max(bin_heights) * 1.1
            ax.hist(
                predictions[:, idx],
                bins=xbin,
                range=xrange,
                label="Generator",
                **config,
            )
            ax.set_xlabel(r"{}".format(xlabels[idx]))
            ax.set_ylim(0, max_y)
            ax.legend()


    @staticmethod
    def set_hist_log_scale(ax, patches, bin_heights):
        """Set log scale on y-axis of `ax`.

        Set log scale without using `ax.set_scale`. It's assumed that
        `bin_heights` are from range [0, 1].
        """
        if ((bin_heights < 0.0) & (bin_heights > 1.0)).any():
            raise ValueError("`bin_heights` should be in range [0, 1].")

        nonzero_idx = bin_heights != 0
        log_heights = np.zeros(bin_heights.shape)
        log_heights[nonzero_idx] = np.log10(bin_heights[nonzero_idx])

        # bottom ylim will be lower than the lowest frequency
        ylim_bot = np.min(log_heights) - 0.5
        log_heights[~nonzero_idx] = ylim_bot

        # modify bin heights to represent logarithm of counts
        for patch, log_height in zip(patches, log_heights):
            patch.set_height(log_height - ylim_bot)
            patch.set_y(ylim_bot)

        minor_ticks = []
        for i in range(0, -int(np.floor(ylim_bot))):
            minor_ticks += [
                np.log10(10 ** (-i) - j / 10 ** (i + 1)) for j in range(1, 9)
            ]
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

        ax.set_ylim(ylim_bot, 0)


class CompareParticlesEventGan(HyperparametersMixin):
    def __init__(
        self,
        xlabels: List[str],
        outdir: Optional[str] = None,
        xranges: Optional[List[Tuple[float, float]]] = None,
        xbins: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        super().__init__()
        self.save_hyperparameters()

    def __call__(
        self,
        angles_predictions: np.ndarray,
        angles_truths: np.ndarray,
        hadrons_predictions: np.ndarray,
        hadrons_truth: np.ndarray,
        tags: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_images = {}

        xranges = self.hparams.xranges
        xbins = self.hparams.xbins
        xlabels = self.hparams.xlabels

        outname = "dummy" if tags is None else tags
        if self.hparams.outdir is not None:
            os.makedirs(self.hparams.outdir, exist_ok=True)
            outname = os.path.join(self.hparams.outdir, outname)
        else:
            outname = None

        fig, axs = create_plots(1, 6)
        config = dict(histtype="step", lw=2, density=True)

        # angles
        for idx in range(2):
            xrange = xranges[idx] if xranges else (-1, 1)
            xbin = xbins[idx] if xbins else 40

            ax = axs[idx]
            max_y = 0
            if len(angles_truths) > 0:
                yvals, _, _ = ax.hist(angles_truths[:, idx], bins=xbin,
                                      range=xrange, label='Truth', **config)
                max_y = np.max(yvals)
            yvals, _, _ = ax.hist(angles_predictions[:, idx], bins=xbin,
                                  range=xrange, label='Generator', **config)
            max_y = max(max_y, np.max(yvals)) * 1.1

            ax.set_xlabel(r"{}".format(xlabels[idx]))
            ax.set_ylim(0, max_y * 1.1)
            ax.legend()

        # 4-momentum
        for idx in range(4):
            xrange = xranges[idx + 2] if xranges else (-1, 1)
            xbin = xbins[idx + 2] if xbins else 40

            ax = axs[idx + 2]
            yvals, _, _ = ax.hist(
                hadrons_truth[:, idx], bins=xbin, range=xrange, label="Truth", **config
            )
            max_y = np.max(yvals) * 1.1
            ax.hist(
                hadrons_predictions[:, idx],
                bins=xbin,
                range=xrange,
                label="Generator",
                **config,
            )
            ax.set_xlabel(r"{}".format(xlabels[idx + 2]))
            ax.set_ylim(0, max_y)
            ax.legend()

        if outname is not None:
            plt.savefig(outname + "-kinematics.png")
            plt.savefig(outname + "-kinematics.pdf")
        # convert the image to a numpy array
        out_images["particle kinematics"] = fig_to_array(fig)
        plt.close("all")

        return out_images