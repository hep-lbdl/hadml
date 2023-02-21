import os
from typing import List, Tuple, Optional
from pytorch_lightning.core.mixins import HyperparametersMixin

import os
import numpy as np
import matplotlib.pyplot as plt

def create_plots(nrows, ncols):
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 4*nrows), constrained_layout=False)
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
        
    def __call__(self, predictions: np.ndarray,
                truths: np.ndarray,
                tags: Optional[str] = None) -> None:
        """Expect predictions = [batch_size, num_kinematics + num_particle_type_indices]."""
        _, num_dims = truths.shape
        assert num_dims == self.hparams.num_kinematics + self.hparams.num_particles
        
        xranges = self.hparams.xranges
        xbins = self.hparams.xbins
        xlabels = self.hparams.xlabels
        
        outname = "dummy" if tags is None else tags
        if self.hparams.outdir is not None:
            os.makedirs(self.hparams.outdir, exist_ok=True)
            outname = os.path.join(self.hparams.outdir, outname)
        else:
            outname = None

        fig, axs = create_plots(1, self.hparams.num_kinematics)
        config = dict(histtype='step', lw=2, density=True)
        for idx in range(self.hparams.num_kinematics):
            xrange = xranges[idx] if xranges else (-1, 1)
            xbin = xbins[idx] if xbins else 40

            ax = axs[idx]
            yvals, _, _ = ax.hist(truths[:, idx], bins=xbin, range=xrange, label='Truth', **config)
            max_y = np.max(yvals) * 1.1
            ax.hist(predictions[:, idx], bins=xbin, range=xrange, label='Generator', **config)
            ax.set_xlabel(r"{}".format(xlabels[idx]))
            ax.set_ylim(0, max_y)
            ax.legend()

        if outname is not None:
            plt.savefig(outname+"-angles.png")
            plt.savefig(outname+"-angles.pdf")
        ## convert the image to a numpy array
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data_angle = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt.close('all')
        
        ## figure out predicted particle type
        num_particles = self.hparams.num_particles
        if num_particles > 0:
            fig, axs = create_plots(1, num_particles)
            ranges = (-0.5, self.hparams.num_particle_ids+0.5)
            bins = self.hparams.num_particle_ids + 1

            for idx in range(num_particles):
                sim_particle_types  = predictions[:, self.hparams.num_kinematics+idx]
                true_particle_types = truths[:, self.hparams.num_kinematics+idx]
                
                ax = axs[idx]
                yvals, _, _ = ax.hist(true_particle_types, bins=bins, range=ranges, label='Truth', **config)
                max_y = np.max(yvals) * 1.1
                ax.hist(sim_particle_types, bins=bins, range=ranges, label='Generator', **config)
                ax.set_xlabel(r"{}".format(f"{idx}th particle type"))
                ax.set_ylim(0, max_y)
                ax.legend()

            if outname is not None:
                plt.savefig(outname+"-types.png")
                plt.savefig(outname+"-types.pdf")

            ## convert the image to a numpy array
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            data_type = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plt.close('all')
            return data_angle, data_type
        else:
            return data_angle, None
