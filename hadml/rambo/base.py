# Code from https://github.com/madgraph-ml/madspace

from typing import Tuple, Iterable, Optional, List
import torch
from torch import Tensor
import torch.nn as nn

# Definition of InputTypes and OutputTypes
ShapeList = List[Tuple[int, ...]]
TensorList = List[Tensor]
TensorTuple = Tuple[Tensor, ...]


class PhaseSpaceMapping(nn.Module):
    """Base class for all phase-space mappings.

    Note:
    This is not a 1:1 replacement of the Mapping class in MadNIS
    as it asks for more shape information and also allows for
    multiple conditions. Further, to make it more easy as a reader.
    the forward pass denotes the mapping from the random numbers to
    the moment and vice versa, i.e.
        ..math::
            forward: f(r) = p.
            inverse: f^{-1}(p) = r.
    """

    def __init__(
        self,
        dims_in: ShapeList,
        dims_out: ShapeList,
        dims_c: Optional[ShapeList] = None,
    ):
        """
        Args:
            dims_in (ShapeList): list of input shapes for the forward map w/o batch dimension ``b``.
                Includes random numbers r and potential auxiliary inputs.
            dims_out (ShapeList): list of output shapes as inputs for inverse map w/o batch dimension ``b``.
                Usually only includes a sigle shape from the momentum tensor.
            dims_c (ShapeList, optional): list of shapes for the conditions. Defaults to None.
        """
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dims_c = dims_c

    def forward(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        inverse=False,
        **kwargs,
    ) -> Tuple[TensorTuple, Tensor]:
        """
        Forward pass of the ps-mapping ``f``. This is the pass
        from the random numbers ``r` to the momenta ``p``, i.e.
            ..math::
                f(r) = p.

        If inverse = True, it encodes the inverse pass ``f^{-1}`` of the ps-mapping.
        This is the pass from the momenta ``p`` to the random numbers ``r``, i.e.
            ..math::
                f^{-1}(p) = r.
        Args:
            inputs (TensorList): forward map inputs with shapes=[(b, *dims_in0), (b, *dims_in1),...].
            condition (TensorList, optional): conditional inputs. Defaults to None.

        Returns:
            out (TensorTuple): tuple including momenta with shape=(b, *dims_out0).
            density (Tensor): the density (det) of the mapping with shape=(b,).
        """
        if inverse:
            return self.map_inverse(inputs, condition, **kwargs)
        return self.map(inputs, condition, **kwargs)

    def map(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[TensorTuple, Tensor]:
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map(...) method"
        )

    def map_inverse(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[TensorTuple, Tensor]:
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map_inverse(...) method"
        )

    def density(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        inverse: bool = False,
        **kwargs,
    ) -> Tensor:
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map_inverse(...) method"
        )
