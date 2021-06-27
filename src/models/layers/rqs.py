from typing import List, Optional
import FrEIA.modules as Fm
from nflows import transforms

class RQSplines(Fm.InvertibleModule):

    def __init__(
        self, 
        dims_in: List[int], 
        num_bins: int=10, 
        tails: Optional[str]=None, 
        tail_bound: float=1.0, 
        identity_init: bool=False,
        ):
        super().__init__(dims_in)

        self.transform = transforms.PiecewiseRationalQuadraticCDF(
            shape=dims_in[0], 
            num_bins=num_bins, 
            tails=tails, 
            tail_bound=tail_bound,
            identity_init=identity_init
            )

    def forward(self, x, rev=False, jac=True):
        # the Jacobian term is trivial to calculate so we return it
        # even if jac=False

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:
            # forward operation
            x, log_jac_det = self.transform.forward(x)
        else:
            # backward operation
            x, log_jac_det = self.transform.inverse(x)

        return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
