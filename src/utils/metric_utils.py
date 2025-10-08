# This file contains the implementation of the r2 score metric
from torcheval.metrics import R2Score
import numpy as np
import logging
logger = logging.getLogger(__name__)
from scipy.special import gammaln
import torch.nn as nn
import torch
from sklearn.metrics import r2_score

#%%
r2_metric = R2Score()
def r2_score_torch(y_true, y_pred, device="cpu"):
    """
    for torch tensors
    """
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()


def compute_R2_main(y, y_pred, clip=True):
    """
    for numpy arrays
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s


def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates[mask] = 0
        spikes[mask] = 0

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates < 1e-9] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result, axis=tuple(range(spikes.ndim - 1)))


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    (never used in the paper)

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)

    # print(np.nansum(spikes))
    return (nll_null - nll_model) / np.nansum(spikes, axis=(tuple(range(spikes.ndim-1))))/ np.log(2)


def Poisson_fraction_deviance_explained(rates, spikes):
    """
    rates: (K, T, N)
    spikes: (K, T, N)
    """

    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )

    null_rates[null_rates < 1e-9] = 1e-9 

    sat_rates = spikes.copy()
    sat_rates[sat_rates < 1e-9] = 1e-9


    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    nll_sat = neg_log_likelihood(sat_rates, spikes, zero_warning=False)

    return 1 - (nll_model - nll_sat) / (nll_null - nll_sat + 1e-9)  # avoid division by zero


#%%    
def GLM_Poisson(weight, bias, factors, spks, lambda_reg=0.01):
    '''
    factors: B x T x C
    spks: B x T x N
    weight: C x N
    bias: N

    return: poisson negative log likelihood
    '''
    tmp = factors @ weight + bias[None, None, :]

    # Clamp for numerical stability
    tmp = torch.clamp(tmp, min=-20.0, max=20.0)

    nll = nn.PoissonNLLLoss(log_input=True)(tmp, spks)

    #reg = lambda_reg * torch.sqrt((weight ** 2).sum() + (bias ** 2).sum())

    return nll #+ reg

#%%

def get_deviance_explained(factors_region, spikes_region, device, verbose=False):
    '''
    factors_region: B x T x C
    spikes_region: B x T x N
    
    return:
    fr_pred: B x T x N
    dfe: N, fraction deviance explained for each neuron
    '''

    weight = torch.zeros(factors_region.size(-1), spikes_region.size(-1), device=device, requires_grad=True)
    bias = torch.zeros(spikes_region.size(-1), device=device, requires_grad=True)
    
    optimizer_lbfgs = torch.optim.LBFGS([weight, bias], lr=0.5, max_iter=20, line_search_fn='strong_wolfe')

    # Training with LBFGS (second-order optimizer)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = GLM_Poisson(weight, bias, factors_region, spikes_region)
        loss.backward()
        return loss

    for epoch in range(100):  # LBFGS converges in fewer iterations
        optimizer_lbfgs.step(closure)
        loss = GLM_Poisson(weight, bias, factors_region, spikes_region)
        #print(f"LBFGS Iter {epoch}: Loss = {loss.item():.4f}")

        if epoch>1 and abs(prev_loss - loss.item()) < 1e-5:
            if verbose:
                print(f'epoch {epoch} '+'converged. ')
            break
        prev_loss = loss.item()

    tmp = factors_region @ weight + bias[None, None, :]
    tmp = torch.clamp(tmp, min=-20.0, max=20.0) #avoid numerical instability

    fr_pred = torch.exp(tmp)

    if fr_pred.isnan().any():
        print('nan in fr_pred')
        dfe = np.nan
    else:
        dfe = Poisson_fraction_deviance_explained(fr_pred.cpu().detach().numpy(), spikes_region.cpu().detach().numpy())

    return fr_pred, weight, bias, dfe



