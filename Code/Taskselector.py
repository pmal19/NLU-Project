import sys
from Models.blocks import *
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
class Taskselector(nn.Module):
	def __init__(self, hidden_dim=100,num_source_tasks=2, gumbel_temperature=1e-20):
		super(Taskselector, self).__init__()
		self.hidden_dim=hidden_dim
		self.gumbel_temperature=gumbel_temperature
		self.init_linear=Linear()(hidden_dim*2, num_source_tasks, bias=True)

	def forward(self, se, n_tasks):
		se=torch.cat(se, dim=2)
		se_prerelu=self.init_linear(se)
		se_postrelu=F.relu(se_prerelu)
		logits=F.log_softmax(se_postrelu)
		selector=self.st_gumbel_softmax(logits)
		return torch.from_numpy(se*np.repeat(selector, self.hidden_dim))
	

	def masked_softmax(self,logits, mask=None):
	    eps = 1e-20
	    probs = F.softmax(logits)
	    if mask is not None:
	        mask = mask.float()
	        probs = probs * mask + eps
	        probs = probs / probs.sum(1, keepdim=True)
	    return probs

	def st_gumbel_softmax(self,logits, temperature=1.0, mask=None):
	    """
	    Return the result of Straight-Through Gumbel-Softmax Estimation.
	    It approximates the discrete sampling via Gumbel-Softmax trick
	    and applies the biased ST estimator.
	    In the forward propagation, it emits the discrete one-hot result,
	    and in the backward propagation it approximates the categorical
	    distribution via smooth Gumbel-Softmax distribution.

	    Args:
	        logits (Variable): A un-normalized probability values,
	            which has the size (batch_size, num_classes)
	        temperature (float): A temperature parameter. The higher
	            the value is, the smoother the distribution is.
	        mask (Variable, optional): If given, it masks the softmax
	            so that indices of '0' mask values are not selected.
	            The size is (batch_size, num_classes).

	    Returns:
	        y: The sampled output, which has the property explained above.
	    """
		def convert_to_one_hot(indices, num_classes):
		    """
		    Args:
		        indices (Variable): A vector containing indices,
		            whose size is (batch_size,).
		        num_classes (Variable): The number of classes, which would be
		            the second dimension of the resulting one-hot matrix.

		    Returns:
		        result: The one-hot matrix of size (batch_size, num_classes).
		    """

		    batch_size = indices.size(0)
		    indices = indices.unsqueeze(1)
		    one_hot = Variable(indices.data.new(batch_size, num_classes).zero_()
		                       .scatter_(1, indices.data, 1))
		    return one_hot

	    eps = 1e-20
	    u = logits.data.new(*logits.size()).uniform_()
	    gumbel_noise = Variable(-torch.log(-torch.log(u + eps) + eps))
	    y = logits + gumbel_noise
	    y = self.masked_softmax(logits=y / temperature, mask=mask)
	    y_argmax = y.max(1)[1]
	    y_hard = convert_to_one_hot(
	        indices=y_argmax,
	        num_classes=y.size(1)).float()
	    y = (y_hard - y).detach() + y
	    return y



