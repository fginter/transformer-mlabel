####
### THE CODE FOR AliasMultinomial() IS VERBATIM FROM https://github.com/Stonesjtu/Pytorch-NCE
###
### BORROWED UNDER THE MIT LICENSE
###
### ADDED STUFF RELEVANT TO THIS PROJECT
### 

from math import isclose

import torch
import data

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''

    @classmethod
    def from_class_stats(cls,class_stats_filename,max_labels,flat=False):
        """
        'stats': a dict of label -> count
        'filtered'
        """
        idx2label,label2idx,class_stats=data.prep_class_stats(class_stats_filename,max_labels)
        probs=torch.zeros((len(class_stats),),dtype=torch.float)
        for label,count in class_stats.most_common(max_labels):
            if not flat:
                probs[label2idx[label]]=count
            else:
                probs[label2idx[label]]=1
        probs/=torch.sum(probs) #probs
        return cls(probs)
        
    
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert isclose(probs.sum().item(), 1, rel_tol=1e-06), 'The noise distribution must sum to 1'
        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)

