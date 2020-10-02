"""
Joint Energy-Based Model.
"""
import torch
import torch.nn as nn


def init_random(bs, shape):
    return torch.FloatTensor(bs, *shape).uniform_(-1, 1)


def get_buffer(args):
    if args.jem_baseline:
        return init_random(args.buffer_size, args.data_size)
    else:
        return None


def sample_p_0(replay_buffer, bs, reinit_freq):
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, replay_buffer.size()[1:])
    choose_random = (torch.rand(bs) < reinit_freq).float()
    if len(replay_buffer.size()) == 2:
        choose_random = choose_random[:, None]
    elif len(replay_buffer.size()) == 3:
        choose_random = choose_random[:, None, None]
    elif len(replay_buffer.size()) == 4:
        choose_random = choose_random[:, None, None, None]
    else:
        raise ValueError
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples, inds


def sample_q(f, replay_buffer, batch_size, n_steps, sgld_lr, sgld_std, reinit_freq, device):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, batch_size, reinit_freq)
    init_sample = init_sample.to(device)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


class JEM(nn.Module):
    """
    JEM model.
    """
    def __init__(self, logp_net):
        super().__init__()
        self.logp_net = logp_net

    def forward(self, x, return_logits=False):
        """
        Forward pass.
        """
        logits = self.logp_net(x)
        if return_logits:
            return logits.logsumexp(1), logits
        else:
            return logits.logsumexp(1)

    def classify(self, x):
        """
        Use model as classifier.
        """
        return self.logp_net(x)
