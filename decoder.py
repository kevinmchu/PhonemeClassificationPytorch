# External
import torch


class ViterbiDecoder:

    def __init__(self, prior_probs, trans_mat, model, device):
        self.prior_probs = prior_probs # unigram log probs
        self.trans_mat = trans_mat # log probs of transition
        self.model = model
        self.device = device

    def calculate_log_likelihood(self, obs):
        post_probs = self.model(obs)
        log_likelihood = post_probs - self.prior_probs

        return log_likelihood

    def decode(self, obs):
        log_likelihood = self.calculate_log_likelihood(obs)

        # Reshape
        log_likelihood = torch.squeeze(log_likelihood, 0)

        # Preallocate Viterbi cells and backtrace
        v = (torch.zeros(log_likelihood.size()[0], log_likelihood.size()[1])).to(self.device)
        bt = (torch.zeros(log_likelihood.size()[0], log_likelihood.size()[1])).to(self.device)

        # Initialization
        v[0, :] = self.prior_probs + log_likelihood[0, :]

        # Recursion
        for t in range(1, log_likelihood.size()[0]):
            for s in range(log_likelihood.size()[1]):
                temp_prob = v[t-1, :] + self.trans_mat[s, :] + log_likelihood[t, s]
                v[t, s] = torch.max(temp_prob)
                bt[t, s] = torch.argmax(temp_prob)

        # Termination
        best_score = torch.max(v[-1, :])

        # Backtrace to get best path
        best_path = torch.zeros(log_likelihood.size()[0], dtype=torch.int)
        best_path[-1] = torch.argmax(v[-1, :])

        for t in range(log_likelihood.size()[0]-2, -1, -1):
            best_path[t] = bt[t+1, best_path[t+1].item()]

        return best_path, best_score
