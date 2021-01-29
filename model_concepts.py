import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, logprobs, reward):
        logprobs = to_contiguous(logprobs).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        # add one to the right to count for the <eos> token
        mask = to_contiguous(torch.cat(
            [mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - logprobs * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output


class CrossEntropyCriterion(nn.Module):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, pred, target, mask, bcmrscores=None):
        # truncate to the same size

        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]
        seq_len = pred.size(1)
        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -pred.gather(1, target) * mask
        if bcmrscores is not None:
            weights = bcmrscores.view(-1).unsqueeze(1).repeat(1, seq_len).view(-1, 1)
        else:
            weights = torch.ones(output.shape).cuda() 
        output = torch.sum(output*weights) / torch.sum(mask)

        return output


class FeatPool(nn.Module):
    def __init__(self, feat_dims, out_size, dropout, SQUEEZE=True):
        super(FeatPool, self).__init__()
        self.squeeze = SQUEEZE
        module_list = []
        for dim in feat_dims:
            module = nn.Sequential(
                nn.Linear(dim, out_size),
                nn.ReLU(),
                nn.Dropout(dropout))
            module_list += [module]
        self.feat_list = nn.ModuleList(module_list)


    def forward(self, feats, stack=False):
        """
        feats is a list, each element is a tensor that have size (N x C x F)
        at the moment assuming that C == 1
        """
        if stack:
            out = torch.stack([m(feats[i].squeeze(1)) for i, m in enumerate(self.feat_list)], 1)
        else:
            if self.squeeze:
                out = torch.cat([m(feats[i].squeeze(1)) for i, m in enumerate(self.feat_list)], 1)
            else:
                out = torch.cat([m(feats[i]) for i, m in enumerate(self.feat_list)], 2)
        return out


class FeatExpander(nn.Module):

    def __init__(self, n=1):
        super(FeatExpander, self).__init__()
        self.n = n

    def forward(self, x):
        if self.n == 1:
            out = x
        else:
            if len(x.shape) == 2:
                out = Variable(x.data.new(self.n * x.size(0), x.size(1)), volatile=x.volatile)
                for i in range(x.size(0)):
                    out[i * self.n:(i + 1) * self.n] = x[i].expand(self.n, x.size(1))
            elif len(x.shape) == 3:
                out = Variable(x.data.new(self.n * x.size(0), x.size(1), x.size(2)), volatile=x.volatile)
                for i in range(x.size(0)):
                    out[i * self.n:(i + 1) * self.n] = x[i].expand(self.n, x.size(1), x.size(2))

        return out

    def set_n(self, x):
        self.n = x


class RNNUnit(nn.Module):

    def __init__(self, opt):
        super(RNNUnit, self).__init__()
        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.drop_prob_lm = opt.drop_prob_lm

        if opt.model_type == 'standard':
            self.input_size = opt.input_encoding_size
        elif opt.model_type in ['concat', 'manet']:
            self.input_size = opt.input_encoding_size + opt.video_encoding_size

        self.rnn = getattr(nn, self.captioner_type.upper())(self.input_size, self.captioner_size, self.captioner_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, state):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state


class MANet(nn.Module):
    """
    MANet: Modal Attention
    """

    def __init__(self, video_encoding_size, captioner_size, num_feats):
        super(MANet, self).__init__()
        self.video_encoding_size = video_encoding_size
        self.captioner_size = captioner_size
        self.num_feats = num_feats

        self.f_feat_m = nn.Linear(self.video_encoding_size, self.num_feats)
        self.f_h_m = nn.Linear(self.captioner_size, self.num_feats)
        self.align_m = nn.Linear(self.num_feats, self.num_feats)

    def forward(self, x, h):
        f_feat = self.f_feat_m(x)
        f_h = self.f_h_m(h.squeeze(0))  # assuming now captioner_layers is 1
        att_weight = nn.Softmax()(self.align_m(nn.Tanh()(f_feat + f_h)))
        att_weight = att_weight.unsqueeze(2).expand(
            x.size(0), self.num_feats, int(self.video_encoding_size / self.num_feats))
        att_weight = att_weight.contiguous().view(x.size(0), x.size(1))
        return x * att_weight




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# TODO make a box enc to FIRST cat then encode, rather than encode then cat
# class BoxEnc(nn.Module):
#     def __init__(self, feat_dims, out_size, dropout, SQUEEZE=True):
#         super(BoxEnc, self).__init__()
#         self.squeeze = SQUEEZE
#         module_list = []
#         for dim in feat_dims:
#             module = nn.Sequential(
#                 nn.Linear(dim, out_size),
#                 nn.ReLU(),
#                 nn.Dropout(dropout))
#             module_list += [module]
#         self.feat_list = nn.ModuleList(module_list)
#
#
#     def forward(self, feats):
#         """
#         feats is a list, each element is a tensor that have size (N x C x F)
#         at the moment assuming that C == 1
#         """
#         if self.squeeze:
#             out = torch.cat([m(feats[i].squeeze(1)) for i, m in enumerate(self.feat_list)], 1)
#         else:
#             out = torch.cat([m(feats[i]) for i, m in enumerate(self.feat_list)], 2)
#         return out

class SVORNN(nn.Module):
    """
    Allow S V and O to be put into LTSM when --pass_all_svo is set to 1
    """

    def __init__(self, opt):
        super(SVORNN, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.att_size = opt.att_size
        self.captioner_layers = opt.captioner_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.pass_all_svo = opt.pass_all_svo
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.captioner_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.l2a_layer = nn.Linear(self.input_encoding_size, self.att_size)
        self.h2a_layer = nn.Linear(self.captioner_size, self.att_size)
        self.att_layer = nn.Linear(self.att_size, 1)

        self.init_weights()
        if self.model_type == 'standard':
            self.feat_pool = FeatPool(self.feat_dims[0:1], self.captioner_layers * self.captioner_size, self.drop_prob_lm)
        else:
            self.feat_pool = FeatPool(self.feat_dims, self.captioner_layers * self.captioner_size, self.drop_prob_lm)

            # encode the box features 1024 -> 512 and 4 -> 512 (linear>relu>dropout)
            self.bfeat_pool_q = FeatPool(self.bfeat_dims, self.captioner_layers * self.captioner_size, self.drop_prob_lm,
                                         SQUEEZE=False)
            self.bfeat_pool_k = FeatPool(self.bfeat_dims, self.captioner_layers * self.captioner_size, self.drop_prob_lm,
                                         SQUEEZE=False)
            self.bfeat_pool_v = FeatPool(self.bfeat_dims, self.captioner_layers * self.captioner_size, self.drop_prob_lm,
                                         SQUEEZE=False)

            # encode the visual features (linear>relu>dropout)
            # 2d cnn -> subject 1536>1024 and 4096>1024
            self.feat_pool_ds = FeatPool(self.feat_dims[0:1], self.captioner_layers * 2 * self.captioner_size, self.drop_prob_lm,
                                         SQUEEZE=False)
            # 2d cnn and verb enc feat -> object .. 1536>512 and 512>512
            self.feat_pool_do = FeatPool([self.feat_dims[0], self.input_encoding_size], self.captioner_layers * self.captioner_size,
                                         self.drop_prob_lm, SQUEEZE=False)
            # 3d cnn and subject enc feat -> verb .. 4096>512 and 512>512
            self.feat_pool_dv = FeatPool([self.feat_dims[1], self.input_encoding_size], self.captioner_layers * self.captioner_size,
                                         self.drop_prob_lm, SQUEEZE=False)

            self.feat_pool_f2h = FeatPool([2 * self.captioner_size], self.captioner_layers * self.captioner_size, self.drop_prob_lm,
                                          SQUEEZE=False)

        self.feat_expander = FeatExpander(self.seq_per_img)

        self.video_encoding_size = self.num_feats * self.captioner_layers * self.captioner_size
        opt.video_encoding_size = self.video_encoding_size
        self.core = RNNUnit(opt)  # the caption generation rnn LSTM(512) with input size 2048

        if self.model_type == 'manet':
            self.manet = MANet(self.video_encoding_size, self.captioner_size, self.num_feats)

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.captioner_type == 'lstm':
            return (Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()),
                    Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()))
        else:
            return Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_())

    def _svo_step(self, feats, bfeats, pos=None, expand_feat=1):
        # pos is svo gt vocab indexs

        # encode the box features with linear->relu->dropout layer into query, key, value feats
        q_feats = self.bfeat_pool_q(bfeats)
        k_feats = self.bfeat_pool_k(bfeats)
        v_feats = self.bfeat_pool_v(bfeats)

        # use the query and key encodings to calculate self-attention weights
        b_att = torch.matmul(q_feats, k_feats.transpose(1, 2)) / math.sqrt(q_feats.shape[-1])
        b_att = F.softmax(b_att, dim=-1)

        # record the attention weights, used for visualisation purposes
        att_rec = b_att.data.cpu().numpy()
        self.attention_record = [np.mean(att_rec[i], axis=0) for i in range(len(att_rec))]

        # apply the box attention to the value encodings
        b_rep = torch.matmul(b_att, v_feats)

        # generalized bb representation (add a random extra box)
        gb_rep = torch.cat((b_rep, torch.rand(b_rep.shape[0], 1, b_rep.shape[-1]).cuda()), 1)

        ################### SUBJECT ###################
        # encode the subject features
        dec_feat_s = self.feat_pool_ds(feats[0:1])

        # use the 2D CNN and box encodings to generate attention for the global box reps
        s_att = torch.matmul(dec_feat_s, gb_rep.transpose(1, 2)) / math.sqrt(dec_feat_s.shape[-1])
        s_att = F.softmax(s_att, -1)
        s_rep = torch.matmul(s_att, gb_rep)  # apply the att
        if expand_feat:  # expand to seq_per_img length
            s_rep = self.feat_expander(s_rep.squeeze(1)).unsqueeze(1)  # [seq_im * batch, 1, d]

        # encode the subject rep with a hidden layer
        s_hid = self.feat_pool_f2h([s_rep])
        # encode to logits and apply softmax to get word probabilities
        s_out = F.log_softmax(self.logit(s_hid), dim=-1)
        # argmax the subject
        s_it = F.softmax(self.logit(s_hid), dim=-1).argmax(-1)

        ################### VERB ###################
        if expand_feat:
            feat_v_exp = self.feat_expander(feats[1].squeeze(1)).unsqueeze(1)  # [seq_im * batch, d]
        else:
            feat_v_exp = feats[1].clone()

        # encode the verb feature and the subject word embedding
        if self.training and pos is not None:
            dec_feat_v = self.feat_pool_dv([feat_v_exp, self.embed(pos[:, 0]).unsqueeze(1)])
        else:
            dec_feat_v = self.feat_pool_dv([feat_v_exp, self.embed(s_it.squeeze(1)).unsqueeze(1)])

        # encode the verb rep with a hidden layer
        v_hid = self.feat_pool_f2h([dec_feat_v])
        # encode to logits and apply softmax to get word probabilities
        v_out = F.log_softmax(self.logit(v_hid), dim=-1)
        # argmax the verb
        v_it = F.softmax(self.logit(v_hid), dim=-1).argmax(-1)

        ################### OBJECT ###################
        if expand_feat:
            feat_o_exp = self.feat_expander(feats[0].squeeze(1)).unsqueeze(1)
        else:
            feat_o_exp = feats[0].clone()

        # encode the object feature and the verb word embedding
        if self.training and pos is not None:
            dec_feat_o = self.feat_pool_do([feat_o_exp, self.embed(pos[:, 1]).unsqueeze(1)])
        else:
            dec_feat_o = self.feat_pool_do([feat_o_exp, self.embed(v_it.squeeze(1)).unsqueeze(1)])

        # calculate attention over the box features based on the word emb
        if expand_feat:
            o_att = torch.matmul(dec_feat_o, self.feat_expander(gb_rep).transpose(1, 2)) / math.sqrt(
                dec_feat_o.shape[-1])
        else:
            o_att = torch.matmul(dec_feat_o, gb_rep.transpose(1, 2)) / math.sqrt(dec_feat_o.shape[-1])

        o_att = F.softmax(o_att, -1)
        if expand_feat:
            o_rep = torch.matmul(o_att, self.feat_expander(gb_rep))
        else:
            o_rep = torch.matmul(o_att, gb_rep)

        # encode the object rep with a hidden layer
        o_hid = self.feat_pool_f2h([o_rep])
        # encode to logits and apply softmax to get word probabilities
        o_out = F.log_softmax(self.logit(o_hid), dim=-1)
        # argmax the object
        o_it = F.softmax(self.logit(o_hid), dim=-1).argmax(-1)

        return torch.cat((s_out, v_out, o_out), dim=1), torch.cat((s_it, v_it, o_it), dim=1)

    def forward(self, feats, bfeats, seq, pos):
        fc_feats = self.feat_pool(feats)
        fc_feats = self.feat_expander(fc_feats)

        # get the svo features and vocab indexs
        svo_out, svo_it = self._svo_step(feats, bfeats, pos)

        ########### RNN ###########
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:  # initially set xt as global feats
                xt = fc_feats
            else:
                # token_idx = 0 corresponding to the <BOS> token
                # (already encoded in seq)

                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, token_idx].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(outputs[-1].data)
                        sample_ind_tokens = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample_ind_tokens)
                        it = Variable(it, requires_grad=False)
                elif self.training and self.mixer_from > 0 and token_idx >= self.mixer_from:
                    prob_prev = torch.exp(outputs[-1].data)
                    it = torch.multinomial(prob_prev, 1).view(-1)
                    it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, token_idx].clone()

                if token_idx >= 1:
                    # store the seq and its logprobs
                    sample_seq.append(it.data)
                    logprobs = outputs[-1].gather(1, it.unsqueeze(1))
                    sample_logprobs.append(logprobs.view(-1))

                # break if all the sequences end, which requires EOS token = 0
                if it.data.sum() == 0:
                    break

                # get word embedding for the verb, concat with last pred word (it, rep as vocab index)
                if self.training:
                    if self.pass_all_svo:
                        lan_cont = self.embed(torch.cat((pos, it.unsqueeze(1)), 1))  # include sub and obj
                    else:
                        lan_cont = self.embed(torch.cat((pos[:, 1:2], it.unsqueeze(1)), 1))  # (bs * seq_per_img, 2, 512)
                else:
                    if self.pass_all_svo:
                        lan_cont = self.embed(torch.cat((svo_it, it.unsqueeze(1)), 1))  # include sub and obj
                    else:
                        lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))

                if self.pass_all_svo:
                    hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 4, state[0].shape[2])  # (bs * seq_per_img, 4, 512)
                else:
                    hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])  # (bs * seq_per_img, 2, 512)
                # calculate attention on encodings of the verb embedding and the RNN hidden state
                alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))  # (bs * seq_per_img, 2, 1)
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)  # (bs * seq_per_img, 1, 2)
                # apply the attention to the verb word embedding and the hidden layer emb
                xt = torch.matmul(alpha, lan_cont).squeeze(1)

            # generate next word
            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)

            # generate the word softmax
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                outputs.append(output)

        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_seq], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_logprobs], 1), \
               svo_out, svo_it, svo_out.gather(2, svo_it.unsqueeze(2)).squeeze(2)

    def sample(self, feats, bfeats, pos, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        expand_feat = opt.get('expand_feat', 0)

        svo_out, svo_it = self._svo_step(feats, bfeats, expand_feat=expand_feat)
        if beam_size > 1:
            return (*self.sample_beam(feats, bfeats, pos, opt)), svo_out, svo_it

        fc_feats = self.feat_pool(feats)
        if expand_feat == 1:
            fc_feats = self.feat_expander(fc_feats)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []

        unfinished = fc_feats.data.new(batch_size).fill_(1).byte()

        # -- if <image feature> is input at the first step, use index -1
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = self.seq_length - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats
            else:
                if token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(batch_size).long().fill_(self.bos_index)
                elif sample_max == 1:
                    # output here is a Tensor, because we don't use backprop
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                    # and flatten indices for downstream processing
                    it = it.view(-1).long()

                lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))
                hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
                alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                xt = torch.matmul(alpha, lan_cont).squeeze(1)

            if token_idx >= 1:
                unfinished = unfinished * (it > 0)

                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

                # requires EOS token = 0
                if unfinished.sum() == 0:
                    break

            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)

            logprobs = F.log_softmax(self.logit(output), dim=1)
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs],
                                                                      1), svo_out, svo_it

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats)
        svo_out, svo_it = self._svo_step(feats, bfeats, expand_feat=0)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            fc_feats_k = fc_feats[k].expand(beam_size, self.video_encoding_size)
            svo_it_k = svo_it[k].expand(beam_size, 3)
            # pos_k = pos[(k-1)*20].expand(beam_size, 3)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k
                elif token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(
                        beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at
                            # vix
                            new_state[state_ix][
                                0, vix] = state[state_ix][
                                0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[
                            token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    if self.pass_all_svo:
                        lan_cont = self.embed(
                            torch.cat((svo_it_k[:, 0:1], svo_it_k[:, 1:2], svo_it_k[:, 2:3], it.unsqueeze(1)), 1))
                        hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 4, state[0].shape[2])
                    else:
                        lan_cont = self.embed(torch.cat((svo_it_k[:, 1:2], it.unsqueeze(1)), 1))
                        hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
                    alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
                    alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                    xt = torch.matmul(alpha, lan_cont).squeeze(1)

                if token_idx >= 1:
                    state = new_state

                if self.model_type == 'standard':
                    output, state = self.core(xt, state)
                else:
                    if self.model_type == 'manet':
                        fc_feats_k = self.manet(fc_feats_k, state[0])
                    output, state = self.core(torch.cat([xt, fc_feats_k], 1), state)

                logprobs = F.log_softmax(self.logit(output), dim=1)

            # self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class RNN_DEC(nn.Module):
    """
    Attention over the 12 features, RNN cap gen attends every iteration
    """

    def __init__(self, opt):
        super(RNN_DEC, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)  # only used for manet
        self.input_feats = opt.input_features

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size
        self.ct_heads = 1

        self.input_encoder_layers = opt.input_encoder_layers
        self.input_encoder_heads = opt.input_encoder_heads
        self.input_encoder_size = opt.input_encoder_size

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.att_size = opt.att_size

        self.captioner_layers = opt.captioner_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # Visual Feature Attention
        assert self.att_size == self.captioner_size
        self.v2a_layer = nn.Linear(self.visual_encoding_size, self.att_size)
        self.h2a_layer = nn.Linear(self.captioner_size, self.att_size)
        self.att_layer = nn.Linear(self.att_size, 1)

        self.feat_enc = list()
        for in_dim in self.feat_dims:
            self.feat_enc.append(nn.Sequential(nn.Linear(in_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.feat_enc = nn.ModuleList(self.feat_enc)
        self.rf_encoder = nn.Sequential(nn.Linear(1024, self.visual_encoding_size-4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.rb_encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))

        # Transformer Encoder
        self.concept_encoder = None
        if self.input_encoder_layers > 0:
            concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size,
                                                               nhead=self.input_encoder_heads,
                                                               dim_feedforward=self.input_encoder_size,
                                                               dropout=self.drop_prob_lm)
            self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.input_encoder_layers)

        self.init_weights()

        self.feat_expander = FeatExpander(self.seq_per_img)

        opt.video_encoding_size = self.visual_encoding_size
        self.core = RNNUnit(opt)  # the caption generation rnn LSTM(512) with input size 2048

        if self.model_type == 'manet':
            self.manet = MANet(self.video_encoding_size, self.captioner_size, self.num_feats)

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.captioner_type == 'lstm':
            return (Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()),
                    Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()))
        else:
            return Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_())

    def forward(self, feats, bfeats, seq, pos):
        feats_enc = list()
        for i, feat in enumerate(feats):
            feats_enc.append(self.feat_enc[i](feat))
        rf_enc = self.rf_encoder(bfeats[0])
        rb_enc = self.rb_encoder(bfeats[1])
        r_enc = torch.cat((rf_enc, rb_enc), dim=-1)
        combined_enc = torch.cat(feats_enc + [r_enc], dim=1)

        feats_pass = []
        if 'i' in self.input_feats:
            feats_pass.append(combined_enc[:,0:1,:])
        if 'm' in self.input_feats:
            feats_pass.append(combined_enc[:,1:2,:])
        if 'r' in self.input_feats:
            feats_pass.append(combined_enc[:,-10:,:])
        if 'c' in self.input_feats and combined_enc.shape[1] == 13:
            feats_pass.append(combined_enc[:,2:3,:])
        combined_enc = torch.cat(feats_pass, dim=1)

        #### ENCODER ####
        if self.concept_encoder is not None:
            combined_enc = self.concept_encoder(combined_enc.permute(1, 0, 2)).permute(1, 0, 2)
        #### END ENCODER ####

        fc_feats = self.feat_expander(combined_enc)

        ########### RNN ###########
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:  # initially set xt as global feats
                xt = fc_feats
            else:
                # token_idx = 0 corresponding to the <BOS> token
                # (already encoded in seq)
                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, token_idx].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(outputs[-1].data)
                        sample_ind_tokens = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample_ind_tokens)
                        it = Variable(it, requires_grad=False)
                elif self.training and self.mixer_from > 0 and token_idx >= self.mixer_from:
                    prob_prev = torch.exp(outputs[-1].data)
                    it = torch.multinomial(prob_prev, 1).view(-1)
                    it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, token_idx].clone()

                if token_idx >= 1:
                    # store the seq and its logprobs
                    sample_seq.append(it.data)
                    logprobs = outputs[-1].gather(1, it.unsqueeze(1))
                    sample_logprobs.append(logprobs.view(-1))

                # break if all the sequences end, which requires EOS token = 0
                if it.data.sum() == 0:
                    break

                # set the new input word
                xt = self.embed(it)

                # calculate attention over visual features based on visual feats
                if self.captioner_layers > 1:
                    hid_cont = state[0][-1].unsqueeze(0).transpose(0, 1).expand(batch_size, 1, state[0].shape[2])  # (bs * seq_per_img, 1, 512)
                elif self.captioner_type in ['gru']:
                    hid_cont = state.transpose(0, 1).expand(batch_size, 1, state.shape[2])  # (bs * seq_per_img, 1, 512)
                else:
                    hid_cont = state[0].transpose(0, 1).expand(batch_size, 1, state[0].shape[2])  # (bs * seq_per_img, 1, 512)
                alpha = self.att_layer(torch.tanh(self.v2a_layer(fc_feats) + self.h2a_layer(hid_cont)))  # (bs * seq_per_img, 2, 1)
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)  # (bs * seq_per_img, 1, 2)
                att_vis_feats = torch.matmul(alpha, fc_feats).squeeze(1)

            # generate next word
            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    fc_feats = self.manet(att_vis_feats, state[0])
                output, state = self.core(torch.cat([xt, att_vis_feats], 1), state)

            # generate the word softmax
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                outputs.append(output)

        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_seq], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_logprobs], 1), \
               None, None, None

    def sample(self, feats, bfeats, pos, opt={}):
        beam_size = opt.get('beam_size', 1)

        if beam_size > 1:
            return ((*self.sample_beam(feats, bfeats, pos, opt)), None, None)
        else:
            return NotImplementedError

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        feats_enc = list()
        for i, feat in enumerate(feats):
            feats_enc.append(self.feat_enc[i](feat))
        rf_enc = self.rf_encoder(bfeats[0])
        rb_enc = self.rb_encoder(bfeats[1])
        r_enc = torch.cat((rf_enc, rb_enc), dim=-1)
        combined_enc = torch.cat(feats_enc + [r_enc], dim=1)

        #### ENCODER ####
        if self.concept_encoder is not None:
            combined_enc = self.concept_encoder(combined_enc.permute(1, 0, 2)).permute(1, 0, 2)
        #### END ENCODER ####

        fc_feats = combined_enc

        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            fc_feats_k = fc_feats[k].expand(beam_size, -1, -1)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k
                    # xt = torch.zeros((fc_feats_k.size(0), self.textual_encoding_size)).cuda()
                elif token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at
                            # vix
                            new_state[state_ix][
                                0, vix] = state[state_ix][
                                0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())

                xt = self.embed(it)
                if self.captioner_layers > 1:
                    hid_cont = state[0][-1].unsqueeze(0).transpose(0, 1).expand(beam_size, 1, state[0].shape[2])
                elif self.captioner_type in ['gru']:
                    hid_cont = state.transpose(0, 1).expand(beam_size, 1, state.shape[2])  # (bs * seq_per_img, 1, 512)
                else:
                    hid_cont = state[0].transpose(0, 1).expand(beam_size, 1, state[0].shape[2])

                alpha = self.att_layer(torch.tanh(self.v2a_layer(fc_feats_k) + self.h2a_layer(hid_cont)))
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                att_vis_feats = torch.matmul(alpha, fc_feats_k).squeeze(1)

                if token_idx >= 1:
                    state = new_state
                    if self.captioner_type in ['gru']:
                        state = new_state[0].unsqueeze(0)

                if self.model_type == 'standard':
                    output, state = self.core(xt, state)
                else:
                    if self.model_type == 'manet':
                        fc_feats_k = self.manet(att_vis_feats, state[0])
                    output, state = self.core(torch.cat([xt, att_vis_feats], 1), state)

                logprobs = F.log_softmax(self.logit(output), dim=1)

            # self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class TRF_DEC(nn.Module):
    """
    Just a transformer decoder, with optional input encoder
    """

    def __init__(self, opt):
        super(TRF_DEC, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.filter_type = opt.filter_type
        self.gt_concepts_while_training = opt.gt_concepts_while_training
        self.svo_length = opt.svo_length

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size

        self.input_encoder_layers = opt.input_encoder_layers
        self.input_encoder_heads = opt.input_encoder_heads
        self.input_encoder_size = opt.input_encoder_size

        self.grounder_layers = opt.grounder_layers
        self.grounder_heads = opt.grounder_heads
        self.grounder_size = opt.grounder_size

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.captioner_heads = opt.captioner_heads

        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # Input feature encoders
        self.feat_enc = list()
        for in_dim in self.feat_dims:
            self.feat_enc.append(nn.Sequential(nn.Linear(in_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.feat_enc = nn.ModuleList(self.feat_enc)
        self.rf_encoder = nn.Sequential(nn.Linear(1024, self.visual_encoding_size-4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.rb_encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))


        # Transformer input features Encoder
        self.concept_encoder = None
        if self.input_encoder_layers > 0:
            concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size,
                                                               nhead=self.input_encoder_heads,
                                                               dim_feedforward=self.input_encoder_size,
                                                               dropout=self.drop_prob_lm)
            self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.input_encoder_layers)

        # grounding module
        if self.filter_type in ['niuc']:
            # non-iterative
            self.feat_pool_q = nn.Sequential(nn.Linear(self.visual_encoding_size, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
            self.feat_pool_k = nn.Sequential(nn.Linear(self.visual_encoding_size, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
            self.feat_pool_v = nn.Sequential(nn.Linear(self.visual_encoding_size, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
            if self.grounder_layers == 1:
                self.feed_forward = nn.Sequential(nn.Linear(self.visual_encoding_size, self.textual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
            else:
                self.feed_forward = nn.ModuleList()
                self.feed_forward.append(nn.Linear(self.visual_encoding_size, self.grounder_size))
                self.feed_forward.append(nn.ReLU())
                self.feed_forward.append(nn.Dropout(self.drop_prob_lm))
                for _ in range(self.grounder_layers-2):
                    self.feed_forward.append(nn.Linear(self.visual_encoding_size, self.grounder_size))
                    self.feed_forward.append(nn.ReLU())
                    self.feed_forward.append(nn.Dropout(self.drop_prob_lm))
                self.feed_forward.append(nn.Linear(self.grounder_size, self.textual_encoding_size))
                self.feed_forward.append(nn.ReLU())
                self.feed_forward.append(nn.Dropout(self.drop_prob_lm))
        elif self.filter_type in ['iuc', 'ioc']:
            self.svo_pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.svo_length+1)
            # iterative
            concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.visual_encoding_size, nhead=self.grounder_heads,
                                                               dim_feedforward=self.grounder_size, dropout=self.drop_prob_lm)
            self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.grounder_layers)


        # Transformer Caption Decoder
        # encode word positions
        self.pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.seq_length)
        # Caption Prediction Module (a decoder on the global feats and k concept embeddings)
        caption_decoder_layer = nn.TransformerDecoderLayer(d_model=self.visual_encoding_size, nhead=self.captioner_heads,
                                                           dim_feedforward=self.captioner_size, dropout=self.drop_prob_lm)
        self.caption_decoder = nn.TransformerDecoder(caption_decoder_layer, num_layers=self.captioner_layers)


        self.feat_expander = FeatExpander(self.seq_per_img)
        opt.video_encoding_size = self.visual_encoding_size

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def non_iterative_grounder(self, feats):
        # embed the features for grounding attention
        q_feats = self.feat_pool_q(feats[:, 0:1]) # use img feat as query
        k_feats = self.feat_pool_k(feats)
        v_feats = self.feat_pool_v(feats)

        # use the query and key encodings to calculate self-attention weights
        att = torch.matmul(q_feats, k_feats.transpose(1, 2)) / math.sqrt(q_feats.shape[-1])
        att = F.softmax(att, dim=-1)

        # # record the attention weights, used for visualisation purposes
        # att_rec = att.data.cpu().numpy()
        # self.attention_record = [np.mean(att_rec[i], axis=0) for i in range(len(att_rec))]

        # apply the attention
        feats_ = torch.matmul(att, v_feats)

        # encode the features
        feats_ = self.feed_forward(feats_)

        # encode the features to concept-vocab size and sigmoid
        scores = self.logit(feats_).squeeze(1)

        concept_probs = F.sigmoid(scores)
        top_v, top_i = torch.topk(concept_probs, k=5)  # get the top 5 preds
        mask = top_v > .5  # mask threshold
        top_emb = self.embed(top_i)

        return scores, top_emb

    def iterative_grounder(self, feats, concepts_gt):

        if self.training and concepts_gt is not None:
            concepts_gt = torch.reshape(concepts_gt, (feats.shape[0], self.seq_per_img, -1))[:, 0]
            concepts_gt = F.pad(concepts_gt, (1, 0, 0, 0), "constant", self.bos_index)
            # concepts_gt = concepts_gt[:, :-1]
            concept_embeddings = self.embed(concepts_gt)
            concept_embeddings = concept_embeddings.permute(1, 0, 2) # change to (time, batch, channel)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, self.svo_length+1).cuda()  # +1 for <bos> token
            tgt_key_padding_mask = (concepts_gt == 0)  # create padding mask
            if self.svo_pos_encoder is not None:
                concept_embeddings = self.svo_pos_encoder(concept_embeddings)
            out = self.concept_decoder(concept_embeddings, feats, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)   # out is target shp
            out = out.permute(1, 0, 2)  # change back to (batch, concepts, channels)

            concept_probs = F.log_softmax(self.logit(out), dim=-1)[:, :-1]
            concept_probs_b = F.softmax(self.logit(out), dim=-1)[:, :-1]
            concept_probs_c = torch.max(concept_probs_b, dim=1)[0]
            concept_idxs = concept_probs_b.argmax(-1)

        else:  # auto-regressive prediction at inference

            concept_probs = torch.zeros((feats.size(0), self.svo_length, self.vocab_size)).cuda()
            concept_probs_b = torch.zeros((feats.size(0), self.svo_length, self.vocab_size)).cuda()
            concept_idxs = torch.zeros((feats.size(0), self.svo_length), dtype=torch.long).cuda()
            concept_idxs = F.pad(concept_idxs, (1, 0, 0, 0), "constant", self.bos_index)

            for i in range(1, self.svo_length+1):
                decoder_input = self.embed(concept_idxs[:, :i])

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, i).cuda()
                decoder_input = decoder_input.permute(1, 0, 2)
                if self.svo_pos_encoder is not None:
                    decoder_input = self.svo_pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.concept_decoder(decoder_input, feats, tgt_mask=tgt_mask)

                concept_idxs[:, i] = F.softmax(self.logit(decoder_output[-1]), dim=-1).argmax(-1)
                concept_probs_b[:, i - 1] = F.softmax(self.logit(decoder_output[-1]), dim=-1)
                concept_probs[:, i - 1] = F.log_softmax(self.logit(decoder_output[-1]), dim=-1)

            concept_idxs = concept_idxs[:, 1:]  # remove '<bos>'


        if self.filter_type in ['iuc']:
            raise NotImplementedError  # concept_probs_c dont work
            # concept_probs = torch.max(concept_probs_b, dim=1)[0]  # use max pred confs
            # concept_probs = torch.clamp(
            #     torch.sum(torch.nn.functional.one_hot(concept_idxs, num_classes=self.vocab_size), axis=1), 0, 1).type(
            #     torch.FloatTensor).cuda()  # use hard pred confs

        return concept_probs, concept_idxs

    def forward(self, feats, bfeats, seq, concepts):

        feats_enc = list()
        for i, feat in enumerate(feats):
            feats_enc.append(self.feat_enc[i](feat))
        rf_enc = self.rf_encoder(bfeats[0])
        rb_enc = self.rb_encoder(bfeats[1])
        r_enc = torch.cat((rf_enc, rb_enc), dim=-1)
        combined_enc = torch.cat(feats_enc + [r_enc], dim=1)

        #### ENCODER ####
        if self.concept_encoder is not None:
            combined_enc = self.concept_encoder(combined_enc.permute(1, 0, 2)).permute(1, 0, 2)
        #### END ENCODER ####

        #### GROUNDER ####
        concept_probs = None
        if self.filter_type in ['niuc', 'iuc', 'ioc']:
            if self.filter_type in ['niuc']:
                concept_probs, concept_seq = self.non_iterative_grounder(combined_enc)
            elif self.filter_type in ['iuc', 'ioc']:
                concept_probs, concept_seq = self.iterative_grounder(combined_enc, concepts_gt=concepts)
            else:
                raise NotImplementedError

            concept_probs = self.feat_expander(concept_probs)

            if self.gt_concepts_while_training and self.training:  # use gt concepts for cap gen
                combined_enc = torch.cat((combined_enc, self.embed(torch.reshape(concepts, (concept_seq.shape[0], self.seq_per_img, -1))[:, 0])), dim=1)
            else: # dont use gt for cap gen
                combined_enc = torch.cat((combined_enc, self.embed(concept_seq)), dim=1)
        #### END GROUNDER ####

        encoded_features = self.feat_expander(combined_enc).permute(1, 0, 2)

        caption_embeddings = self.embed(seq)  # emb indexs -> embeddings
        caption_embeddings = caption_embeddings.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.pos_encoder(caption_embeddings)  # add positional encoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, seq.size(-1)).cuda()  # create sequence mask
        tgt_key_padding_mask = (seq == 0)  # create padding mask

        # Run the decoder
        out = self.caption_decoder(caption_embeddings,
                                   encoded_features,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        out = out[:-1].permute(1, 0, 2)  # remove the last token and change back to (batch, concepts, channels)
        caption_probs = F.log_softmax(self.logit(self.dropout(out)), dim=-1)  # calc word probs
        caption_seq = F.softmax(self.logit(out), dim=-1).argmax(-1)  # get best word indexs

        caption_seq = seq[:, 1:]  # get gt caption (minus the BOS token) # todo check should we be outputting pred of gt?

        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return caption_probs, caption_seq, caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2), \
               concept_probs, concept_seq, concept_probs.gather(2, concept_seq.unsqueeze(2)).squeeze(2)

    def sample(self, feats, bfeats, pos, opt={}):
        beam_size = opt.get('beam_size', 1)
        expand_feat = opt.get('expand_feat', 0)

        feats_enc = list()
        for i, feat in enumerate(feats):
            feats_enc.append(self.feat_enc[i](feat))
        rf_enc = self.rf_encoder(bfeats[0])
        rb_enc = self.rb_encoder(bfeats[1])
        r_enc = torch.cat((rf_enc, rb_enc), dim=-1)
        combined_enc = torch.cat(feats_enc + [r_enc], dim=1)

        #### ENCODER ####
        if self.concept_encoder is not None:
            combined_enc = self.concept_encoder(combined_enc.permute(1, 0, 2)).permute(1, 0, 2)
        #### END ENCODER ####

        #### GROUNDER ####
        concept_probs = None
        if self.filter_type in ['niuc', 'iuc']:
            if self.filter_type in ['niuc']:
                concept_probs, top_emb = self.non_iterative_grounder(combined_enc)
            elif self.filter_type in ['iuc']:
                concept_probs, top_emb = self.iterative_grounder(combined_enc, pos)
            else:
                raise NotImplementedError

            combined_enc = torch.cat((combined_enc, top_emb), dim=1)  # todo use _feats, but maybe dont need to
        #### END GROUNDER ####

        if beam_size > 1:
            return ((*self.sample_beam(combined_enc, opt)), concept_probs)
        else:
            return NotImplementedError

    def sample_beam(self, encoded_features, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)

        batch_size = encoded_features.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            encoded_features_k = encoded_features[k].expand(beam_size, encoded_features.size(1), self.visual_encoding_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            its = torch.LongTensor(self.seq_length, beam_size).zero_().cuda()
            for token_idx in range(start_i, end_i):
                if token_idx == 0:  # input <bos>
                    it = encoded_features.data.new(beam_size).long().fill_(self.bos_index)  # [1,1,1,1,1]
                    its[token_idx] = it
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    its[token_idx] = it



                encoded_features_k = encoded_features_k.permute(1, 0, 2)  # change to (time, batch, channel)  # cat the vis feats and the svo
                decoder_input = self.embed(its[:token_idx+1])
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, token_idx + 1).cuda()

                decoder_input = self.pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.caption_decoder(decoder_input, encoded_features_k, tgt_mask=tgt_mask)

                output = decoder_output[-1]

                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class CONRNN(nn.Module):
    """
    A concepts captioning model
    """

    def __init__(self, opt):
        super(CONRNN, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size
        self.filter_encoder_layers = opt.filter_encoder_layers
        self.filter_encoder_size = opt.filter_encoder_size
        self.filter_decoder_layers = opt.filter_decoder_layers
        self.filter_decoder_size = opt.filter_decoder_size
        self.filter_encoder_heads = opt.filter_encoder_heads
        self.filter_decoder_heads = opt.filter_decoder_heads
        self.ct_heads = 1

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.att_size = opt.att_size

        assert self.filter_decoder_size == self.captioner_size == self.textual_encoding_size, print(self.filter_decoder_size, self.captioner_size, self.textual_encoding_size)
        assert self.filter_decoder_size == self.visual_encoding_size

        self.captioner_layers = opt.captioner_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.pass_all_svo = opt.pass_all_svo
        self.clamp_nearest = opt.clamp_concepts
        self.svo_length = opt.svo_length
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # (S,V,O,past-word + hid_state) attention
        self.l2a_layer = nn.Linear(self.textual_encoding_size, self.att_size)
        self.h2a_layer = nn.Linear(self.captioner_size, self.att_size)
        self.att_layer = nn.Linear(self.att_size, 1)

        self.init_weights()
        if self.model_type == 'standard':
            self.feat_pool = FeatPool(self.feat_dims[0:1], self.visual_encoding_size, self.drop_prob_lm)
        else:
            self.feat_pool = FeatPool(self.feat_dims, self.visual_encoding_size, self.drop_prob_lm)

            # encode the visual features (linear>relu>dropout)
            self.global_encoders = list()
            for feat_dim in self.feat_dims:
                self.global_encoders.append(nn.Sequential(nn.Linear(feat_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
            self.global_encoders = nn.ModuleList(self.global_encoders)

            # encode the box features 1024 -> 512 and 4 -> 512 (linear>relu>dropout)
            self.box_encoder = FeatPool(self.bfeat_dims, int(self.visual_encoding_size/2), self.drop_prob_lm, SQUEEZE=False)  # TODO replace with new box encoder

            concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size, nhead=self.filter_encoder_heads,
                                                               dim_feedforward=self.filter_encoder_size, dropout=self.drop_prob_lm)
            self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.filter_encoder_layers)

            concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_encoder_size, nhead=self.ct_heads,
                                                               dim_feedforward=self.filter_decoder_size, dropout=self.drop_prob_lm)
            self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.filter_decoder_heads)

            self.svo_pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm,
                                                  max_len=self.svo_length)

        self.feat_expander = FeatExpander(self.seq_per_img)

        opt.video_encoding_size = self.visual_encoding_size * self.num_feats
        self.core = RNNUnit(opt)  # the caption generation rnn LSTM(512) with input size 2048

        if self.model_type == 'manet':
            self.manet = MANet(self.video_encoding_size, self.captioner_size, self.num_feats)

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if self.captioner_type == 'lstm':
            return (Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()),
                    Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_()))
        else:
            return Variable(weight.new(self.captioner_layers, batch_size, self.captioner_size).zero_())

    def concept_generator(self, feats, bfeats, pos=None, expand_feat=1):

        # encode all of the global features into a single space
        encoded_global_feats = list()
        for global_feat, global_encoder in zip(feats, self.global_encoders):
            encoded_global_feats.append(global_encoder(global_feat))

        # encode the boxes into a similar space [cat(enc(1024>512), enc(4>512))]
        encoded_boxes = self.box_encoder(bfeats)

        # concat the box features to the global features
        visual_features = torch.cat(encoded_global_feats + [encoded_boxes], dim=-2)

        #### ENCODER ####
        encoded_features = self.concept_encoder(visual_features.permute(1, 0, 2))  # change to (time, batch, channel)
        if expand_feat:
            encoded_features = self.feat_expander(encoded_features.permute(1, 0, 2))  # change to (batch, time, channel)
            encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        #### END ENCODER ####

        #### DECODER ####
        # embed the concepts
        if self.training and pos is not None:
            pos = F.pad(pos, (1, 0, 0, 0), "constant", self.bos_index)
            pos = pos[:, :-1]
            concept_embeddings = self.embed(pos)
            concept_embeddings = concept_embeddings.permute(1,0,2) # change to (time, batch, channel)
            assert self.svo_length == concept_embeddings.size(0), print(self.svo_length, concept_embeddings.size(0))
            assert encoded_features.shape[-1] == concept_embeddings.shape[-1], print(encoded_features.shape[-1], concept_embeddings.shape[-1])
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, self.svo_length).cuda()
            tgt_key_padding_mask = (pos == 0)  # create padding mask
            concept_embeddings = self.svo_pos_encoder(concept_embeddings)
            out = self.concept_decoder(concept_embeddings, encoded_features, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)   # out is target shp

            # generate a concept
            out = out.permute(1, 0, 2)  # change back to (batch, concepts, channels)

            concept_prob = F.log_softmax(self.logit(out), dim=-1)
            concept_idx = F.softmax(self.logit(out), dim=-1).argmax(-1)

        else:  # auto-regressive prediction at inference
            concept_probs = torch.zeros((encoded_features.size(1), self.svo_length, self.vocab_size)).cuda()
            concept_idxs = torch.zeros((encoded_features.size(1), self.svo_length), dtype=torch.long).cuda()
            concept_idxs = F.pad(concept_idxs, (1, 0, 0, 0), "constant", self.bos_index)
            for i in range(1, self.svo_length+1):
                if self.clamp_nearest:
                    decoder_input = self.embed(concept_idxs[:, :i])
                else:
                    decoder_input = decoder_output
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, i).cuda()
                decoder_input = decoder_input.permute(1,0,2)
                decoder_input = self.svo_pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.concept_decoder(decoder_input, encoded_features, tgt_mask=tgt_mask)

                concept_idxs[:, i] = F.softmax(self.logit(decoder_output[-1]), dim=-1).argmax(-1)
                concept_probs[:, i-1] = F.log_softmax(self.logit(decoder_output[-1]), dim=-1)

            concept_prob = concept_probs
            concept_idx = concept_idxs[:, 1:]
        #### END DECODER ####

        return concept_prob, concept_idx

    def forward(self, feats, bfeats, seq, pos):
        fc_feats = self.feat_pool(feats)
        fc_feats = self.feat_expander(fc_feats)

        # get the svo features and vocab indexs
        svo_out, svo_it = self.concept_generator(feats, bfeats, pos)

        ########### RNN ###########
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:  # initially set xt as global feats
                xt = fc_feats  # todo was (544,1024) could encode 1024->textual_encoding_size
                # xt = torch.zeros((fc_feats.size(0), self.textual_encoding_size)).cuda()  # todo init set as zeros (544,512)
            else:
                # token_idx = 0 corresponding to the <BOS> token
                # (already encoded in seq)

                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, token_idx].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(outputs[-1].data)
                        sample_ind_tokens = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample_ind_tokens)
                        it = Variable(it, requires_grad=False)
                elif self.training and self.mixer_from > 0 and token_idx >= self.mixer_from:
                    prob_prev = torch.exp(outputs[-1].data)
                    it = torch.multinomial(prob_prev, 1).view(-1)
                    it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, token_idx].clone()

                if token_idx >= 1:
                    # store the seq and its logprobs
                    sample_seq.append(it.data)
                    logprobs = outputs[-1].gather(1, it.unsqueeze(1))
                    sample_logprobs.append(logprobs.view(-1))

                # break if all the sequences end, which requires EOS token = 0
                if it.data.sum() == 0:
                    break

                # get word embedding for the verb, concat with last pred word (it, rep as vocab index)
                if self.training:
                    if self.pass_all_svo:
                        lan_cont = self.embed(torch.cat((pos, it.unsqueeze(1)),1))
                    else:
                        assert self.svo_length == 3
                        lan_cont = self.embed(torch.cat((pos[:, 1:2], it.unsqueeze(1)), 1))  # (bs * seq_per_img, 2, 512)
                else:
                    if self.pass_all_svo:
                        lan_cont = self.embed(torch.cat((svo_it, it.unsqueeze(1)),1))
                    else:
                        assert self.svo_length == 3
                        lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))

                if self.pass_all_svo:
                    hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], self.svo_length+1, state[0].shape[2])  # (bs * seq_per_img, 4, 512)
                else:
                    assert self.svo_length == 3
                    hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])  # (bs * seq_per_img, 2, 512)
                # calculate attention on encodings of the verb embedding and the RNN hidden state
                alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))  # (bs * seq_per_img, 2, 1)
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)  # (bs * seq_per_img, 1, 2)
                # apply the attention to the verb word embedding and the hidden layer emb
                xt = torch.matmul(alpha, lan_cont).squeeze(1)

            # generate next word
            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)

            # generate the word softmax
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                outputs.append(output)

        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_seq], 1), \
               torch.cat([_.unsqueeze(1) for _ in sample_logprobs], 1), \
               svo_out, svo_it, svo_out.gather(2, svo_it.unsqueeze(2)).squeeze(2)

    def sample(self, feats, bfeats, pos, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        expand_feat = opt.get('expand_feat', 0)

        svo_out, svo_it = self.concept_generator(feats, bfeats, expand_feat=expand_feat)
        if beam_size > 1:
            return ((*self.sample_beam(feats, bfeats, pos, opt)), svo_out, svo_it)

        fc_feats = self.feat_pool(feats)
        if expand_feat == 1:
            fc_feats = self.feat_expander(fc_feats)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []

        unfinished = fc_feats.data.new(batch_size).fill_(1).byte()

        # -- if <image feature> is input at the first step, use index -1
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = self.seq_length - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats  # todo was (544,1024) could encode 1024->textual_encoding_size
                # xt = torch.zeros((fc_feats.size(0), self.textual_encoding_size)).cuda()  # todo init set as zeros (544,512)
            else:
                if token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(batch_size).long().fill_(self.bos_index)
                elif sample_max == 1:
                    # output here is a Tensor, because we don't use backprop
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                    # and flatten indices for downstream processing
                    it = it.view(-1).long()

                lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))
                hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
                alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                xt = torch.matmul(alpha, lan_cont).squeeze(1)

            if token_idx >= 1:
                unfinished = unfinished * (it > 0)

                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

                # requires EOS token = 0
                if unfinished.sum() == 0:
                    break

            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)

            logprobs = F.log_softmax(self.logit(output), dim=1)
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), svo_out, svo_it

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats)
        svo_out, svo_it = self.concept_generator(feats, bfeats, expand_feat=0)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            fc_feats_k = fc_feats[k].expand(beam_size, self.visual_encoding_size*self.num_feats)
            svo_it_k = svo_it[k].expand(beam_size, self.svo_length)
            # pos_k = pos[(k - 1) * 20].expand(beam_size, 3)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k  # todo was (544,1024) could encode 1024->textual_encoding_size
                    xt = torch.zeros((fc_feats_k.size(0), self.textual_encoding_size)).cuda()  # todo init set as zeros (544,512)
                elif token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at
                            # vix
                            new_state[state_ix][
                                0, vix] = state[state_ix][
                                0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    if self.pass_all_svo:
                        lan_cont = self.embed(
                            torch.cat((svo_it_k[:, 0:1], svo_it_k[:, 1:2], svo_it_k[:, 2:3], it.unsqueeze(1)), 1))
                        hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 4, state[0].shape[2])
                    else:
                        lan_cont = self.embed(torch.cat((svo_it_k[:, 1:2], it.unsqueeze(1)), 1))
                        hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
                    alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
                    alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                    xt = torch.matmul(alpha, lan_cont).squeeze(1)

                if token_idx >= 1:
                    state = new_state

                if self.model_type == 'standard':
                    output, state = self.core(xt, state)
                else:
                    if self.model_type == 'manet':
                        fc_feats_k = self.manet(fc_feats_k, state[0])
                    output, state = self.core(torch.cat([xt, fc_feats_k], 1), state)

                logprobs = F.log_softmax(self.logit(output), dim=1)

            # self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class SINTRA(nn.Module):
    """
    A concepts captioning model, with direct connection, no svo/concepts filtering
    """

    def __init__(self, opt):
        super(SINTRA, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size
        self.filter_encoder_layers = opt.filter_encoder_layers
        self.filter_encoder_size = opt.filter_encoder_size
        self.filter_decoder_layers = opt.filter_decoder_layers
        self.filter_decoder_size = opt.filter_decoder_size
        self.filter_encoder_heads = opt.filter_encoder_heads
        self.filter_decoder_heads = opt.filter_decoder_heads

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.captioner_heads = opt.captioner_heads

        assert self.filter_decoder_size == self.captioner_size == self.textual_encoding_size, print(self.filter_decoder_size, self.captioner_size, self.textual_encoding_size)
        assert self.filter_decoder_size == self.visual_encoding_size

        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.pass_all_svo = opt.pass_all_svo
        self.clamp_nearest = opt.clamp_concepts
        self.svo_length =  3 # opt.svo_length # todo fix
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.feat_pool = FeatPool(self.feat_dims, self.visual_encoding_size, self.drop_prob_lm)

        # encode the visual features (linear>relu>dropout)
        self.global_encoders = list()
        for feat_dim in self.feat_dims:
            self.global_encoders.append(nn.Sequential(nn.Linear(feat_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.global_encoders = nn.ModuleList(self.global_encoders)

        # encode the box features 1024 -> 512 and 4 -> 512 (linear>relu>dropout)
        self.box_encoder = FeatPool(self.bfeat_dims, int(self.visual_encoding_size/2), self.drop_prob_lm, SQUEEZE=False)  # TODO replace with new box encoder

        # encode word positions
        self.pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.seq_length)

        # Concept Prediction module (visual feature encoder > k concepts decoder)
        concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size, nhead=self.filter_encoder_heads,
                                                           dim_feedforward=self.filter_encoder_size, dropout=self.drop_prob_lm)
        self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.filter_encoder_layers)
        #
        # concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_encoder_size, nhead=self.filter_decoder_heads,
        #                                                    dim_feedforward=self.filter_decoder_size, dropout=self.drop_prob_lm)
        # self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.filter_decoder_layers)

        # Caption Prediction Module (a decoder on the global feats and k concept embeddings)
        caption_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_encoder_size, nhead=self.captioner_heads,
                                                           dim_feedforward=self.captioner_size, dropout=self.drop_prob_lm)
        self.caption_decoder = nn.TransformerDecoder(caption_decoder_layer, num_layers=self.captioner_layers)

        self.feat_expander = FeatExpander(self.seq_per_img)

        # opt.video_encoding_size = self.visual_encoding_size * self.num_feats

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def visual_encoder(self, feats, bfeats, expand_feat=1):

        # encode all of the global features into a single space
        encoded_global_feats = list()
        for global_feat, global_encoder in zip(feats, self.global_encoders):
            encoded_global_feats.append(global_encoder(global_feat))

        # encode the boxes into a similar space [cat(enc(1024>512), enc(4>512))]
        encoded_boxes = self.box_encoder(bfeats)

        # concat the box features to the global features
        visual_features = torch.cat(encoded_global_feats + [encoded_boxes], dim=-2)

        #### ENCODER ####
        encoded_features = self.concept_encoder(visual_features.permute(1, 0, 2))  # change to (time, batch, channel)
        if expand_feat:
            encoded_features = self.feat_expander(encoded_features.permute(1, 0, 2))  # change to (batch, time, channel)
            encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        #### END ENCODER ####

        return encoded_features

    # def concept_decoder_forward(self, encoded_features, pos):
    #     #### DECODER ####
    #     # embed the concepts
    #     if self.training and pos is not None:
    #         concept_embeddings = self.embed(pos)
    #         concept_embeddings = concept_embeddings.permute(1,0,2) # change to (time, batch, channel)
    #         assert self.svo_length == concept_embeddings.size(0), print(self.svo_length, concept_embeddings.size(0))
    #         concept_embeddings = F.pad(concept_embeddings, (0, 0, 0, 0, 1, 0), "constant", 0)  # prepend with zero pad for first word 'our <bos> persay'
    #         concept_embeddings = concept_embeddings[:-1]  # remove the last concept
    #         assert encoded_features.shape[-1] == concept_embeddings.shape[-1], print(encoded_features.shape[-1], concept_embeddings.shape[-1])
    #         tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, self.svo_length).cuda()
    #         out = self.concept_decoder(concept_embeddings, encoded_features, tgt_mask=tgt_mask)
    #         out = out.permute(1, 0, 2)  # change back to (batch, concepts, channels)
    #
    #         concept_probs = F.log_softmax(self.logit(out), dim=-1)
    #         concept_idxs = F.softmax(self.logit(out), dim=-1).argmax(-1)
    #
    #     else:  # auto-regressive prediction at inference
    #         concept_embeddings = torch.zeros((self.svo_length+1, encoded_features.size(1), self.textual_encoding_size)).cuda()
    #         concept_probs = torch.zeros((encoded_features.size(1), self.svo_length, self.vocab_size)).cuda()
    #         concept_idxs = torch.zeros((encoded_features.size(1), self.svo_length), dtype=torch.long).cuda()
    #         for i in range(self.svo_length):
    #             decoder_input = concept_embeddings[:i+1]
    #             tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, i+1).cuda()
    #             decoder_output = self.concept_decoder(decoder_input, encoded_features, tgt_mask=tgt_mask)
    #
    #             concept_idxs[:, i] = F.softmax(self.logit(decoder_output[-1]), dim=-1).argmax(-1)
    #             concept_probs[:, i] = F.log_softmax(self.logit(decoder_output[-1]), dim=-1)
    #             if self.clamp_nearest:
    #                 # find nearest actual concept
    #                 concept_embeddings[i+1] = self.embed(concept_idxs[:, i])
    #             else:
    #                 # pass raw decoder as input
    #                 concept_embeddings[i+1] = decoder_output[-1]
    #         concept_embeddings = concept_embeddings[1:].permute(1, 0 ,2)
    #     #### END DECODER ####
    #
    #     return concept_probs, concept_idxs, concept_embeddings

    def forward(self, gfeats, bfeats, seq, pos):

        encoded_features = self.visual_encoder(gfeats, bfeats)

        caption_embeddings = self.embed(seq)  # emb indexs -> embeddings
        caption_embeddings = caption_embeddings.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.pos_encoder(caption_embeddings)  # add positional encoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, seq.size(-1)).cuda()  # create sequence mask
        tgt_key_padding_mask = (seq == 0)  # create padding mask

        # Run the decoder
        out = self.caption_decoder(caption_embeddings,
                                   encoded_features,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        out = out[:-1].permute(1, 0, 2)  # remove the last token and change back to (batch, concepts, channels)
        caption_probs = F.log_softmax(self.logit(self.dropout(out)), dim=-1)  # calc word probs
        caption_idxs = F.softmax(self.logit(out), dim=-1).argmax(-1)  # get best word indexs (not used)

        caption_seq = seq[:, 1:]  # get gt caption (minus the BOS token)
        caption_logprobs = caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2)

        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return caption_probs, caption_seq, caption_logprobs, None, None, None

    def sample(self, feats, bfeats, pos, opt={}):
        beam_size = opt.get('beam_size', 1)
        expand_feat = opt.get('expand_feat', 0)

        if beam_size > 1:
            return ((*self.sample_beam(feats, bfeats, pos, opt)), None, None)
        else:
            return NotImplementedError
        # fc_feats = self.feat_pool(feats)
        # if expand_feat == 1:
        #     fc_feats = self.feat_expander(fc_feats)
        # batch_size = fc_feats.size(0)
        # state = self.init_hidden(batch_size)
        #
        # seq = []
        # seqLogprobs = []
        #
        # unfinished = fc_feats.data.new(batch_size).fill_(1).byte()
        #
        # # -- if <image feature> is input at the first step, use index -1
        # start_i = -1 if self.model_type == 'standard' else 0
        # end_i = self.seq_length - 1
        #
        # for token_idx in range(start_i, end_i):
        #     if token_idx == -1:
        #         xt = fc_feats  # todo was (544,1024) could encode 1024->textual_encoding_size
        #         # xt = torch.zeros((fc_feats.size(0), self.textual_encoding_size)).cuda()  # todo init set as zeros (544,512)
        #     else:
        #         if token_idx == 0:  # input <bos>
        #             it = fc_feats.data.new(batch_size).long().fill_(self.bos_index)
        #         elif sample_max == 1:
        #             # output here is a Tensor, because we don't use backprop
        #             sampleLogprobs, it = torch.max(logprobs.data, 1)
        #             it = it.view(-1).long()
        #         else:
        #             if temperature == 1.0:
        #                 # fetch prev distribution: shape Nx(M+1)
        #                 prob_prev = torch.exp(logprobs.data).cpu()
        #             else:
        #                 # scale logprobs by temperature
        #                 prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
        #             it = torch.multinomial(prob_prev, 1).cuda()
        #             # gather the logprobs at sampled positions
        #             sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
        #             # and flatten indices for downstream processing
        #             it = it.view(-1).long()
        #
        #         lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))
        #         hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
        #         alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
        #         alpha = F.softmax(alpha, dim=1).transpose(1, 2)
        #         xt = torch.matmul(alpha, lan_cont).squeeze(1)
        #
        #     if token_idx >= 1:
        #         unfinished = unfinished * (it > 0)
        #
        #         it = it * unfinished.type_as(it)
        #         seq.append(it)
        #         seqLogprobs.append(sampleLogprobs.view(-1))
        #
        #         # requires EOS token = 0
        #         if unfinished.sum() == 0:
        #             break
        #
        #     if self.model_type == 'standard':
        #         output, state = self.core(xt, state)
        #     else:
        #         if self.model_type == 'manet':
        #             fc_feats = self.manet(fc_feats, state[0])
        #         output, state = self.core(torch.cat([xt, fc_feats], 1), state)
        #
        #     logprobs = F.log_softmax(self.logit(output), dim=1)
        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), svo_out, svo_it

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        # fc_feats = self.feat_pool(feats, stack=True)
        encoded_features = self.visual_encoder(feats, bfeats, expand_feat=0).permute(1,0,2)
        batch_size = encoded_features.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            encoded_features_k = encoded_features[k].expand(beam_size, encoded_features.size(1), self.visual_encoding_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            its = torch.LongTensor(self.seq_length, beam_size).zero_().cuda()
            for token_idx in range(start_i, end_i):
                if token_idx == 0:  # input <bos>
                    it = encoded_features.data.new(beam_size).long().fill_(self.bos_index)  # [1,1,1,1,1]
                    its[token_idx] = it
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    its[token_idx] = it



                encoded_features_k = encoded_features_k.permute(1, 0, 2)  # change to (time, batch, channel)  # cat the vis feats and the svo
                decoder_input = self.embed(its[:token_idx+1])
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, token_idx + 1).cuda()

                decoder_input = self.pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.caption_decoder(decoder_input, encoded_features_k, tgt_mask=tgt_mask)

                output = decoder_output[-1]

                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class CONTRA(nn.Module):
    """
    A concepts captioning model
    """

    def __init__(self, opt):
        super(CONTRA, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size
        self.filter_encoder_layers = opt.filter_encoder_layers
        self.filter_encoder_size = opt.filter_encoder_size
        self.filter_decoder_layers = opt.filter_decoder_layers
        self.filter_decoder_size = opt.filter_decoder_size
        self.filter_encoder_heads = opt.filter_encoder_heads
        self.filter_decoder_heads = opt.filter_decoder_heads

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.captioner_heads = opt.captioner_heads

        # assert self.filter_decoder_size == self.captioner_size == self.textual_encoding_size, print(self.filter_decoder_size, self.captioner_size, self.textual_encoding_size)
        # assert self.filter_decoder_size == self.visual_encoding_size

        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.pass_all_svo = opt.pass_all_svo
        self.clamp_nearest = opt.clamp_concepts
        self.svo_length =  opt.svo_length
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.feat_pool = FeatPool(self.feat_dims, self.visual_encoding_size, self.drop_prob_lm)

        # encode the visual features (linear>relu>dropout)
        self.global_encoders = list()
        for feat_dim in self.feat_dims:
            self.global_encoders.append(nn.Sequential(nn.Linear(feat_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.global_encoders = nn.ModuleList(self.global_encoders)

        # encode the box features 1024 -> 512 and 4 -> 512 (linear>relu>dropout)
        self.box_encoder = FeatPool(self.bfeat_dims, int(self.visual_encoding_size/2), self.drop_prob_lm, SQUEEZE=False)  # TODO replace with new box encoder

        # encode word positions
        self.pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.seq_length)
        self.svo_pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.svo_length)

        # Concept Prediction module (visual feature encoder > k concepts decoder)
        concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size, nhead=self.filter_encoder_heads,
                                                           dim_feedforward=self.filter_encoder_size, dropout=self.drop_prob_lm)
        self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.filter_encoder_layers)

        concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_encoder_size, nhead=self.filter_decoder_heads,
                                                           dim_feedforward=self.filter_decoder_size, dropout=self.drop_prob_lm)
        self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.filter_decoder_layers)

        # Caption Prediction Module (a decoder on the global feats and k concept embeddings)
        caption_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_decoder_size, nhead=self.captioner_heads,
                                                           dim_feedforward=self.captioner_size, dropout=self.drop_prob_lm)
        self.caption_decoder = nn.TransformerDecoder(caption_decoder_layer, num_layers=self.captioner_layers)

        self.feat_expander = FeatExpander(self.seq_per_img)

        # opt.video_encoding_size = self.visual_encoding_size * self.num_feats

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def concept_generator_cap(self, feats, bfeats, pos=None, expand_feat=1):

        # encode all of the global features into a single space
        encoded_global_feats = list()
        for global_feat, global_encoder in zip(feats, self.global_encoders):
            encoded_global_feats.append(global_encoder(global_feat))

        # encode the boxes into a similar space [cat(enc(1024>512), enc(4>512))]
        encoded_boxes = self.box_encoder(bfeats)

        # concat the box features to the global features
        visual_features = torch.cat(encoded_global_feats + [encoded_boxes], dim=-2)

        #### ENCODER ####
        encoded_features = self.concept_encoder(visual_features.permute(1, 0, 2))  # change to (time, batch, channel)
        if expand_feat:
            encoded_features = self.feat_expander(encoded_features.permute(1, 0, 2))  # change to (batch, time, channel)
            encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        #### END ENCODER ####

        #### DECODER ####
        # embed the concepts
        if self.training and pos is not None:
            pos = F.pad(pos, (1, 0, 0, 0), "constant", self.bos_index)
            pos = pos[:, :-1]
            concept_embeddings = self.embed(pos)
            concept_embeddings = concept_embeddings.permute(1,0,2) # change to (time, batch, channel)
            assert self.svo_length == concept_embeddings.size(0), print(self.svo_length, concept_embeddings.size(0))
            assert encoded_features.shape[-1] == concept_embeddings.shape[-1], print(encoded_features.shape[-1], concept_embeddings.shape[-1])
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, self.svo_length).cuda()
            tgt_key_padding_mask = (pos == 0)  # create padding mask
            if self.svo_pos_encoder is not None:
                concept_embeddings = self.svo_pos_encoder(concept_embeddings)
            out = self.concept_decoder(concept_embeddings, encoded_features, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)   # out is target shp
            out = out.permute(1, 0, 2)  # change back to (batch, concepts, channels)

            concept_probs = F.log_softmax(self.logit(out), dim=-1)
            concept_idxs = F.softmax(self.logit(out), dim=-1).argmax(-1)


        else:  # auto-regressive prediction at inference
            concept_probs = torch.zeros((encoded_features.size(1), self.svo_length, self.vocab_size)).cuda()
            concept_idxs = torch.zeros((encoded_features.size(1), self.svo_length), dtype=torch.long).cuda()
            concept_idxs = F.pad(concept_idxs, (1, 0, 0, 0), "constant", self.bos_index)

            for i in range(1, self.svo_length+1):
                if self.clamp_nearest:
                    decoder_input = self.embed(concept_idxs[:, :i])
                else:
                    decoder_input = decoder_output

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, i).cuda()
                decoder_input = decoder_input.permute(1, 0, 2)
                if self.svo_pos_encoder is not None:
                    decoder_input = self.svo_pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.concept_decoder(decoder_input, encoded_features, tgt_mask=tgt_mask)

                concept_idxs[:, i] = F.softmax(self.logit(decoder_output[-1]), dim=-1).argmax(-1)
                concept_probs[:, i - 1] = F.log_softmax(self.logit(decoder_output[-1]), dim=-1)

            concept_idxs = concept_idxs[:, 1:]

        #### END DECODER ####

        return concept_probs, concept_idxs

    def concept_generator(self, feats, bfeats, expand_feat=1):

        # encode all of the global features into a single space
        encoded_global_feats = list()
        for global_feat, global_encoder in zip(feats, self.global_encoders):
            encoded_global_feats.append(global_encoder(global_feat))

        # encode the boxes into a similar space [cat(enc(1024>512), enc(4>512))]
        encoded_boxes = self.box_encoder(bfeats)

        # concat the box features to the global features
        visual_features = torch.cat(encoded_global_feats + [encoded_boxes], dim=-2)

        #### ENCODER ####
        encoded_features = self.concept_encoder(visual_features.permute(1, 0, 2))  # change to (time, batch, channel)
        if expand_feat:
            encoded_features = self.feat_expander(encoded_features.permute(1, 0, 2))  # change to (batch, time, channel)
            encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        #### END ENCODER ####

        #### DECODER ####        # init = torch.ones(one_hot.shape).unsqueeze(0).cuda()
        init = torch.ones((1,encoded_features.shape[1],self.filter_decoder_size)).cuda()
        out = self.concept_decoder(init, encoded_features)
        conf = torch.sigmoid(self.logit(out.squeeze()))
        #### END DECODER ####

        return out, conf

    def forward(self, gfeats, bfeats, seq, pos):

        # get the svo features and vocab indexs
        svo_out, svo_it = self.concept_generator_cap(gfeats, bfeats, pos)
        conc_emb, conc_conf = self.concept_generator(gfeats, bfeats)
        one_hot = torch.clamp(torch.sum(torch.nn.functional.one_hot(pos, num_classes=self.vocab_size), axis=1), 0, 1)
        one_hot[:, 0] = 0  # make the padding index 0

        if self.training:
            svo_embs = self.embed(pos)
        else:
            svo_embs = self.embed(svo_it)
        if not self.pass_all_svo:
            svo_embs = svo_embs[:, 1:2]

        svo_embs = svo_embs.permute(1, 0, 2)  # change to (time, batch, channel)

        fc_feats = self.feat_pool(gfeats, stack=True)
        fc_feats = self.feat_expander(fc_feats)
        fc_feats = fc_feats.permute(1, 0, 2)  # change to (time, batch, channel)
        encoded_features = torch.cat([fc_feats, svo_embs], dim=0)  # cat the vis feats and the svo

        caption_embeddings = self.embed(seq)  # emb indexs -> embeddings
        caption_embeddings = caption_embeddings.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.pos_encoder(caption_embeddings)  # add positional encoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, seq.size(-1)).cuda()  # create sequence mask
        tgt_key_padding_mask = (seq == 0)  # create padding mask

        # Run the decoder
        out = self.caption_decoder(caption_embeddings,
                                   encoded_features,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        out = out[:-1].permute(1, 0, 2)  # remove the last token and change back to (batch, concepts, channels)
        caption_probs = F.log_softmax(self.logit(self.dropout(out)), dim=-1)  # calc word probs
        caption_idxs = F.softmax(self.logit(out), dim=-1).argmax(-1)  # get best word indexs (not used)

        caption_seq = seq[:, 1:]  # get gt caption (minus the BOS token)
        caption_logprobs = caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2)

        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return caption_probs, caption_seq, caption_logprobs, svo_out, svo_it, svo_out.gather(2, svo_it.unsqueeze(2)).squeeze(2)

    def sample(self, feats, bfeats, pos, opt={}):
        beam_size = opt.get('beam_size', 1)
        expand_feat = opt.get('expand_feat', 0)

        svo_out, svo_it = self.concept_generator_cap(feats, bfeats, expand_feat=expand_feat)
        if beam_size > 1:
            return ((*self.sample_beam(feats, bfeats, pos, opt)), svo_out, svo_it)
        else:
            return NotImplementedError
        # fc_feats = self.feat_pool(feats)
        # if expand_feat == 1:
        #     fc_feats = self.feat_expander(fc_feats)
        # batch_size = fc_feats.size(0)
        # state = self.init_hidden(batch_size)
        #
        # seq = []
        # seqLogprobs = []
        #
        # unfinished = fc_feats.data.new(batch_size).fill_(1).byte()
        #
        # # -- if <image feature> is input at the first step, use index -1
        # start_i = -1 if self.model_type == 'standard' else 0
        # end_i = self.seq_length - 1
        #
        # for token_idx in range(start_i, end_i):
        #     if token_idx == -1:
        #         xt = fc_feats  # todo was (544,1024) could encode 1024->textual_encoding_size
        #         # xt = torch.zeros((fc_feats.size(0), self.textual_encoding_size)).cuda()  # todo init set as zeros (544,512)
        #     else:
        #         if token_idx == 0:  # input <bos>
        #             it = fc_feats.data.new(batch_size).long().fill_(self.bos_index)
        #         elif sample_max == 1:
        #             # output here is a Tensor, because we don't use backprop
        #             sampleLogprobs, it = torch.max(logprobs.data, 1)
        #             it = it.view(-1).long()
        #         else:
        #             if temperature == 1.0:
        #                 # fetch prev distribution: shape Nx(M+1)
        #                 prob_prev = torch.exp(logprobs.data).cpu()
        #             else:
        #                 # scale logprobs by temperature
        #                 prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
        #             it = torch.multinomial(prob_prev, 1).cuda()
        #             # gather the logprobs at sampled positions
        #             sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
        #             # and flatten indices for downstream processing
        #             it = it.view(-1).long()
        #
        #         lan_cont = self.embed(torch.cat((svo_it[:, 1:2], it.unsqueeze(1)), 1))
        #         hid_cont = state[0].transpose(0, 1).expand(lan_cont.shape[0], 2, state[0].shape[2])
        #         alpha = self.att_layer(torch.tanh(self.l2a_layer(lan_cont) + self.h2a_layer(hid_cont)))
        #         alpha = F.softmax(alpha, dim=1).transpose(1, 2)
        #         xt = torch.matmul(alpha, lan_cont).squeeze(1)
        #
        #     if token_idx >= 1:
        #         unfinished = unfinished * (it > 0)
        #
        #         it = it * unfinished.type_as(it)
        #         seq.append(it)
        #         seqLogprobs.append(sampleLogprobs.view(-1))
        #
        #         # requires EOS token = 0
        #         if unfinished.sum() == 0:
        #             break
        #
        #     if self.model_type == 'standard':
        #         output, state = self.core(xt, state)
        #     else:
        #         if self.model_type == 'manet':
        #             fc_feats = self.manet(fc_feats, state[0])
        #         output, state = self.core(torch.cat([xt, fc_feats], 1), state)
        #
        #     logprobs = F.log_softmax(self.logit(output), dim=1)
        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), svo_out, svo_it

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats, stack=True)
        svo_out, svo_it = self.concept_generator_cap(feats, bfeats, expand_feat=0)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            fc_feats_k = fc_feats[k].expand(beam_size, self.num_feats, self.visual_encoding_size)
            svo_it_k = svo_it[k].expand(beam_size, self.svo_length)
            # pos_k = pos[(k - 1) * 20].expand(beam_size, 3)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            its = torch.LongTensor(self.seq_length, beam_size).zero_().cuda()
            for token_idx in range(start_i, end_i):
                if token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(self.bos_index)  # [1,1,1,1,1]
                    its[token_idx] = it
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    its[token_idx] = it

                svo_embs = self.embed(svo_it_k)

                if not self.pass_all_svo:
                    svo_embs = svo_embs[:, 1:2]

                encoded_features = torch.cat([fc_feats_k, svo_embs], dim=1).permute(1, 0, 2)  # change to (time, batch, channel)  # cat the vis feats and the svo
                decoder_input = self.embed(its[:token_idx+1])
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, token_idx + 1).cuda()

                decoder_input = self.pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.caption_decoder(decoder_input, encoded_features, tgt_mask=tgt_mask)

                output = decoder_output[-1]

                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class CONTRAB(nn.Module):
    """
    A concepts captioning model
    """

    def __init__(self, opt):
        super(CONTRAB, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size
        self.filter_encoder_layers = opt.filter_encoder_layers
        self.filter_encoder_size = opt.filter_encoder_size
        self.filter_decoder_layers = opt.filter_decoder_layers
        self.filter_decoder_size = opt.filter_decoder_size
        self.filter_encoder_heads = opt.filter_encoder_heads
        self.filter_decoder_heads = opt.filter_decoder_heads

        self.captioner_type = opt.captioner_type
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.captioner_heads = opt.captioner_heads

        # assert self.filter_decoder_size == self.captioner_size == self.textual_encoding_size, print(self.filter_decoder_size, self.captioner_size, self.textual_encoding_size)
        # assert self.filter_decoder_size == self.visual_encoding_size

        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.pass_all_svo = opt.pass_all_svo
        self.clamp_nearest = opt.clamp_concepts
        self.svo_length =  opt.svo_length
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.feat_pool = FeatPool(self.feat_dims, self.visual_encoding_size, self.drop_prob_lm)

        # encode the visual features (linear>relu>dropout)
        self.global_encoders = list()
        for feat_dim in self.feat_dims:
            self.global_encoders.append(nn.Sequential(nn.Linear(feat_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.global_encoders = nn.ModuleList(self.global_encoders)

        # encode the box features 1024 -> 512 and 4 -> 512 (linear>relu>dropout)
        self.box_encoder = FeatPool(self.bfeat_dims, int(self.visual_encoding_size/2), self.drop_prob_lm, SQUEEZE=False)  # TODO replace with new box encoder

        # encode word positions
        self.pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.seq_length)
        self.svo_pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.svo_length)

        # Concept Prediction module (visual feature encoder > k concepts decoder)
        concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size, nhead=self.filter_encoder_heads,
                                                           dim_feedforward=self.filter_encoder_size, dropout=self.drop_prob_lm)
        self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.filter_encoder_layers)

        concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_encoder_size, nhead=self.filter_decoder_heads,
                                                           dim_feedforward=self.filter_decoder_size, dropout=self.drop_prob_lm)
        self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.filter_decoder_layers)

        # Caption Prediction Module (a decoder on the global feats and k concept embeddings)
        caption_decoder_layer = nn.TransformerDecoderLayer(d_model=self.filter_decoder_size, nhead=self.captioner_heads,
                                                           dim_feedforward=self.captioner_size, dropout=self.drop_prob_lm)
        self.caption_decoder = nn.TransformerDecoder(caption_decoder_layer, num_layers=self.captioner_layers)

        self.feat_expander = FeatExpander(self.seq_per_img)

        # opt.video_encoding_size = self.visual_encoding_size * self.num_feats

    def set_ss_prob(self, p):
        self.ss_prob = p

    def set_mixer_from(self, t):
        """Set values of mixer_from
        if mixer_from > 0 then start MIXER training
        i.e:
        from t = 0 -> t = mixer_from -1: use XE training
        from t = mixer_from -> end: use RL training
        """
        self.mixer_from = t

    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def concept_generator(self, feats, bfeats, expand_feat=1):

        # encode all of the global features into a single space
        encoded_global_feats = list()
        for global_feat, global_encoder in zip(feats, self.global_encoders):
            encoded_global_feats.append(global_encoder(global_feat))

        # encode the boxes into a similar space [cat(enc(1024>512), enc(4>512))]
        encoded_boxes = self.box_encoder(bfeats)

        # concat the box features to the global features
        visual_features = torch.cat(encoded_global_feats + [encoded_boxes], dim=-2)

        #### ENCODER ####
        encoded_features = self.concept_encoder(visual_features.permute(1, 0, 2))  # change to (time, batch, channel)
        if expand_feat:
            encoded_features = self.feat_expander(encoded_features.permute(1, 0, 2))  # change to (batch, time, channel)
            encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        #### END ENCODER ####

        #### DECODER ####        # init = torch.ones(one_hot.shape).unsqueeze(0).cuda()
        init = torch.ones((1,encoded_features.shape[1],self.filter_decoder_size)).cuda()
        out = self.concept_decoder(init, encoded_features)
        conf = self.logit(out.squeeze())
        # conf = torch.sigmoid(self.logit(out.squeeze()))
        #### END DECODER ####

        return out, conf

    def forward(self, gfeats, bfeats, seq, pos):

        # get the svo features and vocab indexs
        conc_emb, conc_conf = self.concept_generator(gfeats, bfeats)
        one_hot = torch.clamp(torch.sum(torch.nn.functional.one_hot(pos, num_classes=self.vocab_size), axis=1), 0, 1)
        one_hot[:, 0] = 0  # make the padding index 0

        fc_feats = self.feat_pool(gfeats, stack=True)
        fc_feats = self.feat_expander(fc_feats)
        fc_feats = fc_feats.permute(1, 0, 2)  # change to (time, batch, channel)
        encoded_features = torch.cat([fc_feats, conc_emb], dim=0)  # cat the vis feats and the svo

        caption_embeddings = self.embed(seq)  # emb indexs -> embeddings
        caption_embeddings = caption_embeddings.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.pos_encoder(caption_embeddings)  # add positional encoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, seq.size(-1)).cuda()  # create sequence mask
        tgt_key_padding_mask = (seq == 0)  # create padding mask

        # Run the decoder
        out = self.caption_decoder(caption_embeddings,
                                   encoded_features,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        out = out[:-1].permute(1, 0, 2)  # remove the last token and change back to (batch, concepts, channels)
        caption_probs = F.log_softmax(self.logit(self.dropout(out)), dim=-1)  # calc word probs
        caption_idxs = F.softmax(self.logit(out), dim=-1).argmax(-1)  # get best word indexs (not used)

        caption_seq = seq[:, 1:]  # get gt caption (minus the BOS token)
        caption_logprobs = caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2)

        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return caption_probs, caption_seq, caption_logprobs, conc_conf, one_hot

    def sample(self, feats, bfeats, pos, opt={}):
        beam_size = opt.get('beam_size', 1)
        expand_feat = opt.get('expand_feat', 0)

        # svo_out, svo_it = self.concept_generator_cap(feats, bfeats, expand_feat=expand_feat)
        conc_emb, conc_conf = self.concept_generator(feats, bfeats, expand_feat=expand_feat)
        if beam_size > 1:
            return ((*self.sample_beam(feats, bfeats, pos, opt)), conc_emb, conc_conf)
        else:
            return NotImplementedError

    def sample_beam(self, feats, bfeats, pos, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats, stack=True)
        # svo_out, svo_it = self.concept_generator_cap(feats, bfeats, expand_feat=0)
        conc_emb, conc_conf = self.concept_generator(feats, bfeats, expand_feat=0)
        conc_emb = conc_emb.permute(1, 0, 2)
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            fc_feats_k = fc_feats[k].expand(beam_size, self.num_feats, self.visual_encoding_size)
            conc_emb_k = conc_emb[k].expand(beam_size, 1, self.textual_encoding_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1

            its = torch.LongTensor(self.seq_length, beam_size).zero_().cuda()
            for token_idx in range(start_i, end_i):
                if token_idx == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(self.bos_index)  # [1,1,1,1,1]
                    its[token_idx] = it
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # sorted array of logprobs along each previous beam (last
                    # true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in
                            # (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append(
                                {'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.item(), 'r': local_logprob.item()})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    if token_idx > 1:
                        # well need these as reference when we fork beams
                        # around
                        beam_seq_prev = beam_seq[:token_idx - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx - 1].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # append new end terminal at the end of this beam
                        # c'th word is the continuation
                        beam_seq[token_idx - 1, vix] = v['c']
                        beam_seq_logprobs[token_idx - 1, vix] = v['r']  # the raw logprob here
                        # the new (sum) logprob along this beam
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 0 or token_idx == self.seq_length - 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            if token_idx > 1:
                                ppl = np.exp(-beam_logprobs_sum[vix] / (token_idx - 1))
                            else:
                                ppl = 10000
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix],
                                                       'ppl': ppl
                                                       })

                    # encode as vectors
                    it = Variable(beam_seq[token_idx - 1].cuda())
                    its[token_idx] = it

                encoded_features = torch.cat([fc_feats_k, conc_emb_k], dim=1).permute(1, 0, 2)  # change to (time, batch, channel)  # cat the vis feats and the svo
                decoder_input = self.embed(its[:token_idx+1])
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, token_idx + 1).cuda()

                decoder_input = self.pos_encoder(decoder_input)  # add positional encoding
                decoder_output = self.caption_decoder(decoder_input, encoded_features, tgt_mask=tgt_mask)

                output = decoder_output[-1]

                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)