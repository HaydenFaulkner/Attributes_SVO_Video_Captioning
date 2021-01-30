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


class GeneralModel(nn.Module):
    """
    A general model which does it all
    """

    def __init__(self, opt):
        super(GeneralModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.bfeat_dims = opt.bfeat_dims
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.input_features = opt.input_features

        self.grounder_type = opt.grounder_type
        self.captioner_type = opt.captioner_type

        self.textual_encoding_size = opt.input_encoding_size
        self.visual_encoding_size = opt.input_encoding_size

        self.drop_prob_lm = opt.drop_prob_lm
        self.caption_length = opt.seq_length
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1  # index of the <bos> token
        self.ss_prob = 0
        self.mixer_from = 0
        self.attention_record = list()

        self.feat_expander = FeatExpander(self.seq_per_img)
        opt.video_encoding_size = self.visual_encoding_size

        self.embed = nn.Embedding(self.vocab_size, self.textual_encoding_size)  # word embedding layer (1-hot -> enc)
        self.logit = nn.Linear(self.textual_encoding_size, self.vocab_size)  # logit embedding layer (enc -> vocab enc)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        ## INITIALISE INPUT FEATURE ENCODERS
        self.feat_enc = list()
        for in_dim in self.feat_dims:
            self.feat_enc.append(nn.Sequential(nn.Linear(in_dim, self.visual_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm)))
        self.feat_enc = nn.ModuleList(self.feat_enc)
        self.rf_encoder = nn.Sequential(nn.Linear(1024, self.visual_encoding_size-4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.rb_encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(self.drop_prob_lm))


        ## INITIALISE ATTENTION BASED SECONDARY INPUT FEATURES ENCODER MODULE
        self.input_encoder_layers = opt.input_encoder_layers
        self.input_encoder_heads = opt.input_encoder_heads
        self.input_encoder_size = opt.input_encoder_size
        self.concept_encoder = None
        if self.input_encoder_layers > 0:
            concept_encoder_layer = nn.TransformerEncoderLayer(d_model=self.visual_encoding_size,
                                                               nhead=self.input_encoder_heads,
                                                               dim_feedforward=self.input_encoder_size,
                                                               dropout=self.drop_prob_lm)
            self.concept_encoder = nn.TransformerEncoder(concept_encoder_layer, num_layers=self.input_encoder_layers)

        ## INITIALISE GROUNDING MODULE
        self.gt_concepts_while_training = opt.gt_concepts_while_training
        self.num_concepts = opt.num_concepts
        self.grounder_layers = opt.grounder_layers
        self.grounder_heads = opt.grounder_heads
        self.grounder_size = opt.grounder_size
        if self.grounder_type in ['niuc']:
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
        elif self.grounder_type in ['iuc', 'ioc']:
            self.svo_pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.num_concepts+1)
            # iterative
            concept_decoder_layer = nn.TransformerDecoderLayer(d_model=self.visual_encoding_size, nhead=self.grounder_heads,
                                                               dim_feedforward=self.grounder_size, dropout=self.drop_prob_lm)
            self.concept_decoder = nn.TransformerDecoder(concept_decoder_layer, num_layers=self.grounder_layers)

        ## INITIALISE CAPTIONER
        self.captioner_size = opt.captioner_size
        self.captioner_layers = opt.captioner_layers
        self.captioner_heads = opt.captioner_heads
        if self.captioner_type in ['transformer']:  # Transformer Caption Decoder
            # encode word positions
            self.pos_encoder = PositionalEncoding(self.textual_encoding_size, dropout=self.drop_prob_lm, max_len=self.caption_length)

            # The transformer
            caption_decoder_layer = nn.TransformerDecoderLayer(d_model=self.visual_encoding_size, nhead=self.captioner_heads,
                                                               dim_feedforward=self.captioner_size, dropout=self.drop_prob_lm)
            self.caption_decoder = nn.TransformerDecoder(caption_decoder_layer, num_layers=self.captioner_layers)

        elif self.captioner_type in ['rnn', 'lstm', 'gru']:  # RNN Caption Decoder
            # feature attention layers
            self.v2a_layer = nn.Linear(self.visual_encoding_size, opt.att_size)
            self.h2a_layer = nn.Linear(self.captioner_size, opt.att_size)
            self.att_layer = nn.Linear(opt.att_size, 1)

            # The RNN
            self.core = RNNUnit(opt)  # the caption generation rnn LSTM(512) with input size 2048

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

    def non_iterative_grounder(self, feats):
        # embed the features for grounding attention
        q_feats = self.feat_pool_q(feats[:, 0:1])  # use img feat as query
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
        top_v, top_i = torch.topk(concept_probs, k=self.num_concepts)  # get the top preds
        mask = top_v > .5  # mask threshold
        top_emb = self.embed(top_i)

        concept_idxs = top_i
        concept_probs = concept_probs.unsqueeze(1).repeat(1, self.num_concepts, 1)  # repeat so it is the right shape
        return concept_probs, concept_idxs

    def iterative_grounder(self, feats, gt_concepts):

        if self.training and gt_concepts is not None:
            gt_concepts = torch.reshape(gt_concepts, (feats.shape[0], self.seq_per_img, -1))[:, 0]
            gt_concepts = F.pad(gt_concepts, (1, 0, 0, 0), "constant", self.bos_index)
            # gt_concepts = gt_concepts[:, :-1]
            concept_embeddings = self.embed(gt_concepts)
            concept_embeddings = concept_embeddings.permute(1, 0, 2) # change to (time, batch, channel)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, self.num_concepts+1).cuda()  # +1 for <bos> token
            tgt_key_padding_mask = (gt_concepts == 0)  # create padding mask
            if self.svo_pos_encoder is not None:
                concept_embeddings = self.svo_pos_encoder(concept_embeddings)
            out = self.concept_decoder(concept_embeddings, feats, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)   # out is target shp
            out = out.permute(1, 0, 2)  # change back to (batch, concepts, channels)

            concept_probs = F.log_softmax(self.logit(out), dim=-1)[:, :-1]
            concept_probs_b = F.softmax(self.logit(out), dim=-1)[:, :-1]
            concept_probs_c = torch.max(concept_probs_b, dim=1)[0]
            concept_idxs = concept_probs_b.argmax(-1)

        else:  # auto-regressive prediction at inference

            concept_probs = torch.zeros((feats.size(0), self.num_concepts, self.vocab_size)).cuda()
            concept_probs_b = torch.zeros((feats.size(0), self.num_concepts, self.vocab_size)).cuda()
            concept_idxs = torch.zeros((feats.size(0), self.num_concepts), dtype=torch.long).cuda()
            concept_idxs = F.pad(concept_idxs, (1, 0, 0, 0), "constant", self.bos_index)

            for i in range(1, self.num_concepts+1):
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


        if self.grounder_type in ['iuc']:
            raise NotImplementedError  # concept_probs_c dont work
            # concept_probs = torch.max(concept_probs_b, dim=1)[0]  # use max pred confs
            # concept_probs = torch.clamp(
            #     torch.sum(torch.nn.functional.one_hot(concept_idxs, num_classes=self.vocab_size), axis=1), 0, 1).type(
            #     torch.FloatTensor).cuda()  # use hard pred confs

        return concept_probs, concept_idxs

    def captioner_transformer(self, encoded_features, gt_caption):
        encoded_features = encoded_features.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.embed(gt_caption)  # emb indexs -> embeddings
        caption_embeddings = caption_embeddings.permute(1, 0, 2)  # change to (time, batch, channel)
        caption_embeddings = self.pos_encoder(caption_embeddings)  # add positional encoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, gt_caption.size(-1)).cuda()  # create sequence mask
        tgt_key_padding_mask = (gt_caption == 0)  # create padding mask

        # Run the decoder
        out = self.caption_decoder(caption_embeddings,
                                   encoded_features,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        out = out[:-1].permute(1, 0, 2)  # remove the last token and change back to (batch, concepts, channels)
        caption_probs = F.log_softmax(self.logit(self.dropout(out)), dim=-1)  # calc word probs
        caption_seq = F.softmax(self.logit(out), dim=-1).argmax(-1)  # get best word indexs

        caption_seq = gt_caption[:, 1:]  # get gt caption (minus the BOS token)

        return caption_probs, caption_seq

    def captioner_rnn(self, encoded_features, gt_caption):

        batch_size = encoded_features.size(0)
        self.captioner_type = 'lstm'
        state = self.init_hidden(batch_size)
        outputs = []
        sample_seq = []
        sample_logprobs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = gt_caption.size(1) - 1

        for token_idx in range(start_i, end_i):
            if token_idx == -1:  # initially set xt as global feats
                xt = encoded_features
            else:
                # token_idx = 0 corresponding to the <BOS> token
                # (already encoded in seq)
                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = encoded_features.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = gt_caption[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = gt_caption[:, token_idx].data.clone()
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
                    it = gt_caption[:, token_idx].clone()

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
                    hid_cont = state[0][-1].unsqueeze(0).transpose(0, 1).expand(batch_size, encoded_features.shape[1], state[0].shape[2])  # (bs * seq_per_img, 1, 512)
                elif self.captioner_type in ['gru']:
                    hid_cont = state.transpose(0, 1).expand(batch_size, encoded_features.shape[1], state.shape[2])  # (bs * seq_per_img, 1, 512)
                else:
                    hid_cont = state[0].transpose(0, 1).expand(batch_size, encoded_features.shape[1], state[0].shape[2])  # (bs * seq_per_img, 1, 512)

                alpha = self.att_layer(torch.tanh(self.v2a_layer(encoded_features) + self.h2a_layer(hid_cont)))  # (bs * seq_per_img, 2, 1)
                alpha = F.softmax(alpha, dim=1).transpose(1, 2)  # (bs * seq_per_img, 1, 2)
                att_encoded_features = torch.matmul(alpha, encoded_features).squeeze(1)

            # generate next word
            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:
                if self.model_type == 'manet':
                    encoded_features = self.manet(att_encoded_features, state[0])
                output, state = self.core(torch.cat([xt, att_encoded_features], 1), state)

            # generate the word softmax
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)), dim=1)
                outputs.append(output)

        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.cat([_.unsqueeze(1) for _ in sample_seq], 1)

    def feature_filtering(self, feats, bfeats, gt_concepts=None):
        feats_enc = list()
        for i, feat in enumerate(feats):
            feats_enc.append(self.feat_enc[i](feat))
        rf_enc = self.rf_encoder(bfeats[0])
        rb_enc = self.rb_encoder(bfeats[1])
        r_enc = torch.cat((rf_enc, rb_enc), dim=-1)
        encoded_features = torch.cat(feats_enc + [r_enc], dim=1)

        # only use a certain group of features
        if not self.input_features == 'imrc':
            encoded_features_pass = list()
            if 'i' in self.input_features:
                encoded_features_pass.append(encoded_features[:, 0:1, :])
            if 'm' in self.input_features:
                encoded_features_pass.append(encoded_features[:, 1:2, :])
            if 'r' in self.input_features:
                encoded_features_pass.append(encoded_features[:, -10:, :])
            if 'c' in self.input_features and encoded_features.shape[1] == 13:
                encoded_features_pass.append(encoded_features[:, 2:3, :])
            encoded_features = torch.cat(encoded_features_pass, dim=1)

        #### ENCODER ####
        if self.concept_encoder is not None:
            encoded_features = self.concept_encoder(encoded_features.permute(1, 0, 2)).permute(1, 0, 2)
        #### END ENCODER ####

        #### GROUNDER ####
        concept_probs = None
        concept_seq = None
        if self.grounder_type in ['niuc', 'iuc', 'ioc']:
            if self.grounder_type in ['niuc']:
                concept_probs, concept_seq = self.non_iterative_grounder(encoded_features)
            elif self.grounder_type in ['iuc', 'ioc']:
                concept_probs, concept_seq = self.iterative_grounder(encoded_features, gt_concepts=gt_concepts)
            else:
                raise NotImplementedError

            if gt_concepts is not None and self.gt_concepts_while_training and self.training:  # use gt concepts for cap gen
                encoded_features = torch.cat((encoded_features, self.embed(torch.reshape(gt_concepts, (concept_seq.shape[0], self.seq_per_img, -1))[:, 0])), dim=1)
            else:  # dont use gt for cap gen
                encoded_features = torch.cat((encoded_features, self.embed(concept_seq)), dim=1)
        #### END GROUNDER ####

        return encoded_features, concept_probs, concept_seq

    def forward(self, feats, bfeats, gt_caption, gt_concepts):

        combined_enc, concept_probs, concept_seq = self.feature_filtering(feats, bfeats, gt_concepts)

        encoded_features = self.feat_expander(combined_enc)

        if self.captioner_type in ['transformer']:  # Captioner - Transformer
            caption_probs, caption_seq = self.captioner_transformer(encoded_features, gt_caption)
        else:  # Captioner - RNN
            caption_probs, caption_seq = self.captioner_rnn(encoded_features, gt_caption)

        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        if concept_probs is not None:
            concept_probs = self.feat_expander(concept_probs)
            concept_seq = self.feat_expander(concept_seq)

            return caption_probs, caption_seq, caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2), \
                   concept_probs, concept_seq, concept_probs.gather(2, concept_seq.unsqueeze(2)).squeeze(2)
        else:
            return caption_probs, caption_seq, caption_probs.gather(2, caption_seq.unsqueeze(2)).squeeze(2), \
                   None, None, None

    def sample(self, feats, bfeats, gt_concepts, opt={}):
        beam_size = opt.get('beam_size', 1)

        encoded_features, concept_probs, concept_seq = self.feature_filtering(feats, bfeats)

        if beam_size > 1:
            return ((*self.sample_beam(encoded_features, opt)), concept_probs, concept_seq)
        else:
            return NotImplementedError

    def sample_beam(self, encoded_features, opt={}):
        """
        modified from https://github.com/ruotianluo/self-critical.pytorch
        """
        beam_size = opt.get('beam_size', 5)

        batch_size = encoded_features.size(0)

        seq = torch.LongTensor(self.caption_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.caption_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        state = []  # will be empty list for transformer decoder
        for k in range(batch_size):
            if self.captioner_type in ['rnn', 'lstm', 'gru']:
                state = self.init_hidden(beam_size)
            encoded_features_k = encoded_features[k].expand(beam_size, encoded_features.size(1), self.visual_encoding_size)

            beam_seq = torch.LongTensor(self.caption_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.caption_length, beam_size).zero_()
            # running sum of logprobs for each beam
            beam_logprobs_sum = torch.zeros(beam_size)

            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.caption_length - 1

            its = torch.LongTensor(self.caption_length, beam_size).zero_().cuda()  #TODO TRAN ONLY
            for token_idx in range(start_i, end_i):
                if token_idx == 0:  # input <bos>
                    it = encoded_features.data.new(beam_size).long().fill_(self.bos_index)  # [1,1,1,1,1]
                    its[token_idx] = it  #TODO TRAN ONLY

                    # xt = self.embed(Variable(it, requires_grad=False))  # TODO RNN ONLY
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

                    # construct new
                    new_state = [_.clone() for _ in state] ###########TODO RNN ONLY
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

                        # rearrange recurrent states	# TODO BLOCK RNN ONLY
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

                        if v['c'] == 0 or token_idx == self.caption_length - 2:
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
                    its[token_idx] = it  # TODO TRAN ONLY

                if self.captioner_type in ['transformer']:
                    encoded_features_k = encoded_features_k.permute(1, 0, 2)  # change to (time, batch, channel)
                    decoder_input = self.embed(its[:token_idx+1])
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(None, token_idx + 1).cuda()

                    decoder_input = self.pos_encoder(decoder_input)  # add positional encoding
                    decoder_output = self.caption_decoder(decoder_input, encoded_features_k, tgt_mask=tgt_mask)

                    output = decoder_output[-1]
                else:
                    # set the new input word
                    xt = self.embed(it)

                    # calculate attention over visual features based on visual feats
                    if self.captioner_layers > 1:
                        hid_cont = state[0][-1].unsqueeze(0).transpose(0, 1).expand(beam_size, encoded_features_k.shape[1], state[0].shape[2])
                    elif self.captioner_type in ['gru']:
                        hid_cont = state.transpose(0, 1).expand(beam_size, encoded_features_k.shape[1], state.shape[2])
                    else:
                        hid_cont = state[0].transpose(0, 1).expand(beam_size, encoded_features_k.shape[1], state[0].shape[2])

                    alpha = self.att_layer(torch.tanh(self.v2a_layer(encoded_features_k) + self.h2a_layer(hid_cont)))
                    alpha = F.softmax(alpha, dim=1).transpose(1, 2)
                    att_encoded_features = torch.matmul(alpha, encoded_features_k).squeeze(1)

                    if token_idx >= 1:
                        state = new_state
                        if self.captioner_type in ['gru']:
                            state = new_state[0].unsqueeze(0)
                    if self.model_type == 'standard':
                        output, state = self.core(xt, state)
                    else:
                        if self.model_type == 'manet':
                            encoded_features_k = self.manet(att_encoded_features, state[0])
                        output, state = self.core(torch.cat([xt, att_encoded_features], 1), state)

                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])

            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
