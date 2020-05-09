# coding=utf-8

import pdb
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class BiLSTM(nn.Module):
    '''
    input a sequence of embedding [batchsize,seq_max_len,embed_dim]
    out put a sequence of actions [batchsize,seq_max_len,2]
    '''

    def __init__(self, input_size, hidden_size, scene_emb_size, action_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True)
        self.fcn1 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)
        self.fcn2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)
        self.fcn3 = nn.Linear(
            in_features=1,
            out_features=action_size)
        self.fcn4 = nn.Linear(
            in_features=scene_emb_size,
            out_features=hidden_size)
        # self.cnn = nn.Conv1d(hidden_size // 2, hidden_size // 2, 2)
        # self.fcn2 = nn.Linear(
        #     in_features=hidden_size // 2,
        #     out_features=action_size)
        self.take_last = False
        self.batch_first = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward_lstm_with_var_length(self, x, x_len):
        # 根据batch样本长度pad样本
        # 1. sort
        # pdb.set_trace()
        full_len = x.shape[1]
        x_sort_idx = torch.sort(-x_len)[-1].to(self.device)
        x_unsort_idx = torch.sort(x_sort_idx)[-1].to(self.device)
        x_len = list(x_len[x_sort_idx])
        x = x[x_sort_idx]
        # pdb.set_trace()
        # 2. pack
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # 3. forward lstm
        out_pack, (hn, cn) = self.rnn(x_emb_p)
        # 4. unsort h
        # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        hn = hn.permute(1, 0, 2)[x_unsort_idx]  # swap the first two dim
        hn = hn.permute(1, 0, 2)  # swap the first two again to recover
        if self.take_last:
            return hn.squeeze(0)
        else:
            # TODO test if ok
            # unpack: out
            out, _ = pad_packed_sequence(
                out_pack, batch_first=self.batch_first, total_length=full_len)  # (sequence, lengths)
            out = out[x_unsort_idx]
            # unpack: c
            # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            cn = cn.permute(1, 0, 2)[x_unsort_idx]  # swap the first two dim
            cn = cn.permute(1, 0, 2)  # swap the first two again to recover
            return out, (hn, cn)

    def forward(self, x, y, x_len):
        rnn_output = self.forward_lstm_with_var_length(x, x_len)[0]
        # pdb.set_trace()
        res = self.fcn3(
            torch.bmm(
                self.fcn2(
                    F.relu(self.fcn1(rnn_output))), self.fcn4(y.unsqueeze(1)).permute(
                    0, 2, 1)))
        return F.softmax(res, dim=-1), res, rnn_output

    def act(self, embed_output, y, x_len):
        probs, fcn_output, rnn_output = self.forward(embed_output, y, x_len)
        # pdb.set_trace()
        m = Categorical(probs)
        actions = m.sample()
        return actions, m.log_prob(actions), rnn_output, probs

    def infer_act(self, embed_output, y, x_len):
        probs, fcn_output, rnn_output = self.forward(embed_output, y, x_len)
        return torch.argmax(probs, -1)


class PolicyNet(nn.Module):
    '''
    input a sequence of pvs [batchsize,seq_max_len]
    output a sequence of necessary pvs for critic
    '''

    def __init__(
            self,
            p_embedding_layer,
            v_embedding_layer,
            scene_embedding_layer,
            args,
            output_size=2):
        super(PolicyNet, self).__init__()
        self.hidden_size = args.hidden_size
        self.p_embedding_layer = p_embedding_layer
        self.v_embedding_layer = v_embedding_layer
        self.scene_embedding_layer = scene_embedding_layer
        self.policy = BiLSTM(
            p_embedding_layer.weight.data.shape[-1] + v_embedding_layer.weight.data.shape[-1], args.hidden_size, scene_embedding_layer.weight.shape[-1], output_size)
        self.num_rollout = args.num_rollouts

    def forward(self, x_p, x_v, y, x_len):
        embed_y = self.scene_embedding_layer(y)
        with torch.no_grad():
            embed_p = self.p_embedding_layer(x_p)
            embed_v = self.v_embedding_layer(x_v)
        embed = torch.cat([embed_p, embed_v], -1)
        # embed = torch.cat(
        #     [embed_y.repeat(1, embed.shape[1], 1), embed], -1)
        # [batchsize, 40, p_emb_dim + v_emb_dim + scene_emb_dim]

        rollouts = []

        for _ in range(self.num_rollout):  # 取样多次让训练稳定
            rollout = {
                'logps': [],
                'agent_actions': [],
                'agent_states': [],
                'probs': [],
                'B': 0  # baseline rewards
            }
            actions, logps, agent_states, probs = self.policy.act(
                embed, embed_y, x_len)
            rollout['agent_actions'] = actions
            rollout['logps'] = logps  # 当前action的概率
            rollout['agent_states'] = agent_states  # 用于baseline计算reward
            rollout['probs'] = probs
            rollouts.append(rollout)
        return rollouts

    def inference(self, x_p, x_v, y, x_len):
        embed_y = self.scene_embedding_layer(y)
        embed_p = self.p_embedding_layer(x_p)
        embed_v = self.v_embedding_layer(x_v)
        embed = torch.cat([embed_p, embed_v], -1)

        actions = self.policy.infer_act(embed, embed_y, x_len)
        return actions

class FastText(nn.Module):
    '''
    input a sequence of pvs [batchsize,seq_max_len]
    output the log_probs of scenes belong to these pvs [batchsize,scene_size]
    '''

    def __init__(
            self,
            p_embedding_layer,
            v_embedding_layer,
            output_size,
            dropout_rate):
        '''
        :param p_embedding_layer: pre-defined property embedding layer
        :param v_embedding_layer: pre-defined value embedding layer
        :param output_size: the size of Lifestyles
        :param dropout_rate: the prob of net drop
        '''
        super(FastText, self).__init__()
        self.p_embedding_layer = p_embedding_layer
        self.v_embedding_layer = v_embedding_layer
        hidden_size = p_embedding_layer.weight.shape[-1] + v_embedding_layer.weight.shape[-1]
        self.fcn = nn.Linear(hidden_size, output_size)
        self.dropout_rate = dropout_rate

    def forward(self, x_p, x_v, x_len):
        '''
        :param x_p: [batch_size, seq_max_len] / int32 / a sequence of property
        :param x_v: [batch_size, seq_max_len] / int32 / a sequence of value
        :param x_len: [batch_size, 1]  / int32 /  a sequence of property
        :return: [batch_size, output_size]  / float32 /  the prob of each Lifestyle
        '''
        embed_p = self.p_embedding_layer(x_p)
        embed_v = self.v_embedding_layer(x_v)
        embed = torch.cat([embed_p, embed_v], -1)
        hidden = embed.sum(dim=1)
        # pdb.set_trace()
        hidden = hidden/(x_len.float())
        hidden = F.dropout(hidden, p=self.dropout_rate)
        res = self.fcn(hidden)
        return F.log_softmax(res, dim=-1)


class Critic(nn.Module):
    '''
    input a sequence of pvs [batchsize,seq_max_len]
    output the log_probs of scenes belong to these pvs [batchsize,scene_size]
    '''

    def __init__(
            self,
            p_embedding_layer,
            v_embedding_layer,
            hidden_size,
            k_size,
            output_size):
            # seq_len=5):
        super(Critic, self).__init__()
        self.p_embedding_layer = p_embedding_layer
        self.v_embedding_layer = v_embedding_layer
        self.cnn = nn.Conv1d(
            p_embedding_layer.weight.data.shape[-1] + v_embedding_layer.weight.data.shape[-1], hidden_size, k_size, padding=1)
        self.fcn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fcn2 = nn.Linear(hidden_size // 2, output_size)
        self.take_last = True
        self.batch_first = True
        # self.seq_len = seq_len

    def forward(self, x_p, x_v, x_len):
        # x_p = x_p[:, :self.seq_len]
        # x_v = x_v[:, :self.seq_len]
        embed_p = self.p_embedding_layer(x_p)
        embed_v = self.v_embedding_layer(x_v)
        embed = torch.cat([embed_p, embed_v], -1)
        # pdb.set_trace()
        cnn_output = self.cnn(embed.permute(0, 2, 1))
        # rnn_output = self.forward_lstm_with_var_length(embed, x_len)
        # rnn_output = torch.cat()
        # cnn_output = F.leaky_relu(self.cnn(rnn_output.permute(0,2,1)))
        cnn_output = F.max_pool1d(cnn_output, cnn_output.shape[-1]).squeeze(-1)
        res = self.fcn2(F.leaky_relu(self.fcn1(cnn_output)))
        return F.log_softmax(res, dim=-1)

class Baseline(nn.Module):

    def __init__(self, hidden_size):
        super(Baseline, self).__init__()
        self.fcn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fcn2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, rollouts):
        for roll in rollouts:
            roll['B'] = self.fcn2(
                F.relu(self.fcn1(roll['agent_states']))).squeeze(-1)
        return rollouts

