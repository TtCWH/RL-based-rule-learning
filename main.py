# coding=utf-8
import pdb
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch
from dataset import *
from models import *
from args import get_args
from loss import LossAndMetric
from utils import *

class Model(object):
    def __init__(self, args):
        self.saved_model_path = args.saved_model_path
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = LossAndMetric(args)
        for name, arg in vars(self.args).items():
            print('%s: %s' % (name, arg))

    def init_fasttext(self, dataset):
        print('init fastext')
        p_embedding_layer = nn.Embedding(
            len(dataset.property2id) + 1, args.p_embedding_size, padding_idx=0).to(self.device)
        v_embedding_layer = nn.Embedding(
            len(dataset.value2id) + 1, args.v_embedding_size, padding_idx=0).to(self.device)

        p_embedding_layer.weight.data.requires_grad = True
        v_embedding_layer.weight.data.requires_grad = True

        nn.init.xavier_normal_(p_embedding_layer.weight)
        nn.init.xavier_normal_(v_embedding_layer.weight)

        # pad的embedding全为0
        p_embedding_layer.weight.data[0] = 0
        v_embedding_layer.weight.data[0] = 0

        self.fasttext = FastText(p_embedding_layer,
                        v_embedding_layer,output_size=len(dataset.scene2id),dropout_rate=self.args.dropout_rate).to(self.device)

        print('done')

    def init_agent(self, dataset, p_embedding_layer, v_embedding_layer):
        scene_embedding_layer = nn.Embedding(
            len(dataset.scene2id), self.args.s_embedding_size).to(self.device)
        scene_embedding_layer.weight.data.requires_grad = True

        nn.init.xavier_normal_(scene_embedding_layer.weight)

        self.agent = PolicyNet(
            p_embedding_layer,
            v_embedding_layer,
            scene_embedding_layer,
            args).to(self.device)
        self.baseline = Baseline(args.hidden_size).to(self.device)
        print('done')


    def train_fasttext(self, dataset):
        print('fastext')
        self.init_fasttext(dataset)
        print('reading training data...')
        fasttext = self.fasttext
        args = self.args
        saved_model_path = self.args.saved_model_path
        optimizer_fasttext = optim.Adam(self.fasttext.parameters(), args.lr)
        train_data = TensorDataset(
            torch.from_numpy(dataset.ps).long(),  # [None, 40]
            torch.from_numpy(dataset.vs).long(),  # [None, 40]
            torch.from_numpy(dataset.scenes).long(),  # [None]
            torch.from_numpy(dataset.lens).long())  # [None]
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        criterion = nn.NLLLoss(reduction='mean')
        print('training fasttext...')
        for epoch in range(args.fasttext_epochs):
            print('epoch : {}'.format(epoch))
            fasttext.train()
            start_time = time.time()
            for i, data in enumerate(train_loader):
                x_p, x_v, y, x_len = data
                x_p, x_v, y, x_len = x_p.to(self.device), x_v.to(self.device), y.to(self.device), x_len.to(self.device)

                output = fasttext.forward(x_p, x_v, x_len.unsqueeze(-1))
                loss = criterion(output, y)
                print('epoch {} : loss == {}'.format(epoch, loss.item()))

                optimizer_fasttext.zero_grad()
                loss.backward()
                optimizer_fasttext.step()
            print('epoch {} : training time == {}'.format(epoch, time.time()-start_time))
            if (epoch+1)%10 == 0:
                snapshot(fasttext, epoch, saved_model_path)

    def train_agent(self, dataset):
        print('agent')
        self.init_fasttext(dataset)
        fasttext = self.fasttext
        args = self.args
        fasttext, params_fasttext = load(fasttext, self.args.saved_model_path)
        if params_fasttext['epoch']<0:
            self.train_fasttext(dataset)
        fasttext.eval()
        self.init_agent(dataset, fasttext.p_embedding_layer, fasttext.v_embedding_layer)
        agent = self.agent
        baseline = self.baseline
        agent, params_agent = load(agent, self.args.saved_model_path)

        optimizer_agent = optim.Adam(
            [p for p in self.agent.parameters()][2:], args.lr)
        # agent不更新p_embedding和v_embedding
        optimizer_baseline = optim.Adam(self.baseline.parameters(), args.lr)

        print('reading training data...')

        train_data = TensorDataset(
            torch.from_numpy(dataset.ps).long(),  # [None, 40]
            torch.from_numpy(dataset.vs).long(),  # [None, 40]
            torch.from_numpy(dataset.scenes).long(),  # [None]
            torch.from_numpy(dataset.lens).long())  # [None]
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        print('training agent...')
        args = self.args
        # print('model.__name__: {}'.format(model.__class__.__name__))
        for name, arg in vars(self.args).items():
            print('%s: %s' % (name, arg))

        for epoch in range(params_agent['epoch']+1, args.agent_epochs):
            agent.train()
            baseline.train()
            acc_num = 0
            start_time = time.time()
            for i, data in enumerate(train_loader):
                x_p, x_v, y, x_len = data
                x_p, x_v, y, x_len = x_p.to(self.device), x_v.to(self.device), y.to(self.device), x_len.to(self.device)
                # pdb.set_trace()
                rollouts = agent(x_p, x_v, y, x_len)
                rollouts = baseline(rollouts)

                agent_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
                baseline_loss = torch.tensor(0, dtype=torch.float32).to(self.device)

                for rollout in rollouts:
                    # 多次采样稳定训练
                    selected_seq_p = np.apply_along_axis(
                        re_arange, 1, (rollout['agent_actions'] * x_p).cpu().numpy())
                    selected_seq_v = np.apply_along_axis(
                        re_arange, 1, (rollout['agent_actions'] * x_v).cpu().numpy())
                    selected_seq_p = torch.from_numpy(selected_seq_p).to(self.device)
                    selected_seq_v = torch.from_numpy(selected_seq_v).to(self.device)

                    lens = (selected_seq_p != 0).sum(-1).float() + 1
                    # 避免出现长度为0的pv sequence
                    # 把被选中的pv聚合到一起

                    valid_pos = (x_p != 0).float()
                    # 只计算pvs实际长度的部分

                    cal_lens = (selected_seq_p != 0).sum(-1)
                    cal_lens += (cal_lens == 0).long()

                    # with torch.no_grad():
                    #     raw_critic_out = critic(x_p, x_v, x_len)

                    fasttext_output = fasttext(
                        selected_seq_p, selected_seq_v, cal_lens.int().unsqueeze(-1))

                    pred = torch.argmax(fasttext_output, -1)

                    posi_reward = (torch.eq(pred, y.squeeze(-1)).float() * (1 + 1 / lens)
                                   ).unsqueeze(-1).repeat(1, args.max_len)
                    posi_reward *= (lens.unsqueeze(-1) != 1).float()
                    posi_reward *= (lens.unsqueeze(-1) <
                                    args.lhs_len + 2).float()
                    nega_reward = -(posi_reward == 0).float()
                    episodic_reward = posi_reward + nega_reward

                    # print(rollout['B'])

                    reward = (episodic_reward - rollout['B']) * valid_pos
                    # 用baseline稳定训练
                    entropy_tmp = rollout['probs'] * \
                                  torch.log(rollout['probs'] + args.entropy_min)
                    entropy_term = (entropy_tmp.sum(-1) *
                                    valid_pos).sum(-1).mean()

                    baseline_loss = baseline_loss + torch.norm(reward, 2)
                    agent_loss = agent_loss + (-rollout['logps'] * reward).sum(-1).mean() + \
                                 args.entropy_prop * entropy_term

                agent_loss = agent_loss / args.num_rollouts
                if torch.isnan(agent_loss):
                    pdb.set_trace()
                baseline_loss = baseline_loss / args.num_rollouts
                # update baseline
                optimizer_baseline.zero_grad()
                baseline_loss.backward(retain_graph=True)
                optimizer_baseline.step()

                # update agent
                optimizer_agent.zero_grad()
                agent_loss.backward()
                optimizer_agent.step()

                acc, tmp_acc_num = accuracy(pred, y.squeeze(-1))
                acc_num += tmp_acc_num

                print(
                    'entropy == {:.3f}'.format(
                        (args.entropy_prop * entropy_term).item()))
                print('actions_lens == {:.3f}'.format((lens - 1).float().mean()))
                print('epoch == {}'.format(epoch))
                print('agent_loss == {:.3f}'.format(agent_loss.item()))
                print('baseline_loss == {:.3f}'.format(baseline_loss.item()))
                print('acc == {:.3f}'.format(acc))
                print(rollout['agent_actions'][0][:10].tolist())
                print(torch.exp(rollout['logps'][0][:10]).tolist())
            print('train time == {:.3f}'.format(time.time() - start_time))
            total_acc = acc_num / dataset.ps.shape[0]
            print('epoch {} : acc == {:.3f}'.format(epoch, total_acc))
            if epoch%2 == 0:
                snapshot(agent, epoch, args.saved_model_path)
        self.test(dataset)

    def test(self, dataset):
        print('generate rules')
        self.init_fasttext(dataset)
        fasttext = self.fasttext
        fasttext = load(fasttext, self.args.saved_model_path)[0]
        fasttext.eval()
        self.init_agent(dataset, fasttext.p_embedding_layer, fasttext.v_embedding_layer)
        agent = self.agent
        agent = load(agent, self.args.saved_model_path)[0]
        agent.eval()

        print('reading data...')

        train_data = TensorDataset(
            torch.from_numpy(dataset.ps).long(),  # [None, 40]
            torch.from_numpy(dataset.vs).long(),  # [None, 40]
            torch.from_numpy(dataset.scenes).long(),  # [None]
            torch.from_numpy(dataset.lens).long())  # [None]
        train_loader = DataLoader(train_data, 8, shuffle=False)
        ps, vs, labels, preds, probs = [], [], [], [], []
        for i, data in enumerate(train_loader):
            x_p, x_v, y, x_len = data
            x_p, x_v, y, x_len = x_p.to(self.device), x_v.to(self.device), y.to(self.device), x_len.to(self.device)
            with torch.no_grad():
                actions = agent.inference(x_p, x_v, y, x_len)
                selected_seq_p = np.apply_along_axis(
                    re_arange, 1, (actions * x_p).cpu().numpy())
                selected_seq_v = np.apply_along_axis(
                    re_arange, 1, (actions * x_v).cpu().numpy())
                selected_seq_p = torch.from_numpy(selected_seq_p).to(self.device)
                selected_seq_v = torch.from_numpy(selected_seq_v).to(self.device)
                tmp_lens = (selected_seq_p != 0).sum(-1).float()
                one_lens = torch.ones_like(tmp_lens).float()
                lens = torch.where(tmp_lens>0, tmp_lens, one_lens).to(self.device)
                fasttext_output = fasttext(selected_seq_p, selected_seq_v, lens.int().unsqueeze(-1))
            pred = torch.argmax(fasttext_output, -1)
            ps.extend(selected_seq_p.tolist())
            vs.extend(selected_seq_v.tolist())
            labels.extend(y.tolist())
            preds.extend(pred.tolist())
            probs.extend(fasttext_output.tolist())
        print('generating rules...')
        res = show_rule(
                dataset,
                ps,
                vs,
                labels,
                preds,
                probs)
        high_quality_res = res[(res.hc>=0.3) & (res.conf>=0.8)]
        data_path = os.path.join(self.args.result_path, 'res1.csv')
        high_quality_data_path = os.path.join(self.args.result_path, 'high_quality_res1.csv')
        res.to_csv(data_path, sep='\t', index=False, header=False)
        high_quality_res.to_csv(high_quality_data_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    dataset = NLPDataset(args)
    # pdb.set_trace()
    model = Model(args)
    if args.generate_rules == 1:
        model.test(dataset)
    else:
        model.train_agent(dataset)
