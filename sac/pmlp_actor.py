# The skeleton of the code here is adapted from https://github.com/sharma-arjun/phase-ddpg
# Actor and Critic networks are changed to run in SAC setting.

import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from sac.utils import kn, spline_w, compute_multipliers, init_fanin
from torch.distributions import Normal


class PMLPActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dtype=torch.float, n_layers=2, batch_size=1, scale=1.0,
                 args=None, action_space=None, action_dim=None):
        super(PMLPActor, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size[0]
        self.output_size = output_size
        self.hidden_size_2 = hidden_size[-1]
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dtype = dtype
        self.action_space = action_space
        self.args = args
        self.n_actions = action_dim

        self.control_hidden_list = []
        self.control_h2o_list = []

        self.l_00 = nn.Linear(self.input_size, self.hidden_size_1).type(dtype).to(args.device)
        self.h2o_0 = nn.Linear(self.hidden_size_2, self.output_size).type(dtype).to(args.device)
        self.l_10 = nn.Linear(self.input_size, self.hidden_size_1).type(dtype).to(args.device)
        self.h2o_1 = nn.Linear(self.hidden_size_2, self.output_size).type(dtype).to(args.device)
        self.l_20 = nn.Linear(self.input_size, self.hidden_size_1).type(dtype).to(args.device)
        self.h2o_2 = nn.Linear(self.hidden_size_2, self.output_size).type(dtype).to(args.device)
        self.l_30 = nn.Linear(self.input_size, self.hidden_size_1).type(dtype).to(args.device)
        self.h2o_3 = nn.Linear(self.hidden_size_2, self.output_size).type(dtype).to(args.device)

        if n_layers == 2:
            self.l_01 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype).to(args.device)
            self.l_11 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype).to(args.device)
            self.l_21 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype).to(args.device)
            self.l_31 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype).to(args.device)

        self.initialize()

        self.control_hidden_list.append([self.l_00, self.l_10, self.l_20, self.l_30])
        if n_layers == 2:
            self.control_hidden_list.append([self.l_01, self.l_11, self.l_21, self.l_31])

        self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

        self.hidden_list = []
        self.h2o_list = []
        self.phase_list = []

        self.loss = nn.MSELoss().to(args.device)

        if self.action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high[:self.n_actions] - action_space.low[:self.n_actions]) / 2.
            ).to(args.device)

            self.action_bias = torch.FloatTensor(
                (action_space.high[:self.n_actions] + action_space.low[:self.n_actions]) / 2.
            ).to(args.device)
        else:
            self.action_scale = torch.tensor(1.).to(args.device)
            self.action_bias = torch.tensor(0.).to(args.device)

        self.mu = nn.Linear(self.output_size, self.n_actions).to(args.device)
        self.log_sigma = nn.Linear(self.output_size, self.n_actions).to(args.device)

        # to initialize grad of control hidden and h2o
        dummy_x = Variable(torch.zeros(batch_size, input_size), requires_grad=False).type(dtype).to(args.device)
        dummy_y = Variable(torch.zeros(batch_size, output_size), requires_grad=False).type(dtype).to(args.device)
        dummy_criterion = nn.MSELoss()

        if n_layers == 1:
            for l, h2o in zip(self.control_hidden_list[0], self.control_h2o_list):
                dummy_h = F.relu(l(dummy_x))
                dummy_o = h2o(dummy_h)
                dummy_loss = dummy_criterion(dummy_o, dummy_y)
                dummy_loss.backward()

        if n_layers == 2:
            for l0, l1, h2o in zip(self.control_hidden_list[0], self.control_hidden_list[1], self.control_h2o_list):
                dummy_h0 = F.relu(l0(dummy_x))
                dummy_h1 = l1(dummy_h0)
                dummy_o = h2o(dummy_h1)
                dummy_loss = dummy_criterion(dummy_o, dummy_y)
                dummy_loss.backward()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_milestones, gamma=args.lr_factor
        )

    def initialize(self):
        init_fanin(self.l_00.weight)
        init_fanin(self.l_10.weight)
        init_fanin(self.l_20.weight)
        init_fanin(self.l_30.weight)

        init.uniform_(self.h2o_0.weight,-3e-3, 3e-3)
        init.uniform_(self.h2o_0.bias,-3e-3, 3e-3)
        init.uniform_(self.h2o_1.weight,-3e-3, 3e-3)
        init.uniform_(self.h2o_1.bias,-3e-3, 3e-3)
        init.uniform_(self.h2o_2.weight,-3e-3, 3e-3)
        init.uniform_(self.h2o_2.bias,-3e-3, 3e-3)
        init.uniform_(self.h2o_3.weight,-3e-3, 3e-3)
        init.uniform_(self.h2o_3.bias,-3e-3, 3e-3)

        if self.n_layers == 2:
            init_fanin(self.l_01.weight)
            init_fanin(self.l_11.weight)
            init_fanin(self.l_21.weight)
            init_fanin(self.l_31.weight)

    def forward(self,x,phase):

        #w = torch.from_numpy(spline_w(phase.data.cpu().numpy()))
        #w0,w1,w2,w3 = compute_multipliers(w,phase.data.cpu().numpy())
        w = spline_w(phase)
        w0,w1,w2,w3 = compute_multipliers(w,phase)
        w0_h1 = Variable(w0.repeat(1,self.hidden_size_1).type(self.dtype)).to(self.args.device)
        w1_h1 = Variable(w1.repeat(1,self.hidden_size_1).type(self.dtype)).to(self.args.device)
        w2_h1 = Variable(w2.repeat(1,self.hidden_size_1).type(self.dtype)).to(self.args.device)
        w3_h1 = Variable(w3.repeat(1,self.hidden_size_1).type(self.dtype)).to(self.args.device)

        if self.n_layers == 2:
            w0_h2 = Variable(w0.repeat(1,self.hidden_size_2).type(self.dtype)).to(self.args.device)
            w1_h2 = Variable(w1.repeat(1,self.hidden_size_2).type(self.dtype)).to(self.args.device)
            w2_h2 = Variable(w2.repeat(1,self.hidden_size_2).type(self.dtype)).to(self.args.device)
            w3_h2 = Variable(w3.repeat(1,self.hidden_size_2).type(self.dtype)).to(self.args.device)

        w0_o = Variable(w0.repeat(1,self.output_size).type(self.dtype)).to(self.args.device)
        w1_o = Variable(w1.repeat(1,self.output_size).type(self.dtype)).to(self.args.device)
        w2_o = Variable(w2.repeat(1,self.output_size).type(self.dtype)).to(self.args.device)
        w3_o = Variable(w3.repeat(1,self.output_size).type(self.dtype)).to(self.args.device)
                
        h_0 = F.relu(w0_h1*self.control_hidden_list[0][0](x) + w1_h1*self.control_hidden_list[0][1](x) + w2_h1*self.control_hidden_list[0][2](x) + w3_h1*self.control_hidden_list[0][3](x))
        if self.n_layers == 2:
            h_1 = F.relu(w0_h2*self.control_hidden_list[1][0](h_0) + w1_h2*self.control_hidden_list[1][1](h_0) + w2_h2*self.control_hidden_list[1][2](h_0) + w3_h2*self.control_hidden_list[1][3](h_0))
            o = w0_o*self.control_h2o_list[0](h_1) + w1_o*self.control_h2o_list[1](h_1) + w2_o*self.control_h2o_list[2](h_1) + w3_o*self.control_h2o_list[3](h_1)
        else:
            o = w0_o*self.control_h2o_list[0](h_0) + w1_o*self.control_h2o_list[1](h_0) + w2_o*self.control_h2o_list[2](h_0) + w3_o*self.control_h2o_list[3](h_0)
        
        o = F.relu(o)
        
        mu = self.mu(o)
        log_sigma = self.log_sigma(o)
        log_sigma = torch.clamp(log_sigma, min=-20, max=10)

        return mu, log_sigma

    def sample(self, state, phase):
        mu, log_sigma = self.forward(state, phase)
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)

        x = normal.rsample()
        y = torch.tanh(x)

        # Reparametrization
        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) +  1e-6)
        log_prob = log_prob.sum(axis=1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu, sigma

    def reset(self):
        self.hidden_list = []
        self.h2o_list = []
        self.phase_list = []


    def weight_from_phase(self, phase, control_hidden_list, control_h2o_list):
        weight = {}
        w = spline_w(phase)
        for n in range(len(control_hidden_list)):
            for key in control_hidden_list[0][0]._parameters.keys():
                weight[key + '_' + str(n)] = control_hidden_list[n][kn(phase, 1)]._parameters[key].data \
                    + w*0.5*(control_hidden_list[n][kn(phase, 2)]._parameters[key].data - control_hidden_list[n][kn(phase, 0)]._parameters[key].data) \
                    + w*w*(control_hidden_list[n][kn(phase, 0)]._parameters[key].data - 2.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data + 2*control_hidden_list[n][kn(phase, 2)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data) \
                    + w*w*w*(1.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data - 1.5*control_hidden_list[n][kn(phase, 2)]._parameters[key].data + 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 0)]._parameters[key].data)
        for key in control_h2o_list[0]._parameters.keys():
            weight[key] = control_h2o_list[kn(phase, 1)]._parameters[key].data + w*0.5*(control_h2o_list[kn(phase, 2)]._parameters[key].data - control_h2o_list[kn(phase, 0)]._parameters[key].data) \
                + w*w*(control_h2o_list[kn(phase, 0)]._parameters[key].data - 2.5*control_h2o_list[kn(phase, 1)]._parameters[key].data \
                + 2*control_h2o_list[kn(phase, 2)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data) \
                + w*w*w*(1.5*control_h2o_list[kn(phase, 1)]._parameters[key].data - 1.5*control_h2o_list[kn(phase, 2)]._parameters[key].data \
                + 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 0)]._parameters[key].data)
        return weight


    def set_weight(self, w, hiddens, h2o):
        count = 0
        for l in hiddens:
            l._parameters['weight'].data = w['weight_' + str(count)]
            l._parameters['bias'].data = w['bias_' + str(count)]
            count += 1
        h2o._parameters['weight'].data = w['weight']
        h2o._parameters['bias'].data = w['bias']

    def update_control_gradients(self):
        for hiddens, phase in zip(self.hidden_list, self.phase_list):
            w = spline_w(phase)
            count = 0
            for l in hiddens:
                for key in l._parameters.keys():
                    self.control_hidden_list[count][kn(phase,0)]._parameters[key].grad.data += l._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
                    self.control_hidden_list[count][kn(phase,1)]._parameters[key].grad.data += l._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
                    self.control_hidden_list[count][kn(phase,2)]._parameters[key].grad.data += l._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
                    self.control_hidden_list[count][kn(phase,3)]._parameters[key].grad.data += l._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)
                count += 1
        for h2o, phase in zip(self.h2o_list, self.phase_list):
            w = spline_w(phase)
            for key in h2o._parameters.keys():
                self.control_h2o_list[kn(phase,0)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
                self.control_h2o_list[kn(phase,1)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
                self.control_h2o_list[kn(phase,2)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
                self.control_h2o_list[kn(phase,3)]._parameters[key].grad.data += h2o._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)
