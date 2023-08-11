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

class PMLPCritic(nn.Module):
    def __init__(self, input_size_state, input_size_action, output_size, learning_rate, device, lr_milestones,
                  hidden_sizes=[256, 256], dtype=torch.float, n_layers=1, batch_size=1, scale=1.0,
                  lr_factor=0.5, loss='l2'):
        super(PMLPCritic, self).__init__()

        self.input_size_state = input_size_state
        self.input_size_action = input_size_action
        self.input_size = input_size_state + input_size_action
        self.output_size = output_size
        self.hidden_size_1 = hidden_sizes[0]
        self.hidden_size_2 = hidden_sizes[-1]
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.scale = scale 
        self.dtype = dtype
        self.output_size = output_size

        self.device = device

        self.q1_layer = nn.Linear(self.output_size, 1)
        self.q2_layer = nn.Linear(self.output_size, 1)

       

        if loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f'Unkown loss function name: {loss}')

        self.control_hidden_list = []
        self.control_h2o_list = []

        #self.l_00 = nn.Linear(self.input_size_state, self.hidden_size_1)
        self.l_00 = nn.Linear(self.input_size, self.hidden_size_1)
        self.h2o_0 = nn.Linear(self.hidden_size_2, self.output_size)
        #self.l_10 = nn.Linear(self.input_size_state, self.hidden_size_1)
        self.l_10 = nn.Linear(self.input_size, self.hidden_size_1)
        self.h2o_1 = nn.Linear(self.hidden_size_2, self.output_size)
        #self.l_20 = nn.Linear(self.input_size_state, self.hidden_size_1)
        self.l_20 = nn.Linear(self.input_size, self.hidden_size_1)
        self.h2o_2 = nn.Linear(self.hidden_size_2, self.output_size)
        #self.l_30 = nn.Linear(self.input_size_state, self.hidden_size_1)
        self.l_30 = nn.Linear(self.input_size, self.hidden_size_1)
        self.h2o_3 = nn.Linear(self.hidden_size_2, self.output_size)
        

        if self.n_layers == 2:
            #self.l_01 = nn.Linear(self.hidden_size_1+input_size_action, self.hidden_size_2).type(dtype)
            #self.l_11 = nn.Linear(self.hidden_size_1+input_size_action, self.hidden_size_2).type(dtype)
            #self.l_21 = nn.Linear(self.hidden_size_1+input_size_action, self.hidden_size_2).type(dtype)
            #self.l_31 = nn.Linear(self.hidden_size_1+input_size_action, self.hidden_size_2).type(dtype)
            self.l_01 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype)
            self.l_11 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype)
            self.l_21 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype)
            self.l_31 = nn.Linear(self.hidden_size_1, self.hidden_size_2).type(dtype)


        self.initialize()
        
        self.control_hidden_list.append([self.l_00, self.l_10, self.l_20, self.l_30])
        if n_layers == 2:
            self.control_hidden_list.append([self.l_01, self.l_11, self.l_21, self.l_31])

        self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

        self.hidden_list = []
        self.h2o_list = []
        self.phase_list = []

        if device.type == 'cuda':
            self.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
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
        if self.n_layers:
            init_fanin(self.l_01.weight)
            init_fanin(self.l_11.weight)
            init_fanin(self.l_21.weight)
            init_fanin(self.l_31.weight)


    def forward(self,state, action, phase):
        x = torch.cat([state, action], 1)
        #x_s = x[:,0:self.input_size_state]
        #x_a = x[:,self.input_size_state:]
        control_hidden_list = self.control_hidden_list
        control_h2o_list = self.control_h2o_list

        # Modulate by the phase
        w = spline_w(phase)
        w0,w1,w2,w3 = compute_multipliers(w,phase)

        w0_h1 = Variable(w0.repeat(1,self.hidden_size_1).type(self.dtype))
        w1_h1 = Variable(w1.repeat(1,self.hidden_size_1).type(self.dtype))
        w2_h1 = Variable(w2.repeat(1,self.hidden_size_1).type(self.dtype))
        w3_h1 = Variable(w3.repeat(1,self.hidden_size_1).type(self.dtype))

        if self.n_layers == 2:
            w0_h2 = Variable(w0.repeat(1,self.hidden_size_2).type(self.dtype))
            w1_h2 = Variable(w1.repeat(1,self.hidden_size_2).type(self.dtype))
            w2_h2 = Variable(w2.repeat(1,self.hidden_size_2).type(self.dtype))
            w3_h2 = Variable(w3.repeat(1,self.hidden_size_2).type(self.dtype))

        w0_o = Variable(w0.repeat(1,self.output_size).type(self.dtype))
        w1_o = Variable(w1.repeat(1,self.output_size).type(self.dtype))
        w2_o = Variable(w2.repeat(1,self.output_size).type(self.dtype))
        w3_o = Variable(w3.repeat(1,self.output_size).type(self.dtype))

        
        # Forward pass through the MLP
        h_0 = F.relu(w0_h1*control_hidden_list[0][0](x) + w1_h1*control_hidden_list[0][1](x) + w2_h1*control_hidden_list[0][2](x) + w3_h1*control_hidden_list[0][3](x))
        if self.n_layers == 2:
            #h_1 = F.relu(w0_h2*control_hidden_list[1][0](torch.cat((h_0,x_a),1)) + w1_h2*control_hidden_list[1][1](torch.cat((h_0,x_a),1)) + w2_h2*control_hidden_list[1][2](torch.cat((h_0,x_a),1)) + w3_h2*control_hidden_list[1][3](torch.cat((h_0,x_a),1)))
            h_1 = F.relu(w0_h2*control_hidden_list[1][0](h_0) + w1_h2*control_hidden_list[1][1](h_0) + w2_h2*control_hidden_list[1][2](h_0) + w3_h2*control_hidden_list[1][3](h_0))
            o = w0_o*control_h2o_list[0](h_1) + w1_o*control_h2o_list[1](h_1) + w2_o*control_h2o_list[2](h_1) + w3_o*control_h2o_list[3](h_1)
        else:
            o = w0_o*control_h2o_list[0](h_0) + w1_o*control_h2o_list[1](h_0) + w2_o*control_h2o_list[2](h_0) + w3_o*control_h2o_list[3](h_0)
        x1 = self.q1_layer(o)
        x2 = self.q2_layer(o)
        return x1, x2
    
    def reset(self):
        self.hidden_list = []
        self.h2o_list = []
        self.phase_list = []

    def weight_from_phase(self, phase, control_hidden_list, control_h2o_list):
        weight = {}
        w = spline_w(phase)
        for n in range(len(control_hidden_list)):
            for key in control_hidden_list[0][0]._parameters.keys():
                weight[key + '_' + str(n)] = control_hidden_list[n][kn(phase, 1)]._parameters[key].data + w*0.5*(control_hidden_list[n][kn(phase, 2)]._parameters[key].data - control_hidden_list[n][kn(phase, 0)]._parameters[key].data) + w*w*(control_hidden_list[n][kn(phase, 0)]._parameters[key].data - 2.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data + 2*control_hidden_list[n][kn(phase, 2)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_hidden_list[n][kn(phase, 1)]._parameters[key].data - 1.5*control_hidden_list[n][kn(phase, 2)]._parameters[key].data + 0.5*control_hidden_list[n][kn(phase, 3)]._parameters[key].data - 0.5*control_hidden_list[n][kn(phase, 0)]._parameters[key].data)
            for key in control_h2o_list[0]._parameters.keys():
                weight[key] = control_h2o_list[kn(phase, 1)]._parameters[key].data + w*0.5*(control_h2o_list[kn(phase, 2)]._parameters[key].data - control_h2o_list[kn(phase, 0)]._parameters[key].data) + w*w*(control_h2o_list[kn(phase, 0)]._parameters[key].data - 2.5*control_h2o_list[kn(phase, 1)]._parameters[key].data + 2*control_h2o_list[kn(phase, 2)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data) + w*w*w*(1.5*control_h2o_list[kn(phase, 1)]._parameters[key].data - 1.5*control_h2o_list[kn(phase, 2)]._parameters[key].data + 0.5*control_h2o_list[kn(phase, 3)]._parameters[key].data - 0.5*control_h2o_list[kn(phase, 0)]._parameters[key].data)
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
