'''
Main Models
Author: Pu Zhang
Date: 2019/7/1
'''

from utils import *
from basemodel import *

class SR_LSTM(nn.Module):
    def __init__(self, args):
        super(SR_LSTM, self).__init__()
        self.args = args
        self.ifdropout = args.ifdropout
        self.using_cuda = args.using_cuda
        self.inputLayer = nn.Linear(args.input_size, args.input_embed_size)
        self.cell = LSTMCell(args.input_embed_size, args.rnn_size)

        self.gcn = GCN(args,self.args.rela_embed_size, args.rnn_size)

        if self.args.passing_time>1:
            self.gcn1 = GCN(args, self.args.rela_embed_size, args.rnn_size)
            if self.args.passing_time==3:
                self.gcn2 = GCN(args, self.args.rela_embed_size, args.rnn_size)

        self.outputLayer = nn.Linear(args.rnn_size, args.output_size)
        self.dropout = nn.Dropout(args.dropratio)

        self.input_Ac = nn.ReLU()

        if args.using_cuda:
            self = self.cuda(device=args.gpu)
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant(self.inputLayer.bias, 0.0)
        nn.init.normal(self.inputLayer.weight, std=self.args.std_in)

        nn.init.xavier_uniform(self.cell.weight_ih)
        nn.init.orthogonal(self.cell.weight_hh, gain=0.001)

        nn.init.constant(self.cell.bias_ih, 0.0)
        nn.init.constant(self.cell.bias_hh, 0.0)
        n = self.cell.bias_ih.size(0)
        nn.init.constant(self.cell.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant(self.outputLayer.bias, 0.0)
        nn.init.normal(self.outputLayer.weight, std=self.args.std_out)

    def forward(self, inputs,iftest=False):

        nodes_abs, nodes_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum=inputs
        num_Ped = nodes_norm.shape[1]
        
        outputs=torch.zeros(nodes_norm.shape[0],num_Ped, self.args.output_size)
        hidden_states = torch.zeros(num_Ped, self.args.rnn_size)
        cell_states = torch.zeros(num_Ped, self.args.rnn_size)

        value1_sum=0
        value2_sum=0
        value3_sum=0

        if self.using_cuda:
            outputs=outputs.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()
            
        # For each frame in the sequence
        for framenum in range(self.args.seq_length-1):
<<<<<<< HEAD
             #changed by LMM :get speed mask
            seq_length = outputs.shape[0]
            node_pre=seq_list[0]
            speedmask=torch.zeros(seq_length,seq_list.shape[1])
            speedmask=speedmask.cuda()
            for i in range(seq_length):
                speedmask[i]=seq_list[i]*node_pre
                if i>0:
                    node_pre=seq_list[i-1]
            # speedmask=np.repeat(speedmask, 2, axis=1)
=======
                       #changed by LMM :使用相对坐标实验
            # if framenum==0:
            #     outputs[framenum,:]=nodes_norm[framenum,:]
            # else:
>>>>>>> 6d34743f331e4bf6af35e2fb7c2db236dbe0bb5e
            if framenum >= self.args.obs_length and iftest:   #inference阶段
                #不同的node代表不同行人的轨迹
                node_index = seq_list[self.args.obs_length - 1] > 0 #当至少存在一步观察轨迹时，我们才认为这段轨迹是有效轨迹
                nodes_current = outputs[framenum - 1, node_index].clone()  #取有效轨迹上一帧的输出，使用上一帧的输出作为当前帧输入
<<<<<<< HEAD
                if nodes_current.shape[0]==0:
                    continue
                nodes_abs=shift_value[framenum,node_index]+nodes_current   #绝对轨迹值
                nodes_abs_nei=nodes_abs_nei.repeat(nodes_abs.shape[0], 1, 1)
                #此transpose(0,1)并不对corr造成任何变化
                corr_index=nodes_abs_nei.transpose(0,1)-nodes_abs_nei    #Correction

                nodes_abs_pre=outputs[framenum - 2, node_index].clone()+shift_value[framenum-1,node_index]
                nodes_speed=(nodes_abs[:,:]-nodes_abs_pre[:,:])
                nodes_ori= torch.atan2(nodes_speed[:,1], nodes_speed[:, 0])
                nodes_velocity=torch.norm(nodes_speed,dim=1)* speedmask[framenum]
            else: #训练阶段
                node_index=seq_list[framenum]>0    #当前帧有轨迹时，我们才使用这段轨迹 
                nodes_current = nodes_norm[framenum,node_index]  #使用GT作为输入
                # print(nodes_abs[framenum, node_index].shape,nodes_current.shape[0])
                if nodes_current.shape[0]==0:
                    continue
=======
                nodes_abs=shift_value[framenum,node_index]+nodes_current   #绝对轨迹值
                nodes_abs=nodes_abs.repeat(nodes_abs.shape[0], 1, 1)
                #此transpose(0,1)并不对corr造成任何变化
                corr_index=nodes_abs.transpose(0,1)-nodes_abs    #Correction，shift the origin to the latest observed time step
                # print("corr_index",corr_index)
            else: #训练阶段
                node_index=seq_list[framenum]>0   #当前帧有轨迹时，我们才使用这段轨迹
                nodes_current = nodes_norm[framenum,node_index]  #使用GT作为输入
                if nodes_current.shape[0]==0:
                    continue
                # print("nodes_current.shape",nodes_current.shape)
                # print("nodes_current.shape[0]",nodes_current.shape[0])
>>>>>>> 6d34743f331e4bf6af35e2fb7c2db236dbe0bb5e
                corr = nodes_abs[framenum, node_index].repeat(nodes_current.shape[0], 1, 1)
                # print("corr.shape",corr.shape)
                nei_index = nei_list[framenum, node_index]
                nei_index = nei_index[:, node_index]
                # relative coords
                corr_index = corr.transpose(0,1)-corr  
                nei_num_index=nei_num[framenum,node_index]
<<<<<<< HEAD
                if framenum>0:
                    nodes_abs_pre=nodes_abs[framenum-1, node_index]
                    # print(speedmask[framenum].shape,nodes_abs[framenum, node_index].shape,nodes_abs_pre[:,:].shape)
                    nodes_speed=(nodes_abs[framenum, node_index]-nodes_abs_pre[:,:])

                    nodes_ori= torch.atan2(nodes_speed[:,1], nodes_speed[:, 0])* speedmask[framenum, node_index]
                    nodes_velocity=torch.norm(nodes_speed,dim=1)* speedmask[framenum, node_index]
                else:
                    nodes_abs_pre=nodes_abs[framenum, node_index]
                    nodes_speed=torch.zeros_like(nodes_abs[framenum, node_index])
                    nodes_ori=torch.zeros_like(nodes_abs[framenum, node_index][:,0])* speedmask[framenum, node_index]
                    nodes_velocity=torch.norm(nodes_speed,dim=1)* speedmask[framenum, node_index]
            # print("framenum>",framenum,"nodes_speed","",nodes_speed,"nodes_ori",nodes_ori,"nodes_velocity",nodes_velocity)
            # print("nodes_abs[framenum, node_index]-nodes_abs_pre[:,:]",nodes_abs[framenum, node_index],nodes_abs_pre[:,:])
=======
                # print("corr.transpose(0,1)[2,3]",corr.transpose(0,1)[2,3],"corr[2,3]",corr[2,3],"corr.transpose(0,1)[3,2]",corr.transpose(0,1)[3,2])
>>>>>>> 6d34743f331e4bf6af35e2fb7c2db236dbe0bb5e
            hidden_states_current=hidden_states[node_index]
            cell_states_current=cell_states[node_index]
            #TODO:更改这里可以选择使用相对坐标还是绝对坐标
            input_embedded = self.dropout(self.input_Ac(self.inputLayer(nodes_current)))
            lstm_state = self.cell.forward(input_embedded, (hidden_states_current,cell_states_current))

            for p in range(self.args.passing_time ):
                if p==0:
                    lstm_state, look = self.gcn.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn.W_nei)
                    value1, value2, value3 = look
                if p==1:
                    lstm_state, look = self.gcn1.forward(corr_index, nei_index, nei_num_index, lstm_state,self.gcn1.W_nei)


            _, hidden_states_current, cell_states_current = lstm_state

            value1_sum+=value1
            value2_sum+=value2
            value3_sum+=value3

            outputs_current = self.outputLayer(hidden_states_current)
            outputs[framenum,node_index]=outputs_current      #更新当前帧有效轨迹的输出
            hidden_states[node_index]=hidden_states_current  #更新当前帧有效轨迹的h
            cell_states[node_index] = cell_states_current    #更新当前帧有效轨迹的c
            # print("outputs",outputs)

        return outputs, hidden_states, cell_states,(value1_sum/self.args.seq_length,value2_sum/self.args.seq_length,value3_sum/self.args.seq_length)



