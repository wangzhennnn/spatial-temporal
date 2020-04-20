import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp



class KRNN2(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64, num_comps=10):
        """
        build one RNN for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(KRNN2, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.num_timesteps_output = num_timesteps_output

        for r in range(num_comps):
            self.encoders.append(
                nn.GRU(num_features, hidden_size)
            )
            self.decoders.append(
                nn.GRUCell(1, hidden_size)
            )
            self.linears.append(
                nn.Linear(hidden_size, 1)
            )

        self.embed = nn.Parameter(torch.FloatTensor(num_nodes, num_comps))
        self.embed.data.normal_()

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """

        out = []

        sz = X.size()
        X = X.view(-1, sz[2], sz[3]).permute(1, 0, 2)

        for i in range(len(self.encoders)):
            encoder_out, encoder_hid = self.encoders[i](X)
            decoder_out = []

            last_value = X[-1, :, 0].view(-1, 1)
            decoder_hid = encoder_hid.squeeze(dim=0)


            for step in range(self.num_timesteps_output):
                decoder_hid = self.decoders[i](last_value, decoder_hid)
                value = self.linears[i](decoder_hid)

                decoder_out.append(value)

                last_value = value.detach()

            decoder_out = torch.cat(decoder_out, dim=-1).view(sz[0], sz[1], self.num_timesteps_output)

            out.append(decoder_out.unsqueeze(dim=-1))
        
        out = torch.cat(out, dim=-1)
        weight = torch.softmax(self.embed, dim=-1)

        out = torch.einsum('ijkl,jl->ijk', out, weight)

        return out



class local_linear_model(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64):
        """
        build one linear_model for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(local_linear_model, self).__init__()
        self.num_timesteps_output = num_timesteps_output        
        self.linears1= nn.ModuleList()
        self.linears2 = nn.ModuleList()
        self.linears3 = nn.ModuleList()
        self.m = nn.ReLU(inplace=True)
        for _ in range(num_nodes):
            
            self.linears1.append(
                nn.Linear(num_timesteps_input, 2*num_timesteps_input)
            )
            self.linears3.append(
                nn.Linear(2*num_timesteps_input, num_timesteps_output)
            )
            self.linears2.append(
                nn.Linear(num_features,1)
            ) 

    def forward(self, A, X):

        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """
        out = []
        size_x=X.size()
       

        for n in range(X.size(1)):
            input_sequence = X[:, n, :, :].permute(0, 2, 1)
            input_sequence = input_sequence.contiguous().view(-1,size_x[2])
            hid= self.linears1[n](input_sequence)
            hid=self.m(hid) ###(batch*features,time_output)
            hid= self.linears3[n](hid)

            hid2=hid.contiguous().view(size_x[0],size_x[3],-1).permute(0, 2, 1)###(batch,time_output,features)
            hid2=hid2.contiguous().view(-1,size_x[3])
            hid2=self.linears2[n](hid2)
            hid2=self.m(hid2)
            hid2=hid2.contiguous().view(size_x[0],self.num_timesteps_output) ###(batch,time_output)
            out.append(
                hid2.unsqueeze(dim=1)
            )



        out = torch.cat(out, dim=1)

        return out


class krnn_local(nn.Module):  
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64):
        """
        build one linear_model for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(krnn_local, self).__init__()

        self.globalrnn=KRNN2(num_nodes, num_features, num_timesteps_input,
        num_timesteps_output, gcn_type='normal', hidden_size=64, num_comps=10)
        self.local_linear=local_linear_model( num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64)


    def forward(self, A, X):

        out1=self.globalrnn(A,X)
        out2=self.local_linear(A,X)
        out=out1+out2
        return out






















