import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp



class KRNN8(nn.Module):
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
        super(KRNN8, self).__init__()
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

class dila_conv(nn.Module):
    '''
    input:(batch_size , time_series_len, c_in)
    1dconv on time_dimension;kernel_size and dilation is the key
    output:(batch_size,  time_series_len ,c_out)
    '''
    def __init__(self,c_in,c_out, kernel_size,dilation):
        super(dila_conv,self).__init__()
        self.c_in=c_in
        self.c_out=c_out
        self.dilation=dilation
        self.kernel_size=kernel_size
        self.di_conv=nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size= kernel_size, dilation=dilation)
    def forward(self,x):
        # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len        
        x = x.permute(0,2,1)
        out=self.di_conv(x)
        out=out.permute(0,2,1)
        # batch_size x embedding_size x text_len   -> batch_size x text_len x embedding_size 
        return out

class local_conv_model(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, kernel_size=2,dilation_size=2,layers=3, hidden_size=64):
        """
        build one conv1d_model for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(local_conv_model, self).__init__()
        self.num_timesteps_output = num_timesteps_output
        self.num_timesteps_input = num_timesteps_input     
        self.kernel_size=kernel_size
        self.dilation_size=dilation_size
        self.layers=layers
        #h_size=num_timesteps_input-layers*kernel_size
        self.linears1= nn.ModuleList()
        self.linears2 = nn.ModuleList()
        self.conv1= nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.m = nn.ReLU(inplace=True)
        for _ in range(num_nodes):

            self.linears1.append(
                nn.Linear(in_features= 9,out_features=num_timesteps_output)
            )
            self.linears2.append(
                nn.Linear(num_features,1)
            ) 
            self.conv1.append(
                dila_conv(c_in=num_features,c_out=num_features,kernel_size=kernel_size,dilation=dilation_size)
            )

            self.conv2.append(
                dila_conv(c_in=num_features,c_out=num_features,kernel_size=kernel_size,dilation=dilation_size)
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
            input_sequence = X[:, n, :, :]
 
            hid= self.conv1[n](input_sequence)
            hid=self.m(hid) ###(batch,time_output,c_out)

            for l in range(self.layers-1):
                    hid=self.conv1[n](hid)
                    hid=self.m(hid) ###(batch,time_output,c_out)
            hid_size=hid.size() ###(batch,time_output,c_out)
  #          print('hid_size:',hid_size)
            hid = hid.permute(0,2,1).contiguous().view(-1,hid_size[1])
            hid= self.linears1[n](hid)
 



            hid2=hid.contiguous().view(size_x[0],size_x[3],-1).permute(0, 2, 1)###(batch,time_output,features)
            hid2=hid2.contiguous().view(-1,size_x[3])
            hid2=self.linears2[n](hid2)
            
            hid2=hid2.contiguous().view(size_x[0],self.num_timesteps_output) ###(batch,time_output)
            out.append(
                hid2.unsqueeze(dim=1)
            )



        out = torch.cat(out, dim=1)

        return out


class krnn_conv_local(nn.Module):  
    def __init__(self,num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, kernel_size=2,dilation_size=2,layers=3,gcn_type='normal', hidden_size=64):
        """
        build one linear_model for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(krnn_conv_local, self).__init__()

        self.globalrnn=KRNN7(num_nodes, num_features, num_timesteps_input,
        num_timesteps_output, gcn_type='normal', hidden_size=64, num_comps=10)
        self.local_linear=local_conv_model(num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, kernel_size=2,dilation_size=2,layers=3, hidden_size=64)


    def forward(self, A, X):

        out1=self.globalrnn(A,X)
        out2=self.local_linear(A,X)
        out=out1+out2
        return out


















