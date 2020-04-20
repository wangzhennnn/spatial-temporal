import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import sys

###TODO1: set multi dimension by dilated convolution
###TODO2: attentiom mech.


class Linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(Linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class dila_conv(nn.Module):
    '''
    input:(batch_size , time_series_len, c_in)
    1dconv on time_dimension;kernel_size and dilation is the key
    output:(batch_size,  time_series_len ,c_out)
    '''
    def __init__(self,c_in,c_out,dilation, kernel_size):
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
        out=out.permute(2,0,1)
        # batch_size x embedding_size x text_len   ->  text_len x  batch_size x embedding_size
        return out


class KRNN4(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,num_timesteps_output,
     gcn_type='normal',kernel_size_set=[2,2,2,3,3,3,4,4,4], dilation_size_set=[1,2,4,1,2,4,1,2,4] ,hidden_size=64,num_comps=9):
        """
        build one RNN for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(KRNN4, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.linears1 = nn.ModuleList()
        self.dila_conv1 = nn.ModuleList()

        self.num_timesteps_output = num_timesteps_output
        self.hidden_size=hidden_size
        self.num_comps=num_comps
        self.kernel_size_set=kernel_size_set
        self.dilation_size_set=dilation_size_set

        for r in range(num_comps):
            self.encoders.append(
                nn.GRU(num_features, hidden_size)
            )
            self.decoders.append(
                nn.GRUCell(1, hidden_size)
            )
            self.linears1.append(
                nn.Linear(hidden_size, 1)
            )
            self.dila_conv1.append(
                dila_conv(c_in=num_features,c_out=num_features,kernel_size=kernel_size_set[r],dilation=dilation_size_set[r])
            )

        self.embed = nn.Parameter(torch.FloatTensor(num_nodes, num_comps))
        self.embed.data.normal_()
        
#        self.embed1 = nn.Parameter(torch.FloatTensor(num_timesteps_input,num_timesteps_output))
#        self.embed1.data.normal_()

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """

        out = []
#        query=X[:,:,:,0]
#        print('query:',query.size()) 
        sz = X.size()
        X = X.contiguous().view(-1, sz[2], sz[3])
        X1=X.permute(1, 0, 2)

        for i in range(len(self.encoders)):

            h_val= self.dila_conv1[i](X)

 #           h_val=h_val.permute( 1,0, 2)
 #           print(X1.size(),h_val.size())               
            encoder_out, encoder_hid = self.encoders[i]( h_val)
            decoder_out = []

            last_value = X1[-1, :, 0].contiguous().view(-1, 1)
            decoder_hid = encoder_hid.squeeze(dim=0)
            '''
            print('decoder_hid:',decoder_hid.size())
            value2=self.linears1[i](decoder_hid)
            value2=value2.contiguous().view(sz[0], sz[1], -1)
            print('value2:',value2.size())
            '''           
            for step in range(self.num_timesteps_output):
                decoder_hid = self.decoders[i](last_value, decoder_hid)
                value = self.linears1[i](decoder_hid)
           
                decoder_out.append(value)

                last_value = value.detach()

            decoder_out = torch.cat(decoder_out, dim=-1).contiguous().view(sz[0], sz[1], self.num_timesteps_output)

            out.append(decoder_out.unsqueeze(dim=-1))
            
        out = torch.cat(out, dim=-1)
        weight = torch.softmax(self.embed, dim=-1)

        out = torch.einsum('ijkl,jl->ijk', out, weight)

#        weight=torch.einsum('ijk,kl -> ijl', query, self.embed1 )#(32,207,15),(15,3)->(32,207,3)
#        weight=torch.einsum('ijl,ijlh -> ijh', weight, out )#(32,207,3),(32,207,3,10)->(32,207,10)
#       weight = torch.softmax(weight, dim=-1)#(32,207,10)
        
#        out = torch.einsum('ijkl,ijl->ijk', out, weight)

#        print('out_size2:',out.size())

        return out







