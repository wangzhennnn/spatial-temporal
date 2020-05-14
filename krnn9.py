import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

class dila_conv1(nn.Module):
    '''
    input:(batch_size , time_series_len, c_in)
    1dconv on time_dimension;kernel_size and dilation is the key
    output:(batch_size,  time_series_len ,c_out)
    '''
    def __init__(self,c_in,c_out,dilation, kernel_size):
        super(dila_conv1,self).__init__()
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


class KRNN8(nn.Module):
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
        super(KRNN8, self).__init__()
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
                dila_conv1(c_in=num_features,c_out=num_features,kernel_size=kernel_size_set[r],dilation=dilation_size_set[r])
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


class dila_conv(nn.Module):
    '''
    input:(batch_size , time_series_len, c_in)
    1dconv on time_dimension;kernel_size and dilation is the key
    output:(batch_size,  time_series_len ,c_out)
    '''
    def __init__(self,c_in,c_out, kernel_size,dilation,group):
        super(dila_conv,self).__init__()
        self.c_in=c_in
        self.c_out=c_out
        self.dilation=dilation
        self.kernel_size=kernel_size
        self.di_conv=nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size= kernel_size, dilation=dilation,groups=group, bias=True)
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
     
        self.num_nodes=num_nodes
        #h_size=num_timesteps_input-layers*kernel_size
        #将每一维分开做卷积
        self.conv= dila_conv(c_in=num_nodes,c_out=num_nodes,kernel_size=kernel_size,dilation=dilation_size,group=num_nodes)
        self.m = nn.ReLU(inplace=True)

        ###线性变换可不可以通过卷积实现？？
        #self.linear=nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=, stride=1, padding=1, bias=True,groups=num_nodes)
        self.linear=nn.Linear(in_features= 9,out_features=num_timesteps_output)

        ###1*1卷积实现通道融合
        self.depth_linear=nn.Conv2d(in_channels= num_features, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, A, X):

        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """
        out = []
        size_x=X.size()
       
        input_seq=X.permute(0,3,2,1).contiguous().view(-1,size_x[2],size_x[1])


        hid= self.conv(input_seq)
        hid=self.m(hid) ###(batch*features,time_output,num_nodes)

        for l in range(self.layers-1):
                hid=self.conv(hid)
                hid=self.m(hid) ###(batch*features,time_output,num_nodes)
       
        hid=hid.permute(0,2,1).contiguous().view(size_x[0]*size_x[3]*size_x[1],-1) 
        hid_size=hid.size() ###(batch*features*num_nodes,time_output)
        
        hid= self.linear(hid)
       
        hid=hid.contiguous().view(size_x[0],size_x[3],size_x[1],-1).permute(0,1,3,2) ###(batch,features,num_nodes,time_output)
       
        out=self.depth_linear(hid)
       
        out=out.permute(0,3,2,1).contiguous().view(size_x[0],size_x[1],-1)
       
        return out

class dila_conv2(nn.Module):
    '''
    input:(batch_size ,  c_in,time_series_len)
    1dconv on time_dimension;kernel_size and dilation is the key
    output:(batch_size,  c_out,time_series_len )
    '''
    def __init__(self,c_in,c_out, kernel_size,dilation,group):
        super(dila_conv2,self).__init__()
        self.c_in=c_in
        self.c_out=c_out
        self.dilation=dilation
        self.kernel_size=kernel_size
        self.di_conv=nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size= kernel_size, dilation=dilation,groups=group, bias=True)
    def forward(self,x):
        #  batch_size x embedding_size x text_len        
 
        out=self.di_conv(x)

        # batch_size x embedding_size x text_len   
        return out

class krnn_conv_local1(nn.Module):  
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
        super(krnn_conv_local1, self).__init__()

        self.globalrnn=KRNN8(num_nodes, num_features, num_timesteps_input,num_timesteps_output,
     gcn_type='normal',kernel_size_set=[2,2,2,3,3,3,4,4,4], dilation_size_set=[1,2,4,1,2,4,1,2,4] ,hidden_size=64,num_comps=9)
        self.local_linear=local_conv_model(num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, kernel_size=2,dilation_size=2,layers=3, hidden_size=64)
        self.dil=dila_conv2(c_in=num_nodes,c_out=num_nodes,kernel_size=2,dilation=2,group=num_nodes)
    

        self.scale1 = torch.from_numpy(np.array(num_timesteps_input*num_timesteps_output)).float()   
        self.embed1 = nn.Parameter(torch.FloatTensor(num_timesteps_input-2,num_timesteps_output))
        self.embed1.data.normal_()

    def forward(self, A, X):
        query=X[:,:,:,0]
        query=self.dil(query)
    #    print(query.size())
        out1=self.globalrnn(A,X)
    #    out2=self.local_linear(A,X)
    #    out0=torch.stack((out1,out2),3)
    #    print(out1.size(),out2.size(),out0.size(),X.size())
    #    print('out1.size:',out1.size(),out2.size())
    #    weight = torch.softmax(self.embed1, dim=-1)
    #    weight=torch.einsum('ijk,kl -> ijl', query, self.embed1 )#(32,207,15),(15,3)->(32,207,3)
    #    weight=torch.einsum('ijl,ijlh -> ijh', weight, out0 )#(32,207,3),(32,207,3,2)->(32,207,2)
    #    weight = torch.softmax(weight, dim=-1)#(32,207,2)
    #    scale=self.scale1.sqrt()
    #    weight=torch.div( weight, scale)
    #    out = torch.einsum('ijkl,jl->ijk', out0, weight)
    #    out = torch.einsum('ijkl,ijl->ijk', out0, weight)
        out=out1
 
        return out



class krnn_conv_local2(nn.Module):  
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
        super(krnn_conv_local2, self).__init__()



        self.globalrnn=torch.load('net.pkl')  
        self.local_linear=local_conv_model(num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, kernel_size=2,dilation_size=2,layers=3, hidden_size=64)
        self.dil=dila_conv2(c_in=num_nodes,c_out=num_nodes,kernel_size=2,dilation=2,group=num_nodes)
    

    #    self.scale1 = torch.from_numpy(np.array(num_timesteps_input*num_timesteps_output)).float()   
    #    self.embed1 = nn.Parameter(torch.FloatTensor(num_timesteps_input-2,num_timesteps_output))
    #    self.embed1.data.normal_()

    def forward(self, A, X):
        query=X[:,:,:,0]
        query=self.dil(query)
    #    print(query.size())
   
        for p in self.globalrnn.parameters():
            p.requires_grad = False
        out1=self.globalrnn(X)
        out2=self.local_linear(A,X)
    #    out0=torch.stack((out1,out2),3)
    #    print(out1.size(),out2.size(),out0.size(),X.size())
    #    print('out1.size:',out1.size(),out2.size())
    #    weight = torch.softmax(self.embed1, dim=-1)
    #    weight=torch.einsum('ijk,kl -> ijl', query, self.embed1 )#(32,207,15),(15,3)->(32,207,3)
    #    weight=torch.einsum('ijl,ijlh -> ijh', weight, out0 )#(32,207,3),(32,207,3,2)->(32,207,2)
    #    weight = torch.softmax(weight, dim=-1)#(32,207,2)
    #    scale=self.scale1.sqrt()
    #    weight=torch.div( weight, scale)
    #    out = torch.einsum('ijkl,jl->ijk', out0, weight)
    #    out = torch.einsum('ijkl,ijl->ijk', out0, weight)
        out=out1+out2
        return out




















