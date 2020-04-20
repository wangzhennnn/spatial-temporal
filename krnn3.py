import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

###TODO1: set multi dimension.
###TODO2: attentiom mech.



class KRNN3(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, gcn_type='normal', hidden_size=64, size=[3,4,5,6,7,8,9,10,11,12],num_comps=10):
        """
        build one RNN for each time series
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(KRNN3, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.linears1 = nn.ModuleList()
        self.linears2 = nn.ModuleList()

        self.num_timesteps_output = num_timesteps_output
        self.hidden_size=hidden_size
        self.num_comps=num_comps
        self.size=size
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
            self.linears2.append(
                nn.Linear(num_timesteps_input, size[r])
            )

#        self.embed = nn.Parameter(torch.FloatTensor(num_nodes, num_comps))
#        self.embed.data.normal_()
        
        self.embed1 = nn.Parameter(torch.FloatTensor(num_timesteps_input,num_timesteps_output))
        self.embed1.data.normal_()

    def forward(self, A, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: deprecated in pure TS model.
        """

        out = []
        query=X[:,:,:,0]
#        print('query:',query.size()) 
        sz = X.size()
        X = X.contiguous().view(-1, sz[2], sz[3]).permute(1, 0, 2)
        

        for i in range(len(self.encoders)):
            h=X.permute(1,2,0).contiguous().view(-1,sz[2])
           
            h_val= self.linears2[i](h)
            
            h_val=h_val.contiguous().view(-1,sz[3],self.size[i]).permute(2,0,1)
                  
            encoder_out, encoder_hid = self.encoders[i](h_val)
            decoder_out = []

            last_value = X[-1, :, 0].contiguous().view(-1, 1)
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

        weight=torch.einsum('ijk,kl -> ijl', query, self.embed1 )#(32,207,15),(15,3)->(32,207,3)
        weight=torch.einsum('ijl,ijlh -> ijh', weight, out )#(32,207,3),(32,207,3,10)->(32,207,10)
        weight = torch.softmax(weight, dim=-1)#(32,207,10)

        out = torch.einsum('ijkl,ijl->ijk', out, weight)

#        print('out_size2:',out.size())

        return out







