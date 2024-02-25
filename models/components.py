import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from models.utils import *

class NormalizedReLU(nn.Module):
    def __init__(self, dim=-1):
        super(NormalizedReLU, self).__init__()
        
        print(dim, type(dim))
        
        self.eps = 1e-5
        self.dim = dim
        self.relu = nn.ReLU()
        
    def forward(self, data):

        num = self.relu(data)
        den = torch.max(self.relu(data),self.dim,keepdim=True)[0] + self.eps
        return num/den

def make_activation(activation):
    
    if "nrelu" in activation:
        return NormalizedReLU(int(re.split('=', activation)[1]))
        
    elif activation == "relu":
        return nn.ReLU()
        
    else:
        print("Unknown activation:",activation)
        sys.exit()
    
def make_mlp(dim_list, activations, dropout=0):

    if len(dim_list) == 0 and len(activations) == 0:
        return nn.Identity()

    assert len(dim_list) == len(activations)+1
    
    layers = []
    for dim_in, dim_out, activation in zip(dim_list[:-1], dim_list[1:], activations):
                
        # append layer
        layers.append(nn.Linear(dim_in, dim_out))
        
        # # # # # # # # # # # # 
        # append activations  #
        # # # # # # # # # # # #
            
        activation_list = re.split('-', activation)
        for activation in activation_list:
            
            # first because of "in"
            if 'leakyrelu' in activation:
                layers.append(nn.LeakyReLU(negative_slope=float(re.split('=', activation)[1]), inplace=True))
                
            elif activation == 'relu':
                layers.append(nn.ReLU())
                
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
                
            elif activation == "none":
                pass
                                
            elif activation == "batchnorm":
                layers.append(nn.BatchNorm1d(dim_out))    
                        
            else:
                print("unknown activation")
                sys.exit()
            
            if dropout > 0:
                print("dropout")
                layers.append(nn.Dropout(p=dropout))
                   
    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self, units, kernels, paddings, activations):
        super(CNN, self).__init__()
        #for key, value in args.__dict__.items():
        #    setattr(self, key, value)
        
        assert len(units) == len(activations)+1
        assert len(units) == len(kernels)+1
        assert len(kernels) == len(paddings)
        
        layers = []
        for dim_in, dim_out, kernel, pad, activation in zip(units[:-1], units[1:], kernels, paddings, activations):
            layers.append(nn.Conv1d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, padding=pad))
            if activation == "relu":
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        
    def forward(self, data):
        
        # data must be [batch, seq len, channels]
        # input shape must be [batch, channels, length]
        
        data = data.permute(0,2,1)
        data = self.model(data)
        data = data.permute(0,2,1)
        
        return data
            
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.model = eval("nn."+self.gaze_encoder_type)(self.gaze_encoder_units[0],self.gaze_encoder_units[1],self.gaze_encoder_units[2],batch_first=False)
    
    def init_hidden(self):
        # [num_layers, batch, hidden_size]
        return (torch.zeros(self.gaze_encoder_units[2], self.padded_length*1, self.gaze_encoder_units[1]).cuda(), torch.zeros(self.gaze_encoder_units[2], self.padded_length*1, self.gaze_encoder_units[1]).cuda())
        #return (torch.zeros(self.gaze_encoder_units[2], self.batch_size*10, self.gaze_encoder_units[1]).cuda(), torch.zeros(self.gaze_encoder_units[2], self.batch_size*10, self.gaze_encoder_units[1]).cuda())
    
    def forward(self, data):
    
        # data = [1*self.padded_length, gaze_length, 3]
        data = data.permute(1,0,2) # [gaze_length, 1*self.padded_length, 3]
        
        hi = self.init_hidden()
        h = []
        for d in data:
            _, hi = self.model(torch.unsqueeze(d,0),hi)
            h.append(hi[0])
        h = torch.cat(h) # [gaze_length, batch*15, 128]
        
        return h.permute(1,0,2) # return [batch*15, gaze_length, 128]

# attention over the gaze relative to each object / grid
class multi_attention(nn.Module):
    def __init__(self, units, activations, padded_length):
        super(multi_attention, self).__init__()
        self.fc = make_mlp(units, activations)
        self.padded_length = padded_length
        
    def forward(self, data, unpadded_length):
                
        # truncate data
        data = data[:unpadded_length] # data = [num_objects/grid, gaze_length, hidden_size]
                
        # reshape
        data = data.view(-1, unpadded_length, data.shape[1], data.shape[2]) # [batch (1), num_objects/grid, gaze_length, hidden size]   
        
        # attention
        scores = self.fc(data)                       # [batch (1), num_objects/grid, gaze_length, 1]
        scores = F.softmax(scores,dim=-2)            # [batch (1), num_objects/grid, gaze_length, 1] torch.sum(scores[0,0]) = 1 
        data = data * scores                         # [batch (1), num_objects/grid, gaze_length, hidden size]
        data = torch.sum(data,dim=-2)                # [batch (1), num_objects/grid, hidden size]
         
        # pad it back
        scores_pad = torch.zeros([scores.shape[0], self.padded_length, scores.shape[2], scores.shape[3]], dtype=scores.dtype, device=scores.device) # [1, 100, gaze_length, 1]
        scores_pad[:scores.shape[0], :scores.shape[1], :scores.shape[2], :scores.shape[3]] = scores
        data_pad = torch.zeros([data.shape[0], self.padded_length, data.shape[2]], dtype=data.dtype, device=data.device) # [1, 100, hidden_size]
        data_pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        
        return scores_pad, data_pad

# attention over each object
class attention(nn.Module):
    def __init__(self, units, activations, padded_length):
        super(attention, self).__init__()
        
        assert len(units) != 0
        assert len(activations) != 0
        
        self.fc = make_mlp(units, activations)
        self.padded_length = padded_length
        
    def forward(self, data, unpadded_length):
    
        # truncate data
        data = data[:,:unpadded_length] # data = [batch (1), num_objects/grid, hidden_size]
                     
        # attention   
        scores = self.fc(data)             # [batch (1), num_objects/grid, 1]
        scores = F.softmax(scores,dim=-2)  # [batch (1), num_objects/grid, 1]
        data = data * scores               # [batch (1), num_objects/grid, feature dim]
        data = torch.sum(data,dim=-2)      # [batch (1), feature dim]
        
        # pad it back
        scores_pad = torch.zeros([scores.shape[0], self.padded_length, scores.shape[2]], dtype=scores.dtype, device=scores.device) # [1, 100, 1]
        scores_pad[:scores.shape[0], :scores.shape[1], :scores.shape[2]] = scores
                        
        return scores_pad, data

# create GRU hidden state        
def init_hidden(batch_size, hidden_units, num_layers, bidirectional):
    # [num_layers, batch, hidden_size]
    num_directions = 2 if bidirectional == 1 else 1
    return torch.zeros(num_directions*num_layers, batch_size, hidden_units).cuda()

# # # # # # # # # # # # # # # # # # # #
# construct the mask for padded data  #
# # # # # # # # # # # # # # # # # # # #
def get_key_padding_mask(unpadded_length, padded_length):

    # todo 
    # - convert to bool or Byte

    # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    # non-zero positions will be ignored (masked) by the attention 
    mask = torch.ones(unpadded_length.shape[0], padded_length).to(device=torch.cuda.current_device()) # [batch, pose_padded_length]
    for i,length in enumerate(unpadded_length):
        mask[i,:length] = 0
    mask = mask.bool()
    return mask
    
# # # # # # # # # # # # # # # #
# cosine positional embedder  #
# # # # # # # # # # # # # # # #
def progress_ratio_position_embedder(sequence_length, unpadded_lengths, padded_length, units, reverse):
    
    pr = [torch.linspace(0,1,length.item()) for length in sequence_length]      # [batch, padded_lengths, units]
    pr = [torch.unsqueeze(pr[i][-l:],1) for i,l in enumerate(unpadded_lengths)] # [batch, unpadded_lengths, units]
        
    pr = pad(pr, unpadded_lengths, padded_length)
    return pr

# # # # # # # # # # # # # # # #
# cosine positional embedder  #
# # # # # # # # # # # # # # # #
def cosine_positional_embedder1(batch_size, padded_length, units):

    # create positional embedding for one vector
    position = torch.arange(padded_length).unsqueeze(1)                                             # [pose_padded_length, 1]
    div_term = torch.exp(torch.arange(0, units, 2) * (-torch.log(torch.Tensor([10000.0])) / units)) # [units/2]
    pe = torch.zeros([padded_length, 1, units])                         # [padded_length, 1, units]
    pe[:, 0, 0::2] = pe[:, 0, 0::2] + torch.sin(position * div_term)    # [padded_length, 1, units]
    pe[:, 0, 1::2] = pe[:, 0, 1::2] + torch.cos(position * div_term)    # [padded_length, 1, units]
    pe = pe[:,0,:]                                                      # [padded_length, units]    
    pe = pe[None,:,:].repeat(batch_size,1,1)                            # [batch_size, padded_length, units] 
    pe = pe.to(device=torch.cuda.current_device())
    return pe

# no need to unpad because we use a mask input to the transformer
# - wrong because i use padded lengths ? should I have used unpadded lengths to create the sine wave before padding it ?
def cosine_positional_embedder(unpadded_lengths, padded_length, units):

    # create positional embedding for one vector
    position = torch.arange(padded_length).unsqueeze(1)                                             # [pose_padded_length, 1]
    div_term = torch.exp(torch.arange(0, units, 2) * (-torch.log(torch.Tensor([10000.0])) / units)) # [units/2]
    pe = torch.zeros(size=(padded_length, 1, units))    # [padded_length, 1, units]
    pe[:, 0, 0::2] = torch.sin(position * div_term)     # [padded_length, 1, units]
    pe[:, 0, 1::2] = torch.cos(position * div_term)     # [padded_length, 1, units]
    pe = pe[:,0,:]                                      # [padded_length, units]    
        
    # create positional embedding for every sample
    pe = [pe[:l] for l in unpadded_lengths]             # [batch, unpadded_lengths, units]
    
    # sanity check
    """for i in range(len(pe)):
        for j in range(pe[i].shape[0]):
            print(j,torch.sum(torch.abs(pe[i][j])))
    sys.exit()"""
      
    #if reverse:
    #    pe = [torch.flip(x,dims=[0]) for x in pe]
    # before reverse [ 0.9999,  0.5291, -0.4282, -0.9918, -0.6435,  0.2964,  0.9638,  0.7451, -0.1586, -0.9165, -0.8318,  0.0177,  0.8509,  0.9018,  0.1236, -0.7683, -0.9538]  
    # after reverse  [-0.9538, -0.7683,  0.1236,  0.9018,  0.8509,  0.0177, -0.8318, -0.9165, -0.1586,  0.7451,  0.9638,  0.2964, -0.6435, -0.9918, -0.4282,  0.5291,  0.9999]
    
    pe = pad_variable_lengths(pe, unpadded_lengths, padded_length)    
    return pe
    
# no need to unpad because we use a mask input to the transformer
def cosine_positional_embedder2(unpadded_lengths, padded_length, units):

    pe_list = []
    for unpadded_length in unpadded_lengths:
        position = torch.arange(unpadded_length).unsqueeze(1)                                           # [unpadded_length, 1]
        div_term = torch.exp(torch.arange(0, units, 2) * (-torch.log(torch.Tensor([10000.0])) / units)) # [units/2]
        pe = torch.zeros(size=(unpadded_length, units)) # [unpadded_length, units]
        pe[:, 0::2] = torch.sin(position * div_term)    # [unpadded_length, units]
        pe[:, 1::2] = torch.cos(position * div_term)    # [unpadded_length, units]
        pe = pad(pe,padded_length)                      # [padded_length, units]
        pe_list.append(pe)
    pe = torch.stack(pe_list)                           # [batch, padded_length, units]
    return pe