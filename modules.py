import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    '''
    d_m: hidden dimension
    num_heads: number of different heads used in the attention
    d_k: dimension of internal keys/queries. If not set, d_k will be calculated as d_m // num_heads
    d_v: dimension of internal keys/queries. If not set, d_k will be calculated as d_m // num_heads
    '''
    def __init__(self, d_m, num_heads, d_k=None, d_v=None):
        # dimension of query and key (d_k) can be different from value dimension d_v
        # all heads have same d_k and d_v to be able to store them in 4d-Tensor and use torch.einsum
        super(MultiheadAttention, self).__init__()
        self.d_m = d_m
        self.numHeads = num_heads
        if d_k is not None:
            self.d_k = d_k
        else:
            assert d_m % num_heads == 0, 'd_m needs to be divisible by num_heads'
            self.d_k = d_m // num_heads
        if d_v is not None:
            self.d_v = d_v
        else:
            assert d_m % num_heads == 0, 'd_m needs to be divisible by num_heads'
            self.d_v = d_m // num_heads

        # The linear projections from the different heads are stacked to be stored as a single
        # dense-layer
        self.W_k = nn.Linear(d_m, self.d_k*num_heads, bias=False)
        self.W_v = nn.Linear(d_m, self.d_v*num_heads, bias=False)
        self.W_q = nn.Linear(d_m, self.d_k*num_heads, bias=False)
        # Transform concatenated heads back into dimensions of the query
        self.W_o = nn.Linear(self.d_v*num_heads, d_m, bias=False)
        
    def forward(self, query, key_value, mask=None):
        '''
        query: tensor used to create queries from                       [b, m, d_m]
        key_value: the same tensor is used to create keys and values    [b, n, d_m]
        mask: apply masked attention if specified. Mask must be of dimensions [..., m, n], so that it
              can be broadcasted by pytorch to a 4D-attention-tensor [b, h, m, n]
        '''
        b = query.shape[0]
        m = query.shape[1]      # sequence length of query
        n = key_value.shape[1]  # sequence length of key/value (n=m if self-attention)

        # create queries, keys and values 
        Q = self.W_q(query).    reshape(b, m, self.numHeads, self.d_k).permute(0,2,1,3) # [b, m, d_k*h] -> [b, h, m, d_k]
        K = self.W_k(key_value).reshape(b, n, self.numHeads, self.d_k).permute(0,2,1,3) # [b, n, d_k*h] -> [b, h, n, d_k]
        V = self.W_v(key_value).reshape(b, n, self.numHeads, self.d_v).permute(0,2,1,3) # [b, n, d_k*h] -> [b, h, n, d_k]

        # dot-product attention  Q * (K)^T
        attention = torch.einsum("bhmk,bhnk->bhmn", [Q, K])   # [b, h, m, n]

        # apply mask if specified. Mask must be of dimensions (..., m, n), so that it
        # can be broadcasted by pytorch to be applied to 4D-attention-tensor
        if mask is not None:  
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        # normalize and apply softmax row-wise
        attention = torch.softmax(attention / (self.d_k**(1/2)), dim=-1)
        # A' * V
        attention = torch.einsum("bhmn,bhnv->bhmv", [attention, V])   # [b, h, m, d_v]
        attention = attention.permute(0,2,1,3).reshape(b, m, self.d_v*self.numHeads)    # [b, h, m, d_v] -> [b, m, d_v*h]

        # linear project to hidden dimension
        return self.W_o(attention)  #[b, m, d_m]
    

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0, 'dim must be divisible by 2'
        self.dim = dim
        # register div as non-trainable parameter (moved to device with model) that should not be loaded/saved
        self.register_buffer('div', torch.pow(theta, 2*torch.arange(0, dim//2) / dim), persistent=False)

    def forward(self, pos):
        """ pos: input positions (B, N) -> (B, N, d_m)
        """
        B, N = pos.shape
        pos = pos.unsqueeze(2)
        out = torch.empty(B, N, self.dim, device=pos.device)
        out[:, :, 0::2] = torch.sin(pos/self.div)
        out[:, :, 1::2] = torch.cos(pos/self.div)
        return out
    

# TODO remove old sinusoidal embedding and test my imp
class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_m, numberOfTokenIDs, base, device):
        super(SinusoidalEmbedding, self).__init__()
        self.d_m = d_m
        self.base = base
        self.device = device
        self.embedding = nn.Embedding(numberOfTokenIDs, d_m)

    # Position encoding. For more info see
    # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    def encodePosition(self, sequenceLen, d_m, n):
        # Initialize empty matrix
        mat = torch.empty((sequenceLen, d_m))
        # Iterate over embedding dimension (column)
        for i in range(int(d_m/2)):
            denoms = torch.arange(sequenceLen)/(n**(2*i/d_m))
            # Set two colums
            mat[:, 2*i]   = torch.sin(denoms)
            mat[:, 2*i+1] = torch.cos(denoms)
        return mat.to(self.device)
    
    def forward(self, x):
        x = self.embedding(x)
        x += self.encodePosition(x.shape[1], self.d_m, self.base)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, d_m, num_heads, dropout, ff_expansion, d_k=None, d_v=None):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_m, num_heads, d_k=d_k, d_v=d_v)
        self.norm = nn.LayerNorm(d_m)
        self.ff = nn.Sequential(
            nn.Linear(d_m, d_m*ff_expansion),
            nn.ReLU(),
            nn.Linear(d_m*ff_expansion, d_m)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        '''
        x: input tensor [b, n, d_m]
        mask: mask to use in self-attention
        '''
        out = self.self_attn(x, x, mask)
        x = self.dropout(self.norm(out + x))    # Norm + skip-connection + dropout

        out = self.ff(x)
        return self.dropout(self.norm(out + x))
    

class DecoderBlock(nn.Module):
    def __init__(self, d_m, num_heads, dropout, ff_expansion, d_k=None, d_v=None):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_m, num_heads, d_k=d_k, d_v=d_v)
        self.enc_dec_attn = MultiheadAttention(d_m, num_heads, d_k=d_k, d_v=d_v)
        self.norm = nn.LayerNorm(d_m)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_m, d_m*ff_expansion),
            nn.ReLU(),
            nn.Linear(d_m*ff_expansion, d_m)
        )
    
    def forward(self, x, enc_kv, self_attn_mask=None, cross_attn_mask=None):
        '''
        x: input tensor [b, n, d_m]
        enc_kv: encoding used for keys and values in cross-attention [b, n, d_m]
        self_attn_mask: mask to use in self attention
        cross_attn_mask: mask to use for cross-attention between encoder and decoder
        '''
        out = self.self_attn(x, x, self_attn_mask)
        x = self.dropout(self.norm(out + x))    # Norm + skip-connection + dropout

        out = self.enc_dec_attn(x, enc_kv, cross_attn_mask)
        x = self.dropout(self.norm(out + x))

        out = self.ff(x)
        return self.dropout(self.norm(out + x))
    

class Transformer(nn.Module):
    def __init__(self, d_m, vocabSize, dropout, expansion, numEncoders, numDecoders, 
                 numEncoderHeads, numDecoderHeads, device, inputEmbedding, outputEmbedding, d_k=None, d_v=None):
        super(Transformer, self).__init__()
        self.d_m = d_m
        self.inputEmbedding = inputEmbedding
        self.outputEmbedding = outputEmbedding
        self.device = device
        # Use nn.ModuleList so that modules' parameters are stored on correct device
        self.encoders = nn.ModuleList([EncoderBlock(d_m, numEncoderHeads, dropout, expansion, d_k, d_v) for _ in range(numEncoders)])
        self.decoders = nn.ModuleList([DecoderBlock(d_m, numDecoderHeads, dropout, expansion, d_k, d_v) for _ in range(numDecoders)])
        self.classifier = nn.Sequential(
            nn.Linear(d_m, vocabSize),
            nn.Softmax(dim=-1)
        )

            
    def forward(self, src, trg):
        # ========== ENCODING ==========
        # Create a padding mask BEFORE doing inputEmbedding (this is sample specific!)
        encPaddingMask = self.makePaddingMask(src, src).unsqueeze(1)     # (b, 1, m, m)
        # Also create padding for encoder-decoder-attention (sample specific)
        encDecPaddingMask = self.makePaddingMask(trg, src).unsqueeze(1)  # (b, 1, m, m)
        src = self.inputEmbedding(src)                         # (b, n) -> (b, n, d_m)
        for encoder in self.encoders:
            src = encoder(src, encPaddingMask)
        
        # ========== DECODING ==========
        # Create a look-ahead mask, so transformer can't peek at future output tokens!
        lookAheadMask = torch.tril(torch.ones(trg.shape[1], trg.shape[1])).to(self.device) # (m, m)
        # Feed src (encoding) into Decoders
        trg = self.outputEmbedding(trg)        # (b, m) -> (b, m, d_m)
        for decoder in self.decoders:
            trg = decoder(trg, src, lookAheadMask, encDecPaddingMask)
            
        # ======= CLASSIFICATION =======
        return self.classifier(trg)
    

    # PADDING: for practical reasons I define padding as 0 and valid data as 1, contrary
    #          to how this is done in most other implementations   
    @staticmethod
    def makePaddingMask(m, key_padding):
        '''
        This function creates a padding mask with key padding only
        Returns a mask of shape [b, m, n]
        QK|1__1__0__1
        1 |1  1  0  1
        0 |1  1  0  1
        1 |1  1  0  1
        m: sequence length of queries
        key_padding: padding info vector for the keys [b, n]. 1=valid data, 0=padding
        '''
        return key_padding.unsqueeze(1).expand(-1, m, -1)
    
    @staticmethod
    def makeLookAheadMask(m, device):
        '''
        Creates a square matrix of shape [m, m] with ones as lower trinagular matrix
        '''
        return torch.tril(torch.ones(m, m, device=device))  # [m, m]

    @staticmethod
    def combinePaddingMasks(mask1, mask2):
        '''
        combine two padding masks. Padding in one of them results in padding in the returned mask.
        Masks can lack the batch dimension because of broadcasting
        mask1, mask2: masks with shape [b, m, n] or [m, n]
        '''
        return mask1 * mask2