import torch as torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, t_hidden_dim ,num_layers=1, t_num_layers=1, num_heads = 3, p = 0.1, encodeLabels = True):
        super(Model, self).__init__()

        # base requirements
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.p = p
        self.encodeLabels = encodeLabels

        # transformer requirements
        self.t_hidden_dim = t_hidden_dim # pytorch default 2048
        self.t_num_layers = t_num_layers
        self.num_heads = num_heads
        self.n_classes = 3
        
        # 1. encoder in a siamese fashion
        encoder = []
        
        for _ in range(self.num_layers):
            encoder.append(nn.Linear(self.input_dim, self.hidden_dim)) # default hiddendim == outputdim
            encoder.append(nn.SELU())
            encoder.append(nn.AlphaDropout(p=self.p))
            self.input_dim = self.hidden_dim # added

        # - linear layer
        encoder.append(nn.Linear(self.input_dim, self.output_dim)) 

        # - create encoder
        self.encoder = nn.Sequential(*encoder)

        # 2. layernorm
        self.ln =  nn.LayerNorm(self.output_dim, elementwise_affine=False)
    
        # 3. label encoder
        self.label_encoder = nn.Embedding(self.n_classes, self.output_dim)

        # 4. transformer encoder (take care in and outpout dim * 2 since label concat)
        if self.encodeLabels:
            self.d_model = self.output_dim * 2
        else:
            self.d_model = self.output_dim
       
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.num_heads, dim_feedforward=self.t_hidden_dim, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.t_num_layers)

        # disable nested tensor optimization for odd num_heads as it's not supported
        if self.num_heads % 2 != 0:
            encoder_layer.enable_nested_tensor = False

        # 5. mlp head 
        self.wideDim = self.d_model * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.wideDim),
            nn.ReLU(),
            nn.Linear(self.wideDim , 1)
            )

    def forward(self, query_mol, p_supp, n_supp, train=True):

        # Step 1: Encode query, negative and positive supportset: 
        p = self.ln(self.encoder(p_supp))
        n = self.ln(self.encoder(n_supp))
        q = self.ln(self.encoder(query_mol))

        if self.encodeLabels:
            # get params for label encoding
            device = p.device
            n_supps = p.shape[1]
            currentBatchSize = p.shape[0]
            
            # Step2: Encode labels of query,as well as negative and positive supportset
            pl = self.label_encoder( torch.ones(currentBatchSize, n_supps, dtype=torch.long).to(device) )
            nl = self.label_encoder (torch.zeros(currentBatchSize, n_supps, dtype=torch.long).to(device) )
            ql = self.label_encoder( torch.ones(currentBatchSize, dtype=torch.long).to(device) *2 )
                
            # Step 3: Concatenate input embeddings with label embeddings
            p = torch.cat((p, pl), dim=-1)
            n = torch.cat((n, nl), dim=-1)
            q = torch.cat((q, ql), dim=-1)

        # Step 4: Concatenate query with support
        q = q.unsqueeze(1)  # Add sequence dimension, Shape: (batch_size, 1, output_dim / 2*output_dim with label encoding)
        q_s = torch.cat((q, n, p), dim=1)  
        
        # Step 5: Pass through Transformer encoder
        out = self.transformer_encoder(q_s)
        
        # Step 6: Get logits via Sigmoid Function of the first dim of MLP head
        logits = torch.sigmoid(self.mlp(out[:, 0, :])).squeeze(1) # Shape: (batch_size) 
        return logits