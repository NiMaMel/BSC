import torch as torch
import torch.nn as nn

class fstModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, t_hidden_dim, batch_size ,num_layers=1, t_num_layers=1, num_heads = 3, p = 0.1):
        super(fstModel, self).__init__()

        # base requirements
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.p = p

        # transformer requirements
        self.t_hidden_dim = t_hidden_dim
        self.t_num_layers = t_num_layers
        self.num_heads = num_heads
        self.n_classes = 3
        
        # 1. encoder in a siamese fashion
        encoder = []
        
        for _ in range(self.num_layers):
            encoder.append(nn.Linear(self.input_dim, self.hidden_dim)) # default hiddendim == outputdim
            encoder.append(nn.SELU())
            encoder.append(nn.AlphaDropout(p=self.p))
            input_dim = hidden_dim # added

        # - linear layer
        encoder.append(nn.Linear(self.input_dim, self.output_dim)) 

        # - create encoder
        self.encoder = nn.Sequential(*encoder)

        # 2. layernorm
        self.ln =  nn.LayerNorm(self.output_dim, elementwise_affine=False)
    
        # 3. label encoder
        self.label_encoder = nn.Embedding(self.n_classes, self.output_dim)

        # 4. transformer encoder (take care in and outpout dim * 2 since label concat)
        self.d_model = self.output_dim * 2
       
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.num_heads, dim_feedforward=self.t_hidden_dim, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.t_num_layers)

        # 5. mlp head 
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.n_classes)
            )

    def forward(self, query_mol, p_supp, n_supp,train=True):

        # Step 1: Encode query, negative and positive supportset: 
        p = self.ln(self.encoder(p_supp))
        n = self.ln(self.encoder(n_supp))
        q = self.ln(self.encoder(query_mol))

        # Step2: Encode labels of query,as well as negative and positive supportset
        pl = self.label_encoder(torch.tensor([2] * self.batch_size))
        nl = self.label_encoder(torch.tensor([0] * self.batch_size))
        ql = self.label_encoder(torch.tensor([1] * self.batch_size))
        
        # Step 3: Concatenate input embeddings with label embeddings
        p = torch.cat((p, pl), dim=-1)
        n = torch.cat((n, nl), dim=-1)
        q = torch.cat((q, ql), dim=-1)

        # Step 4: Concatenate query with support
        q = q.unsqueeze(1)  # Add sequence dimension, Shape: (batch_size, 1, 2*output_dim)
        q_s = torch.cat((q, n, p), dim=1)  
        
        # Step 5: Pass through Transformer encoder
        out = self.transformer_encoder(q_s)

        # Step 6: Get logits via Sigmoid Function of the first dim of MLP head
        logits = torch.sigmoid(self.mlp(out[:, 0, :])) 
        
        # Step 7: Get Prediction using  argmax 
        prediction = torch.argmax(logits, dim=1)

        return prediction 