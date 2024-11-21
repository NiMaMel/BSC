# One dimensional Example of the fst_architecture

import torch as torch
import torch.nn as nn


# mol encoder

encoder = []

input_dim = 10
hidden_dim = 4
output_dim = 4
n_classes = 3

for _ in range(2):
    encoder.append(nn.Linear(input_dim, hidden_dim)) # default hiddendim == outputdim
    encoder.append(nn.SELU())
    encoder.append(nn.AlphaDropout(p=0.2))
    input_dim = hidden_dim 

# Final linear layer
encoder.append(nn.Linear(input_dim, output_dim)) 

# Create encoder
mol_encoder = nn.Sequential(*encoder)


# label encoder
label_encoder = nn.Embedding(n_classes, output_dim)

# transformer encoder
d_model = 2*input_dim

encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead = 4, dim_feedforward=d_model, batch_first = True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

# mlp head -> logits ( probabilities of bein active or inactive)
mlp = nn.Sequential(
    nn.Linear(d_model, input_dim),
    nn.ReLU(),
    nn.Linear(input_dim, 2)
    )

# creating dummy data
query_mol = torch.rand((10,))
query_label = torch.tensor([2]) # 2 is unknown

supp_mols = torch.rand((3, 10)) 
supp_labels = torch.tensor([1, 1, 0])


# encode dummy data
q_embed = mol_encoder(query_mol)
supp_embed = mol_encoder(supp_mols)

ql_embed = label_encoder(query_label)
sl_embed = label_encoder(supp_labels)

# concat molecules with labels
q = torch.cat((q_embed, ql_embed.squeeze(0)), dim=-1)
s = torch.cat((supp_embed, sl_embed), dim=-1)

# concat query with support
q = q.unsqueeze(0)  # Add batch dimension
q_s = torch.cat((q, s), dim=0)

# encode encoded dummy data with transformer
out = transformer_encoder(q_s)
out

# sigmoid of mlp to get logtis
logits = torch.sigmoid(mlp(out[0]))
logits

# get predicted class
pred = torch.argmax(logits)
print(pred)