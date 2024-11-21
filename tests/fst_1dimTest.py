# One dimensional Example of the fst_architecture

import torch as torch
import torch.nn as nn

# 1. mol encoder

encoder = []

batch_size = 4
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
    nn.Linear(input_dim, 1)
    )

# 1.1 print dims

for i in ['batch_size','input_dim','hidden_dim','output_dim','n_classes']:
    print(f"{i} = {locals()[i]}")

# 2. Creating dummy data

query_mol = torch.rand((batch_size, 10))
query_label = torch.ones(batch_size,  dtype=torch.long) * 2 # 2 is unknown

p_mols = torch.rand((batch_size, 3, 10))
n_mols = torch.rand((batch_size, 3, 10)) 

p_labels = torch.ones(batch_size, 3, dtype=torch.long)
n_labels = torch.zeros(batch_size, 3, dtype=torch.long)

# old
#p_labels = torch.tensor([[1, 1, 1]] * batch_size) # Shape: (batch_size, n_supps)
#n_labels = torch.tensor([[0, 0, 0]] * batch_size) # Shape: (batch_size, n_supps)

# 2.1 print dummy data
for i in ['query_mol', 'query_label', 'p_mols', 'p_labels','n_mols', 'n_labels']:
    print(f"{i} = {locals()[i].shape}\n")

# 3. Encode dummy data
q_embed = mol_encoder(query_mol)  # Shape: (batch_size, output_dim)
p_embed = mol_encoder(p_mols) # Shape: (batch_size, 3, output_dim)
n_embed = mol_encoder(p_mols) # Shape: (batch_size, 3, output_dim)

ql_embed = label_encoder(query_label)  # Shape: (batch_size, output_dim)
pl_embed = label_encoder(p_labels)  # Shape: (batch_size, n_supp=3, output_dim)
nl_embed = label_encoder(n_labels)  # Shape: (batch_size, n_supp=3, output_dim)

# 3.1 print encoded dummy data
for i in ['q_embed', 'p_embed','n_embed','ql_embed', 'pl_embed','nl_embed']:
    print(f"{i} = {locals()[i].shape}\n")

# 4. Concatenate molecules with labels
q = torch.cat((q_embed, ql_embed), dim=-1)  # Shape: (batch_size, 2*output_dim)
p = torch.cat((p_embed, pl_embed), dim=-1)  # Shape: (batch_size, n_supp=3 , 2*output_dim)
n = torch.cat((n_embed, nl_embed), dim=-1)  # Shape: (batch_size, n_supp=3 , 2*output_dim)

# 4.1 Print concatenated embeds
for i in ['q', 'p','n']:
    print(f"{i} = {locals()[i].shape}")

# 5. Concatenate query with support
q = q.unsqueeze(1)  # Add sequence dimension, Shape: (batch_size, 1, 2*output_dim)
q_s = torch.cat((q, n, p), dim=1)   # Shape: (batch_size, 4, 2*output_dim)

# 6. Encode encoded dummy data with transformer
out = transformer_encoder(q_s)  # Shape: (batch_size, 4, 2*output_dim)

# 7. Sigmoid of mlp to get logits
logits = torch.sigmoid(mlp(out[:, 0, :]))  # Shape: (batch_size, 2)

# 8.Get predicted class
pred = torch.argmax(logits, dim=1)  # Shape: (batch_size,)

# print
print(f"mlp_out={mlp(out[:, 0, :])}\n\n{logits=}\n\n{pred=}\n")