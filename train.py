import json
import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier 

## Parameters
learning_rate = 0.002

## Model
model = SimpleGemmaTransformerClassifier(device=torch.device('mps'))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

## read BTC-USD_news_with_price.json
with open('BTC-USD_news_with_price.json', 'r') as f:
    training = json.load(f)

## Define our labels
sell = [1., 0., 0.]
hold = [0., 1., 0.]
buy  = [0., 0., 1.]

## Generate our features and labels
features = []
labels = []

for item in training:
    features.append(
        "\n".join([f"Price: {item['price']}",
        f"Headline: {item['title']}",
        f"Summary: {item['summary']}"])
    )

    #"difference": -448.9140625,
    #"percentage": -0.414575337871938

    if item['percentage'] < -0.05:
        labels.append(sell)
    elif item['percentage'] > 0.05:
        labels.append(buy)
    else:
        labels.append(hold)

print(features[0:2])
print(labels[0:2])
## TODO
## TODO
## TODO merge features together in a single context
## TODO
## TODO




## TODO 
## TODO 
## TODO 
## TODO ⛔️ data shuffle with DataLoader and TensorDataset
## TODO  ✅ to_device() #<<- find devices automatically
## TODO 
## TODO 

"""
dataset = TensorDataset(
    features,
    torch.from_numpy(np.array(labels)).long(),
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
"""

batch = 4
epochs = 10
model.train()
for epoch in range(epochs):
    #for batch_features, batch_labels in dataloader:
    
    stochastic = np.random.permutation(len(features))
    inputs = np.array(features)[stochastic]
    targets = np.array(labels)[stochastic]

    for i in range(len(features) // batch):
        input  =  inputs[i * batch : i * batch + batch]
        target = torch.from_numpy(targets[i * batch : i * batch + batch])
        #target.float().to(torch.device('mps'))

        optimizer.zero_grad()
        logits = model(input)

        loss = criterion(logits, target.float().to(torch.device('mps')))
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).detach().cpu()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f} probs={probs}")


