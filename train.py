import json
import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
from sklearn.metrics import f1_score 

## Parameters
learning_rate = 0.005
batch = 1
epochs = 20

## Define our labels
sell = [1., 0., 0.]
hold = [0., 1., 0.]
buy  = [0., 0., 1.]

## Model
model = SimpleGemmaTransformerClassifier(device=torch.device('mps'))
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

## read BTC-USD_news_with_price.json
with open('BTC-USD_news_with_price.json', 'r') as f:
    training = json.load(f)


## Generate our features and labels
features = []
labels = []

for item in training:
    features.append(
        "\n".join([f"Price: {item['price']}",
        f"Headline: {item['title']}",
        f"Summary: {item['summary']}"])
    )

    ## Generate labels based on percentage change
    ## Actions Buy / Sell / Hold
    if item['percentage'] < -0.01:
        labels.append(sell)
    elif item['percentage'] > 0.01:
        labels.append(buy)
    else:
        labels.append(hold)

## Separate Train and Test sets
split_index = int(0.8 * len(features))
train_features = features[:split_index]
train_labels = labels[:split_index]
test_features = features[split_index:]
test_labels = labels[split_index:]

item_losses = []
model.train()
for epoch in range(epochs):
    stochastic = np.random.permutation(len(train_features))
    inputs = np.array(train_features)[stochastic]
    targets = np.array(train_labels)[stochastic]

    for i in range(len(inputs) // batch):
        input  =  inputs[i * batch : i * batch + batch]
        target = torch.from_numpy(targets[i * batch : i * batch + batch])

        optimizer.zero_grad()
        logits = model(input)

        loss = criterion(
            logits,
            target.float().to(torch.device('mps'))
        )
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).detach().cpu()
        item_losses.append(loss.item())
        cost = sum(item_losses[-250:]) / len(item_losses[-250:])
        print(f"Epoch {epoch + 1}: loss={cost:.4f} probs={probs}")

## Evaluate Model Accuracy
correct = 0
total = 0
all_predictions = []
all_actuals = []

model.eval()
with torch.no_grad():
    for i in range(len(test_features)):
        input  = [test_features[i]]
        target = torch.tensor(test_labels[i])

        logits = model(input)
        probs = logits.softmax(dim=-1).cpu()
        predicted = torch.argmax(probs, dim=-1)
        actual = torch.argmax(target.float().to(torch.device('mps')))

        all_predictions.append(predicted.item())
        all_actuals.append(actual.item())

        if predicted.item() == actual.item():
            correct += 1
        total += 1

# Calculate Accuracy
print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")

# Calculate F1 Score
f1 = f1_score(all_actuals, all_predictions, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")
print("Done")

