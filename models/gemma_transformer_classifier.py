"""Minimal Gemma + PyTorch Transformer text classifier for 3-way decisions."""

from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer


class SimpleGemmaTransformerClassifier(nn.Module):
    """Wraps Google Gemma embeddings with a tiny Transformer encoder head."""

    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_embedding: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", device=str(self.device))
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.project = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.freeze_embedding = freeze_embedding
        self.to(self.device)

    def forward(self, texts: Iterable[str]) -> Tensor:
        features = self.embedding_model.tokenize(list(texts))
        features = {name: tensor.to(self.device) for name, tensor in features.items()}

        if self.freeze_embedding:
            with torch.no_grad():
                outputs = self.embedding_model(features)
        else:
            outputs = self.embedding_model(features)

        token_embeddings = outputs["token_embeddings"]
        attention_mask = features["attention_mask"]

        hidden = self.project(token_embeddings)

        key_padding_mask = attention_mask == 0
        encoded = self.transformer(hidden, src_key_padding_mask=key_padding_mask)

        mask = attention_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


def example_forward() -> None:
    model = SimpleGemmaTransformerClassifier()
    texts = ["Buy the dip?", "Market is flat."]
    logits = model(texts)
    print("Logits:", logits.detach().cpu())
    print("Probabilities:", logits.softmax(dim=-1).detach().cpu())


def example_train(epochs: int = 2, lr: float = 1e-4) -> None:
    """
    Tiny illustrative training loop. Replace ``texts`` and ``labels`` with
    real data when integrating into your pipeline.
    """
    model = SimpleGemmaTransformerClassifier(freeze_embedding=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    texts = [
        "Strong earnings beat expectations.",
        "Regulatory concerns weigh on the sector.",
        "Momentum indicators point to consolidation.",
    ]
    labels = torch.tensor([2, 0, 1], device=model.device)  # class ids: buy, sell, hold

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).detach().cpu()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f} probs={probs}")


if __name__ == "__main__":
    example_forward()
