import numpy as np

class CBOW:
    def __init__(self, dataset, vocab, window_size=2, embedding_dim=100, epochs=1000, learning_rate=0.01, neg_sample_size=15):
        self.dataset = dataset
        self.vocab = vocab
        self.V = vocab.size
        self.D = embedding_dim

        scale = np.sqrt(1.0 / embedding_dim)
        self.context_embedding = np.random.normal(0, scale, (self.V, self.D))
        self.samples_embedding = np.random.normal(0, scale, (self.V, self.D))

        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.neg_sample_size = neg_sample_size
    

    def _sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))


    def _calculate_loss(self, preds):
        preds = np.clip(preds, 1e-12, 1 - 1e-12)
        return -np.log(preds[0]) - np.sum(np.log(1 - preds[1:]))
    

    def fit(self):
        for epoch in range(self.epochs):
            epoch_losses = []
            current_lr = self.learning_rate * (1 - epoch / self.epochs)

            for text in self.dataset:
                unk_idx = self.vocab.word2idx['<UNK>']
                indices = [self.vocab.word2idx.get(w, unk_idx) for w in text]

                for target in range(len(text)):
                    target_idx = self.vocab.word2idx.get(text[target], unk_idx)
                    if target_idx == unk_idx: continue
                    
                    context_words_indices = indices[max(0, target-self.window_size) : target] + indices[target+1 : min(len(text), target+self.window_size+1)]
                    if len(context_words_indices) == 0: 
                        continue

                    context_vectors = self.context_embedding[context_words_indices]
                    v_context = np.mean(context_vectors, axis=0)

                    neg_samples = np.zeros(self.neg_sample_size, dtype=int)
                    
                    while target_idx in neg_samples:
                        mask = (neg_samples == target_idx)
                        new_samples = np.random.choice(self.V, size=np.sum(mask), p=self.vocab.probs)
                        neg_samples[mask] = new_samples

                    samples = [target_idx] + neg_samples
                    M_samples = self.samples_embedding[samples]

                    logits = M_samples @ v_context
                    preds = self._sigmoid(logits)
                    loss = self._calculate_loss(preds)
                    epoch_losses.append(loss)

                    labels = np.zeros(len(samples))
                    labels[0] = 1
                    
                    grad_context = (preds - labels) @ M_samples
                    grad_samples = np.outer(preds - labels, v_context)

                    self.context_embedding[context_words_indices] -= current_lr * grad_context / len(context_words_indices)
                    self.samples_embedding[samples] -= current_lr * grad_samples

            print(f"avg loss ({epoch}): {np.mean(epoch_losses)}")

    
    def evaluate(self, dataset):
        all_losses = []
        for text in dataset:
            unk_idx = self.vocab.word2idx['<UNK>']
            indices = [self.vocab.word2idx.get(w, unk_idx) for w in text]

            for target in range(len(text)):
                target_idx = self.vocab.word2idx.get(text[target], unk_idx)
                if target_idx == unk_idx: continue
                
                context_words_indices = indices[max(0, target-self.window_size) : target] + indices[target+1 : min(len(text), target+self.window_size+1)]
                if len(context_words_indices) == 0: 
                    continue

                context_vectors = self.context_embedding[context_words_indices]
                v_context = np.mean(context_vectors, axis=0)

                neg_samples = np.zeros(self.neg_sample_size, dtype=int)
                
                while target_idx in neg_samples:
                    mask = (neg_samples == target_idx)
                    new_samples = np.random.choice(self.V, size=np.sum(mask), p=self.vocab.probs)
                    neg_samples[mask] = new_samples

                samples = [target_idx] + neg_samples
                M_samples = self.samples_embedding[samples]

                logits = M_samples @ v_context
                preds = self._sigmoid(logits)
                loss = self._calculate_loss(preds)
                all_losses.append(loss)

        print(f"avg loss: {np.mean(all_losses)}")


                    

                    
                    

