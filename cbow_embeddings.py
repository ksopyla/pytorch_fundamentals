import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)


raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

data = []

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size=CONTEXT_SIZE):
        super(CBOW,self).__init__() 
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # embeddings，
        # projections        
        self.linear1 = nn.Linear(embedding_dim, 128)
        # output
        self.output = nn.Linear(128, vocab_size) 
 
    def forward(self, inputs):
        embeds =sum(self.embeddings(inputs)).view(1,-1)
        out = F.relu(self.linear1(embeds))
        out = self.output(out)
        
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example

losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, embedding_dim=20, context_size=CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(150):
    total_loss = torch.FloatTensor([0])

    for context, target in data:
        
        #prepare data
        context_idxs = make_context_vector(context, word_to_ix)
        target_idx = word_to_ix[target]        
        target_var = Variable(torch.LongTensor([target_idx]))
        
        # run forward
        model.zero_grad()
        log_probs = model(context_idxs)
 
        loss = loss_function(log_probs,target_var)
        #backprop
        loss.backward()
        #optimize step
        optimizer.step()
 
        total_loss += loss.item()

    losses.append(total_loss)


import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
