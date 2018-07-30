import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
char_to_ix = {}

for sent,_ in training_data:
    for w in sent:
        
        for char in w:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
    

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
CHAR_EMBEDDING = 3
CHAR_LEVEL_REPRESENTATION_DIM = 3

def prepare_both_sequences(sentence, word_to_ix, char_to_ix):
    chars = [prepare_sequence(w, char_to_ix) for w in sentence]
    return prepare_sequence(sentence, word_to_ix), chars

class LSTMCharTagger(nn.Module):
    '''
    Augmented model, takes both sequence of words and char to predict tag.
    Characters are embedded and then get their own representation for each WORD.
    It is this representation that is merged with word embeddings and then fed to the sequence
    LSTM which decodes the tags.
    '''
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim,
                 hidden_char_dim, vocab_size, charset_size, tagset_size):
        super(LSTMCharTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_char_dim = hidden_char_dim

        # Word embedding:
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)

        # Char embedding and encoding into char-lvl representation of words (c_w):
        self.char_embedding = nn.Embedding(charset_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, hidden_char_dim)

        # Sequence model:
        self.lstm = nn.LSTM(word_embedding_dim + hidden_char_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Init hidden state for lstms
        self.hidden = self.init_hidden(self.hidden_dim)
        self.hidden_char = self.init_hidden(self.hidden_char_dim)

    def init_hidden(self, size, batch_size=1):
        "Batch size argument used when PackedSequence are used"
        return (autograd.Variable(torch.zeros(1, batch_size, size)),
                autograd.Variable(torch.zeros(1, batch_size, size)))

    def forward_one_word(self, word_sequence, char_sequence):
        ''' For a word by word processing.
        '''
        # Word Embedding
        word_embeds = self.word_embedding(word_sequence)
        # Char lvl representation of each words with 1st LSTM
        char_embeds = self.char_embedding(char_sequence)
        char_lvl, self.hidden_char = self.char_lstm(char_embeds.view(len(char_sequence),1,-1), self.hidden_char)
        # Merge
        merged = torch.cat([word_embeds.view(1,1,-1), char_lvl[-1].view(1,1,-1)], dim=2)
        # Predict tag with 2nd LSTM:
        lstm_out, self.hidden = self.lstm(merged, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def forward(self, word_sequence, char_sequence):
        ''' Importantly, char_sequence is a list of tensors, one per word, and one tensor 
        must represent a whole sequence of character for a given word.
        E.g.: is word_sequence has length 4, char_seq must be of length 4, thus char_lstm
        will output 4 char-level word representations (c_w).

        Here we deal with variable lengths of character tensors sequence using nn.utils.rnn.pack_sequence
        '''
        # Word Embedding
        word_embeds = self.word_embedding(word_sequence)

        # Char lvl representation of each words with 1st LSTM
        # We will pack variable length embeddings in PackedSequence. Must sort by decreasing length first.
        sorted_length = np.argsort([char_sequence[k].size()[0] for k in range(len(char_sequence))])
        sorted_length = sorted_length[::-1] # decreasing order
        char_embeds = [self.char_embedding(char_sequence[k]) for k in sorted_length]
        packed = nn.utils.rnn.pack_sequence(char_embeds) # pack variable length sequence
        out, self.hidden_char = self.char_lstm(packed, self.hidden_char)
        encodings_unpacked, seqlengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # unpack and pad
        # We need to take only last element in sequence of lstm char output for each word:
        unsort_list = np.argsort(sorted_length) # indices to put list of encodings in orginal word order
        char_lvl = torch.stack([encodings_unpacked[k][seqlengths[k]-1] for k in unsort_list])

        # Merge
        merged = torch.cat([word_embeds, char_lvl], dim=1) # gives tensor of size (#words, #concatenated features)

        # Predict tag with 2nd LSTM:
        lstm_out, self.hidden = self.lstm(merged.view(len(word_sequence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(word_sequence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def get_batch_size(seq2pack):
    "Need this to correctly initialize batch lstm hidden states when packing variable length sequences..."
    sorted_length = np.argsort([seq2pack[k].size()[0] for k in range(len(seq2pack))])
    sorted_length = sorted_length[::-1] # decreasing order
    packed = nn.utils.rnn.pack_sequence([seq2pack[k] for k in sorted_length]) 
    return max(packed.batch_sizes)

model = LSTMCharTagger(EMBEDDING_DIM, CHAR_EMBEDDING, HIDDEN_DIM, CHAR_LEVEL_REPRESENTATION_DIM,
                       len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
words_in, chars_in = prepare_both_sequences(training_data[0][0], word_to_ix, char_to_ix)
model.hidden_char = model.init_hidden(model.hidden_char_dim, batch_size=get_batch_size(chars_in))
tag_score = model(words_in, chars_in)
print(tag_score)

t0 = time()
for epoch in range(300): 
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        model.zero_grad()

        # Step 2. Get our inputs ready
        sentence_in, chars_in = prepare_both_sequences(sentence, word_to_ix, char_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        model.hidden = model.init_hidden(model.hidden_dim)
        model.hidden_char = model.init_hidden(model.hidden_char_dim, batch_size=get_batch_size(chars_in))

        # Step 3. Run our forward pass.
        tag_score = model(sentence_in, chars_in)

        # Step 4. Compute the loss, gradients, and update the parameters
        loss = loss_function(tag_score, targets)
        loss.backward()
        optimizer.step()
print("300 epochs in %.2f sec for model with packed sequences"%(time()-t0))

model = LSTMCharTagger(EMBEDDING_DIM, CHAR_EMBEDDING, HIDDEN_DIM, CHAR_LEVEL_REPRESENTATION_DIM,
                       len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

t0 = time()
for epoch in range(300):
    for sentence, tags in training_data:
        sentence_score = []
        # Step 1. Remember that Pytorch accumulates gradients.
        model.zero_grad()

        # Step 2. Get our inputs ready
        sentence_in, chars_in = prepare_both_sequences(sentence, word_to_ix, char_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        model.hidden = model.init_hidden(model.hidden_dim)
        #model.hidden_char = model.init_hidden(model.hidden_char_dim)

        # Step 3. Run our forward pass on each word
        for k in range(len(sentence)):
            # Clear hidden state between EACH word (char level representation must be independent of previous word)
            model.hidden_char = model.init_hidden(model.hidden_char_dim)
            tag_score = model.forward_one_word(sentence_in[k], chars_in[k])
            sentence_score.append(tag_score)
            loss = loss_function(tag_score, targets[k].view(1,))
            loss.backward(retain_graph=True) # accumulate gradients now
            #tag_score = autograd.Variable(torch.cat(sentence_score), requires_grad=True)

        # Step 4. Update parameters at the end of sentence
        optimizer.step()
print("300 epochs in %.2f sec for model at word level"%(time()-t0))

# See what the scores are after training
words_in, chars_in = prepare_both_sequences(training_data[0][0], word_to_ix, char_to_ix)
model.hidden_char = model.init_hidden(model.hidden_char_dim, batch_size=get_batch_size(chars_in))
tag_score = model(words_in, chars_in)
print(tag_score)