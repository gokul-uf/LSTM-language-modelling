### Task 1.2: Conditional generation

To complete sentences in ../data/sentences.continuation run `python3 12.py`

The implementation uses the model of task 1.1 c) and applies the same parsing strategy for sentences as in the
evaluation part (prepending '<bos>', appending '<eos>' and padding with '<pad>'). For every batch, we then use
a loop to complete sentences one word per iteration (evaluating soft-max logits for the entire batch, inserting the
logit-maximizing word in the position past the last supplied/predicted word in each sentence by updating the batch
data set accordingly) until either '<eos>' is predicted or the sentence has 18 symbols, which together with '<bos>'
and '<eos>' adds up to the maximum sentence length of 20. Note that we include '<unk>' in our continuations, whereas we do not accept
'<bos>', '<pad>' or multiple occurrences of '<eos>'.