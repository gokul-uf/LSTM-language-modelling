### nlu_project
Code repo for projects of the natural language understanding course

# data

Data is kept in the data folder (not part of this repo) as follows:

data/sentences.continuation
data/sentences_test
data/sentences.train


# Evaluation

The perplexity is calculated based on the formula on the lower half of
page 2 on the assignment sheet on test sentences parsed as
               <bos> w_2 w_3 ... w_{n-1} <eos>
with the '<bos>' and '<eos>' symbols being part of the sentence, which
consists of n symbols. The model is evaluated after 10 epochs of training
in each case. The submitted result files are obtained from the command line
output (snipping of a few lines at the beginning and at the end), while for
task 1.2 the output is produced in a file. Any adaptations to fit input/output
sentence numbers to a multiple of the batch_size (== 64) is handled directly
in the implementation with the exception of perplexity evaluation, which is
performed in a manual postprocessing step.