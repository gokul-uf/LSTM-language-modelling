class Config:
    batch_size = 64
    embed_size = 100
    vocab_size = 20000
    num_hidden_state = 1024 # change to 1024 for task 1c
    proj_hidden_state = 512 # only for task 1c
    seq_length = 30
    completed_sentence_length = 20
    top_words = 20000 - 4
    lr = 0.001
    num_epochs = 20
    pkl_dir = "pkls"
    ckpt_dir = "../1a/checkpoints/"
    mode = "CONTINUATION" # could be "TRAIN" or "TEST" or "CONTINUATION"
    ckpt_file = ""
    word2vec_path = "../wordembeddings-dim100.word2vec"
    test_file = "../data/sentences.eval"
    continuation_file = "../data/sentences.eval"
