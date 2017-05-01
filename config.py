class Config:
    batch_size = 64
    embed_size = 100
    vocab_size = 20000
    num_hidden_state = 512
    seq_length = 30
    top_words = 20000 - 4
    lr = 0.001
    num_epochs = 20
    pkl_dir = "pkls"
    ckpt_dir = "checkpoints"
    mode = "TRAIN" # could be "TRAIN" or "TEST"
    ckpt_file = ""
    test_flie = "data/sentences.eval"
