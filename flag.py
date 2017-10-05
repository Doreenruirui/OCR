import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 40, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
#tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_integer("max_seq_len", 100, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("out_dir", "/tmp", "Output directory")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("dev", "dev", "the prefix of development file")
tf.app.flags.DEFINE_integer("nthread", 8, "number of threads.")
tf.app.flags.DEFINE_string("lmfile1", None, "arpa file of the language model.")
tf.app.flags.DEFINE_string("lmfile2", None, "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0, "Language model relative weight.")
tf.app.flags.DEFINE_float("beta", 0, "Language model relative weight.")
tf.app.flags.DEFINE_float("gpu_frac", 0.3, "GPU Fraction to be used.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_integer("start", 0, "Decode from.")
tf.app.flags.DEFINE_integer("end", 0, "Decode to.")
tf.app.flags.DEFINE_float("variance", 3, "The context window size for decoding")
tf.app.flags.DEFINE_float("scalar", 3, "The scalar for sharpness of prior distribution")
tf.app.flags.DEFINE_float("weight", 0.8, "The weight for the original distribution.")

FLAGS = tf.app.flags.FLAGS
