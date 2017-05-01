from config import Config as conf
from tqdm import tqdm
import numpy as np
import sys
if sys.version.split()[0].startsiwth("2"): #Python2
	import cPickle as pkl
else:
	import pickle as pkl

class preprocessor:
	tokens = {}
	top_20k = []
	word2idx = {}
	idx2word = {}
	lines = []
	preprocessed_data = []
	loaded = False

	def __init__(self, dir_name = None):
		if dir_name == None:
			self.tokens = {}
			self.top_20k = []
			self.word2idx = {}
			self.idx2word = {}
			self.lines = []
			self.preprocessed_data = []
			self.loaded = False
		else:
			self.tokens = {}
			self.top_20k =  pkl.load(open(dir_name+"/top_20k.pkl"))
			self.word2idx = pkl.load(open(dir_name+"/word2idx.pkl"))
			self.idx2word = pkl.load(open(dir_name+"/idx2word.pkl"))
			self.lines =    pkl.load(open(dir_name+"/lines.pkl"))
			self.preprocessed_data = lines
			self.loaded = True

	def preprocess(self, filename):
		if not self.loaded:
			print("Reading file")
			self.extract_tokens(filename)
			print("Extracting top 20k words")
			self.get_top20k()
			del self.tokens
			print("Creating word to ID mapping")
			self.create_mapping()
			print("Dumping the pkl files")
			print("Dumping top_20k")
			pkl.dump(self.top_20k, open("top_20k.pkl", "wb"))
			print("Dumping word2idx")
			pkl.dump(self.word2idx, open("word2idx.pkl", "wb"))
			print("Dumping idx2word")
			pkl.dump(self.idx2word, open("idx2word.pkl", "wb"))
			# print "Dumping lines"
			# pkl.dump(self.lines, open("lines.pkl", "w"))
		else:
			print("All loaded, nothing to do!")

	def extract_tokens(self, filename):
		self.tokens = {}
		self.lines = []
		total_lines = 0
		overflow_lines = 0
		with open(filename) as f:
			for line in tqdm(f, total = 2000000):
				total_lines+=1
				words = line.split(" ")
				if len(words) > 28:
					overflow_lines+=1
					continue
				else:
					self.lines.append(words)
					for word in words:
						if word in self.tokens:
							self.tokens[word] += 1
						else:
							self.tokens[word] = 1
			assert len(self.lines) + overflow_lines == total_lines
			print("total lines: {}".format(total_lines))
			print("overflow lines ( > 28 words): {}".format(overflow_lines))

	def get_top20k(self):
		top_words = sorted(self.tokens, key = self.tokens.get, reverse = True)[:conf.top_words]
		top_words.extend(["<bos>", "<eos>", "<pad>", "<unk>"])
		assert len(top_words) == 20000
		self.top_20k = top_words

	def create_mapping(self):
		for idx, word in enumerate(self.top_20k):
			self.word2idx[word] = idx
			self.idx2word[idx] = word

	def lines2idx(self):
		for i in tqdm(xrange(len(self.lines))):
			line = self.lines[i]
			if len(line) < 28:
				line = line + ["<pad>"]*(28 - len(line))
			line = ["<bos>"] + line + ["<eos>"]
			assert len(line) == 30
			line = [[self.word2idx.get(word, self.word2idx["<unk>"])] for word in line]
			self.lines[i] = line
		self.preprocessed_data = self.lines

	def get_batch(self):
		np.random.shuffle(self.lines)
		for i in range(0, 64 * (len(self.lines) // 64), conf.batch_size):
			new_batch = []
			batch = self.lines[i: i+64]
			for line in batch:
				if len(line) < 28:
					line = line + ["<pad>"]*(28 - len(line))
				line = ["<bos>"] + line + ["<eos>"]
				assert len(line) == 30
				line = [[self.word2idx.get(word, self.word2idx["<unk>"])] for word in line]
				new_batch.append(line)
			batch = new_batch
			batch = np.asarray(batch)
			if batch.shape != (64, 30, 1):
				print(len(self.lines))
			assert batch.shape == (64, 30, 1)
			yield batch[:, :-1,:], batch[:, 1:, :]