from config import Config as conf
from tqdm import tqdm

class preprocessor:
	tokens = {}
	top_20k = []
	word2idx = {}
	idx2word = {}
	lines = []

	def preprocess(self, filename):
		print "reading file"
		self.extract_tokens(filename)
		print "extracting top 20k words"
		self.get_top20k()
		print "creating word to ID mapping"
		self.create_mapping()

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
					self.lines.append(line)
					for word in words:
						if word in self.tokens:
							self.tokens[word] += 1
						else:
							self.tokens[word] = 1
			assert len(self.lines) + overflow_lines == total_lines
			print "total lines: {}".format(total_lines)
			print "overflow lines ( > 28 words): {}".format(overflow_lines)

	def get_top20k(self):
		top_words = sorted(self.tokens, key = self.tokens.get, reverse = True)[:conf.top_words]
		top_words.extend(["<bos>", "<eos>", "<pad>", "<unk>"])
		assert len(top_words) == 20000
		self.top_20k = top_words

	def create_mapping(self):
		for idx, word in enumerate(self.top_20k):
			self.word2idx[word] = idx
			self.idx2word[idx] = word