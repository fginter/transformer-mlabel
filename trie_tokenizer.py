import pygtrie
import sys
import tqdm

class TrieTokenizer:

    def __init__(self,vocab_file,unk="[UNK]"):
        with open(vocab_file,"rt") as f:
            vocab_items=[line.rstrip("\n") for line in f if line.rstrip("\n")]
            self.trie=pygtrie.Trie.fromkeys(vocab_items)
            self.unk=unk

    def tokenize(self,s):
        tokenized=[]
        for w in s.split():
            tokenized.extend(self.tokenize_word(w))
        return tokenized
        
    def tokenize_word(self,w):
        tokenized=[]
        idx=0
        while idx<len(w):
            if idx==0:
                prfx,_=self.trie.longest_prefix(w[idx:])
            else:
                prfx,_=self.trie.longest_prefix("##"+w[idx:])
            if not prfx: #no match, eat a character, try again
                tokenized.append(self.unk)
                idx+=1
            else:
                tokenized.append("".join(prfx))
                if idx==0:
                    idx+=len(prfx)
                else:
                    idx+=len(prfx)-2 #accounts for ##
        return tokenized
