import json
import torch
import sys
import transformers
import gzip
import tqdm
import trie_tokenizer
import random

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["pbert-v1"]="proteiinipertti-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["pbert-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["pbert-v1"]={'do_lower_case': False}


def prep_ctrl_data(inp,bert_tokenizer,bert_tokenizer_trie,class_labels_file):
    idx2label=json.load(open(class_labels_file)) #this is a list
    label2idx=dict((c,i) for (i,c) in enumerate(idx2label))

    input_sequences=[]
    classidx_sequences=[]
    for line_idx,line in tqdm.tqdm(enumerate(inp)):
        line=line.rstrip("\n")
        if not line:
            continue
        cols=line.split("\t")
        if len(cols)==2:
            #looks like training data with classes
            assert cols[0].startswith("GO"), cols
            seq=cols[1]
            cls=cols[0]
        elif len(cols)==1:
            #looks like prediction data without classes
            assert not cols[0].startswith("GO"), cols
            seq=cols[0]
            cls=None
        else:
            assert False
        seq_tok=bert_tokenizer_trie.tokenize(seq)
        if seq_tok[-1]=="[UNK]":
            print(len(seq.split()[-1]),file=sys.stderr)
            print(seq_tok,file=sys.stderr)
            print(file=sys.stderr)
        #print(seq_tok,file=sys.stderr)
        input_sequences.append(torch.tensor(bert_tokenizer.convert_tokens_to_ids(seq_tok),dtype=torch.int32))
        if cls is not None:
            labels=cls.split(",")
            if len(labels)>1000:
                print("{} labels at line {}".format(len(labels),line_idx+1),file=sys.stderr)
            classidx_sequences.append(torch.tensor([label2idx[l] for l in labels],dtype=torch.int32))
        else:
            classidx_sequences.append(torch.tensor([],dtype=torch.int32))
    return input_sequences, classidx_sequences

def yield_batched(prepped_data_file,batchsize=10000,shuffle=True,max_epochs=1):
    input_sequences, classidx_sequences=torch.load(prepped_data_file)
    for epoch in range(max_epochs):
        if shuffle:
            idx_seq=list(range(len(input_sequences)))
            random.shuffle(idx_seq)
            input_sequences=[input_sequences[idx] for idx in idx_seq]
            classidx_sequences=[classidx_sequences[idx] for idx in idx_seq]
        assert len(input_sequences)==len(classidx_sequences)
        batch_inp=[]
        batch_classidx=[]
        inp_lengths=[]
        classidx_lengths=[]
        for inp_seq, classidx_seq in zip(input_sequences,classidx_sequences):
            batch_inp.append(inp_seq)
            batch_classidx.append(classidx_seq)
            inp_lengths.append(inp_seq.shape[0])
            classidx_lengths.append(classidx_seq.shape[0])
            #How much of data we've gathered for this batch?
            inp_size=max(inp_lengths)*len(inp_lengths) #this is how big the input matrix will be after padding
            classidx_size=max(classidx_lengths)*len(classidx_lengths)
            if inp_size+classidx_size > batchsize:
                #time to yield!
                batch_inputs_padded=torch.nn.utils.rnn.pad_sequence(batch_inp,batch_first=True)
                batch_classidx_padded=torch.nn.utils.rnn.pad_sequence(batch_classidx,batch_first=True)
                yield batch_inputs_padded, batch_classidx_padded
                batch_inp=[]
                batch_classidx=[]
                inp_lengths=[]
                classidx_lengths=[]
        else:
            if inp_lengths:
                batch_inputs_padded=torch.nn.utils.rnn.pad_sequence(batch_inp,batch_first=True)
                batch_classidx_padded=torch.nn.utils.rnn.pad_sequence(batch_classidx,batch_first=True)
                batch_inputs_padded=torch.nn.utils.rnn.pad_sequence(batch_inp,batch_first=True)
                batch_classidx_padded=torch.nn.utils.rnn.pad_sequence(batch_classidx,batch_first=True)
                yield batch_inputs_padded, batch_classidx_padded


if __name__=="__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-labels-file",default="labels-ctrl.json",help="Class labels file. Default %(default)s")
    parser.add_argument("--bert-tokenizer",default="pbert-v1",help="Which BERT model tokenizer to load? Default %(default)s")
    parser.add_argument("--prep-data-in",nargs="+",help="List of .txt.gz files with the data in tab-sep format. Will be pre-tokenized and saved into torch bin format.")
    parser.add_argument("--prep-data-out",help="Directory to which the prepped data will be chached, use with --prep-data-in")
    args=parser.parse_args()

    # for x,y in tqdm.tqdm(yield_batched(args.prep_data_out,10000,max_epochs=100)):
    #     pass
    # sys.exit()
    
    if args.prep_data_out:
        bert_tokenizer= transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)
        bert_tokenizer_trie=trie_tokenizer.TrieTokenizer(transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"][args.bert_tokenizer])
        os.makedirs(args.prep_data_out,exist_ok=True)
        for fname in args.prep_data_in:
            print("********** Preprocessing",fname,file=sys.stderr)
            with gzip.open(fname,"rt",encoding="ASCII") as f:
                inputs_padded, classidx_padded=prep_ctrl_data(f,bert_tokenizer,bert_tokenizer_trie,args.class_labels_file)
                basename=os.path.basename(fname).replace(".txt.gz",".torchbin")
                torch.save((inputs_padded, classidx_padded), os.path.join(args.prep_data_out,basename))
            

    

