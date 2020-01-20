import json
import torch
import sys
import transformers
import gzip
import tqdm
import trie_tokenizer
import random
import collections
import alias_multinomial
import random

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["pbert-v1"]="proteiinipertti-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["pbert-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["pbert-v1"]={'do_lower_case': False}

def prep_class_stats(class_stats_filename,max_labels):
    class_stats=json.load(open(class_stats_filename)) #this is a dict: label -> count
    class_stats=collections.Counter(class_stats)
    
    idx2label=sorted(class_stats.keys())
    label2idx=dict((c,i) for (i,c) in enumerate(idx2label))
    
    return idx2label,label2idx,class_stats

def prep_ctrl_data(inp,bert_tokenizer,bert_tokenizer_trie,class_stats_file,max_labels):

    idx2label,label2idx,class_stats=prep_class_stats(class_stats_file,max_labels)
    filtered_labels=set(k for k,v in class_stats.most_common(max_labels))
    
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
        seq_tok=["[CLS]"]+bert_tokenizer_trie.tokenize(seq)+["[SEP]"]
        if seq_tok[-2]=="[UNK]":
            print(len(seq.split()[-1]),file=sys.stderr)
            print(seq_tok,file=sys.stderr)
            print(file=sys.stderr)
        #print(seq_tok,file=sys.stderr)
        if cls is not None:
            labels=list(lab for lab in cls.split(",") if lab in filtered_labels)
            if len(labels)>1000:
                print("{} labels at line {}".format(len(labels),line_idx+1),file=sys.stderr)
            if len(labels)==0: #no label survived the filter!
                continue
            labels_int=[label2idx[l] for l in labels]
            classidx_sequences.append(torch.tensor(labels_int,dtype=torch.int32))
        else:
            classidx_sequences.append(torch.tensor([],dtype=torch.int32))
        input_sequences.append(torch.tensor(bert_tokenizer.convert_tokens_to_ids(seq_tok),dtype=torch.int32))
    return input_sequences, classidx_sequences

def yield_batched(prepped_data_file,batchsize=10000,shuffle=True,max_epochs=1,alias=None):
    with open(prepped_data_file,"rb") as f:
        input_sequences, classidx_sequences=torch.load(f)
    for epoch in range(max_epochs):
        if shuffle:
            idx_seq=list(range(len(input_sequences)))
            random.shuffle(idx_seq)
            input_sequences=[input_sequences[idx] for idx in idx_seq]
            classidx_sequences=[classidx_sequences[idx] for idx in idx_seq]
        assert len(input_sequences)==len(classidx_sequences)
        batch_inp=[]
        batch_classidx=[]
        batch_negidx=[]
        inp_lengths=[]
        classidx_lengths=[]
        negidx_lengths=[]
        for inp_seq, classidx_seq in zip(input_sequences,classidx_sequences):
            #make negatives?
            if alias is not None:
                classidx_set=set(classidx_seq)
                negs_set=set(alias.draw(len(classidx_set)*3))
                negs_set-=classidx_set
                negs=list(negs_set)
                random.shuffle(negs)
                negs=negs[:len(classidx_set)]
                negidx_seq=torch.tensor(negs,dtype=inp_seq.dtype)
            else:
                negidx_seq=torch.tensor([],dtype=inp_seq.dtype)
            #Should I yield before appending this one?
            if inp_lengths:
                inp_size=max(max(inp_lengths),inp_seq.shape[0])*(len(inp_lengths)+1)
                classidx_size=max(max(classidx_lengths),classidx_seq.shape[0])*(len(classidx_lengths)+1)
                negs_size=max(max(negidx_lengths),negidx_seq.shape[0])*(len(negidx_lengths)+1)
                if inp_size+classidx_size > batchsize: #appending this one would blow our batchsize, do yield first!
                    #time to yield!
                    batch_inputs_padded=torch.nn.utils.rnn.pad_sequence(batch_inp,batch_first=True)
                    batch_classidx_padded=torch.nn.utils.rnn.pad_sequence(batch_classidx,batch_first=True)
                    batch_negidx_padded=torch.nn.utils.rnn.pad_sequence(batch_negidx,batch_first=True)
                    yield batch_inputs_padded, batch_classidx_padded, batch_negidx_padded
                    batch_inp=[]
                    batch_classidx=[]
                    batch_negidx=[]
                    inp_lengths=[]
                    classidx_lengths=[]
                    negidx_lengths=[]
            batch_inp.append(inp_seq)
            batch_classidx.append(classidx_seq)
            batch_negidx.append(negidx_seq)
            inp_lengths.append(inp_seq.shape[0])
            classidx_lengths.append(classidx_seq.shape[0])
            negidx_lengths.append(negidx_seq.shape[0])
            #How much of data we've gathered for this batch?
        else:
            if inp_lengths:
                batch_inputs_padded=torch.nn.utils.rnn.pad_sequence(batch_inp,batch_first=True)
                batch_classidx_padded=torch.nn.utils.rnn.pad_sequence(batch_classidx,batch_first=True)
                batch_negidx_padded=torch.nn.utils.rnn.pad_sequence(batch_negidx,batch_first=True)
                yield batch_inputs_padded, batch_classidx_padded, batch_negidx_padded


if __name__=="__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-class-stats-file",help="Produce class stats file from --prep-data-in. Default %(default)s")
    parser.add_argument("--class-stats-file",default="labels-ctrl.json",help="Class labels file. Default %(default)s")
    parser.add_argument("--bert-tokenizer",default="pbert-v1",help="Which BERT model tokenizer to load? Default %(default)s")
    parser.add_argument("--prep-data-in",nargs="+",help="List of .txt.gz files with the data in tab-sep format. Will be pre-tokenized and saved into torch bin format.")
    parser.add_argument("--prep-data-out",help="Directory to which the prepped data will be chached, use with --prep-data-in")
    parser.add_argument("--max-labels",type=int,default=1000,help="How many most-common labels to use? Default: %(default)d")
    args=parser.parse_args()

    # for x,y in tqdm.tqdm(yield_batched(args.prep_data_out,10000,max_epochs=100)):
    #     pass
    # sys.exit()

    if args.make_class_stats_file:
        def yield_classes():
            for fname in args.prep_data_in:
                with gzip.open(fname,"rt",encoding="ASCII") as f:
                    for line in tqdm.tqdm(f):
                        cols=line.split("\t")
                        assert len(cols)==2
                        yield from cols[0].split(",")
        class_counter=collections.Counter(yield_classes())
        with open(args.make_class_stats_file,"wt",encoding="ASCII") as f:
            json.dump(dict(class_counter.items()),f)
            
                        
    if args.prep_data_out:
        bert_tokenizer= transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)
        bert_tokenizer_trie=trie_tokenizer.TrieTokenizer(transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"][args.bert_tokenizer])
        os.makedirs(args.prep_data_out,exist_ok=True)
        for fname in args.prep_data_in:
            print("********** Preprocessing",fname,file=sys.stderr)
            with gzip.open(fname,"rt",encoding="ASCII") as f:
                inputs_padded, classidx_padded=prep_ctrl_data(f,bert_tokenizer,bert_tokenizer_trie,args.class_stats_file,args.max_labels)
                basename=os.path.basename(fname).replace(".txt.gz",".torchbin")
                torch.save((inputs_padded, classidx_padded), os.path.join(args.prep_data_out,basename))
            

    

