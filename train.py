import transformers
import torch
import modelling_transformer_mlabel as tml
import data
import sys
import gc
import alias_multinomial
import tqdm
import json
import torch.optim as optim
import os

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased-v1"]={'do_lower_case': False}

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["pbert-v1"]="proteiinipertti-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["pbert-v1"]="proteiinipertti-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["pbert-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["pbert-v1"]={'do_lower_case': False}

def do_train(args):
    alias=alias_multinomial.AliasMultinomial.from_class_stats(args.class_stats_file,args.max_labels)
    
    encoder_model = transformers.BertModel.from_pretrained("pbert-v1",output_hidden_states=True)
    if torch.cuda.is_available():
        encoder_model = encoder_model.cuda()
    tokenizer = transformers.BertTokenizer.from_pretrained("pbert-v1")
    model=tml.MlabelDecoder.from_bert(encoder_model,"decoder_config.json")

    
    for batch_in,batch_out,batch_neg in tqdm.tqdm(data.yield_batched(args.train,6000,max_epochs=1,alias=alias)):
        batch_in=batch_in.long()[:,:510]
        batch_out=batch_out.long()[:,:510]
        batch_neg=batch_neg.long()[:,:510]

        print("IN",batch_in)
        print("POS",batch_out)
        print("NEG",batch_neg)
        continue
        
        batch_in_c=batch_in.cuda()
        batch_out_c=batch_out.cuda()
        batch_neg_c=batch_neg.cuda()
        cls_in=torch.ones_like(batch_out_c)
        encoder_output,encoder_attention_mask,decoder_output_pos=model(batch_out_c,encoder_input=batch_in_c)

        _,_,decoder_output_neg=model(batch_neg_c,encoder_output=encoder_output,encoder_attention_mask=encoder_attention_mask)

        del batch_in_c,batch_out_c,batch_neg_c,encoder_output,decoder_output_pos,decoder_output_neg,encoder_attention_mask

def do_train_simple(args):
    class_stats=json.load(open(args.class_stats_file)) #this is a dict: label -> count
    alias=alias_multinomial.AliasMultinomial.from_class_stats(args.class_stats_file,args.max_labels)


    #Do we load from checkpoint?
    if args.from_cpoint:
        model,d=tml.MlabelSimple.from_cpoint(args.from_cpoint)
        model=model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        if d.get("optimizer_state_dict"):
            optimizer.load_state_dict(d["optimizer_state_dict"])
        it_counter=d.get("it_counter",0)
    else:
        #start from fresh
        os.makedirs(args.store_cpoint,exist_ok=True)
        encoder_model = transformers.BertModel.from_pretrained("pbert-v1",output_hidden_states=False)
        encoder_model = encoder_model.cuda()
        model=tml.MlabelSimple(encoder_model,len(class_stats))
        model=model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)    
        it_counter=0

    for batch_in,batch_out,batch_neg in tqdm.tqdm(data.yield_batched(args.train,args.batch_elements,max_epochs=100,alias=alias)):
        batch_in=batch_in.long()[:,:510]
        batch_out=batch_out.long()[:,:510]
        batch_neg=batch_neg.long()[:,:510]

        batch_in_c=batch_in.cuda()
        batch_out_c=batch_out.cuda()
        batch_neg_c=batch_neg.cuda()

        optimizer.zero_grad()
        preds=model(batch_in_c)
        preds_pos=torch.gather(preds,-1,batch_out_c)
        preds_neg=torch.gather(preds,-1,batch_neg_c)
        diff=preds_pos-preds_neg
        loss=-torch.mean(torch.min(diff,torch.zeros_like(diff)))
        loss.backward()
        optimizer.step()
        print("loss",loss.item(),flush=True)

        del batch_in_c,batch_out_c,batch_neg_c,preds_pos,preds_neg,diff,loss,preds
        
        if it_counter%500==0:
            model.save(os.path.join(args.store_cpoint,"model_{:09}.torch".format(it_counter)),{"optimizer_state_dict":optimizer.state_dict(), "it_counter":it_counter})
        it_counter+=1
        
                       
        



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",help=".torchbin file with the training data")
    parser.add_argument("--dev",help=".torchbin file with the dev data")
    parser.add_argument("--gpu",type=int,default=0, help="ID of the GPU to use. Set to -1 for 'I don't care'. Default %(default)d")
    parser.add_argument("--max-labels",type=int,default=1000, help="Max number of most common labels to use. Default %(default)d")
    parser.add_argument("--class-stats-file",help="Class stats file. Default %(default)s")
    parser.add_argument("--from-cpoint",help="Filename of checkpoint")
    parser.add_argument("--store-cpoint",help="Directory for checkpoints")
    parser.add_argument("--lr",type=float,default=1.0,help="Learning rate. Default %(default)f")
    parser.add_argument("--batch-elements",type=int,default=5000,help="How many elements in a batch? (sum of minibatch matrix sizes, not sequence count). Increase if you have more GPU mem. Default %(default)d")
    args=parser.parse_args()

    
    
    with torch.cuda.device(args.gpu):
        do_train_simple(args)

    

