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
    alias=alias_multinomial.AliasMultinomial.from_class_stats(args.class_stats_file,args.max_labels,flat=True)
    
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



def get_predictions(preds, all_labels):

    predicted_labels=[]
    prediction_values=[]
    
    all_labels = torch.tensor(list(all_labels)).cuda()
    
    repeated = all_labels.repeat(preds.size()[0], 1)
        
    preds_filtered = torch.gather(preds, -1, repeated) # pick only labels used in training
        
    sorted_indices = torch.argsort(preds_filtered, descending=True)
        
    pred_values_sorted = torch.gather(preds_filtered, -1, sorted_indices) # prediction values sorted
    classes_sorted = torch.gather(repeated, -1, sorted_indices) # class numbers sorted

    for c_row, c_value in zip(classes_sorted, pred_values_sorted):
        predicted_labels.append(c_row.tolist())
        prediction_values.append(c_value.tolist())
        
    return predicted_labels, prediction_values


def train_batch(model, data, positives, negatives, optimizer, loss_margin=0.3, evaluate=False):
    if evaluate:
        model.eval()
        torch.set_grad_enabled(False)
    #else:
    #    model.train()
    #    torch.set_grad_enabled(True) # should be okay to remove this line...
    data=data.long()[:,:510]
    positives=positives.long()
    negatives=negatives.long()
    

    batch_in_c=data.cuda()
    batch_out_c=positives.cuda()
    batch_neg_c=negatives.cuda()

    optimizer.zero_grad()
    preds=model(batch_in_c)
    preds_pos=torch.gather(preds,-1,batch_out_c)
    preds_neg=torch.gather(preds,-1,batch_neg_c)

    diff=preds_pos.unsqueeze(-1).repeat(1,1,preds_neg.shape[1])-preds_neg.unsqueeze(1)-loss_margin
    X_mask=(batch_out_c!=0).float().unsqueeze(-1).repeat(1,1,preds_neg.shape[1])
    Y_mask=(batch_neg_c!=0).float().unsqueeze(1).repeat(1,preds_pos.shape[1],1)
    #diff=preds_pos-preds_neg-torch.full_like(preds_pos,loss_margin)
    mask=X_mask*Y_mask
    loss=-(torch.sum(torch.min(diff*X_mask*Y_mask,torch.zeros_like(diff)))/mask.sum())
    if evaluate:
        # calculate fscore
        all_labels = set(positives[0].tolist()) | set(negatives[0].tolist()) - set([0])
        predicted_labels, prediction_values = get_predictions(preds, all_labels)
        model.train()
        torch.set_grad_enabled(True) # pre-set this for the next loop
    else:
        loss.backward()
        optimizer.step()
        predicted_labels=None
    loss_value=loss.item()
    return loss_value, predicted_labels
    
    
#def eval_batch(model, data, positives, negatives, optimizer,loss_margin=0.3):
#    torch.set_grad_enabled(False)

#    data=data.long()[:,:510]
#    positives=positives.long()[:,:510]
#    negatives=negatives.long()[:,:510]
    
#    batch_in_c=data.cuda()
#    batch_out_c=positives.cuda()
#    batch_neg_c=negatives.cuda()
#    mask=(batch_out_c!=0).type(torch.float)
#    non_zeros_in_batch = (mask).sum()



#    optimizer.zero_grad()
#    preds=model(batch_in_c)
#    preds_pos=torch.gather(preds,-1,batch_out_c)
#    preds_neg=torch.gather(preds,-1,batch_neg_c)
#    diff=preds_pos-preds_neg-torch.full_like(preds_pos,loss_margin)
#    loss=-(torch.sum(torch.min(diff*mask,torch.zeros_like(diff)))/non_zeros_in_batch)
    
#    loss_value=loss.item()
#    del batch_in_c,batch_out_c,batch_neg_c,preds_pos,preds_neg,diff,loss,preds
#    return loss_value
    
    
def train(args):
    #class_stats=json.load(open(args.class_stats_file)) #this is a dict: label -> count
    idx2label,label2idx,class_stats=data.prep_class_stats(args.class_stats_file, args.max_labels)
    filtered_labels=set(k for k,v in class_stats.most_common(args.max_labels))
    all_label_indices = torch.tensor([ label2idx[l] for l in filtered_labels ]) # TODO zero class!
    
    gold_labels = read_gold_labels("data/devel.txt.gz")

    os.makedirs(args.store_cpoint,exist_ok=True)
    #Do we load from checkpoint?
    if args.from_cpoint:
        model,d=tml.MlabelSimple.from_cpoint(args.from_cpoint)
        model=model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9)
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
        optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9)    
        it_counter=0
    model.train()
    torch.set_grad_enabled(True)
    
    
    
    # TODO dev data
    with open(args.dev, "rb") as f:
        input_sequences, classidx_sequences=torch.load(f)
        input_sequences = input_sequences[:1000]
        classidx_sequences = classidx_sequences[:1000]
    #dev_iter = data.yield_batched((input_sequences, classidx_sequences), args.batch_elements, shuffle=False, max_epochs=1, all_class_indices=set(all_label_indices))
    
    train_losses = []
    for batch_in,batch_out,batch_neg in tqdm.tqdm(data.yield_batched(args.train,args.batch_elements,max_epochs=100,all_class_indices=set(all_label_indices))):
        
        train_loss, _ = train_batch(model, batch_in, batch_out, batch_neg, optimizer)
        train_losses.append(train_loss)

        if it_counter%args.report_every==0:
            print("IT",it_counter,"TRAIN_LOSS", sum(train_losses)/len(train_losses), flush=True)
            train_losses = []
            # DEV
            dev_losses = []
            predictions = []
            for dev_data, dev_pos, dev_neg in data.yield_batched((input_sequences, classidx_sequences), args.batch_elements, shuffle=False, max_epochs=1, all_class_indices=set(all_label_indices)):
                dev_loss, predicted_labels = train_batch(model, dev_data, dev_pos, dev_neg, optimizer, evaluate=True)
                predictions+=predicted_labels
                dev_losses.append(dev_loss)
            print("DEV_LOSS", sum(dev_losses)/len(dev_losses),flush=True)
            calculate_fscore(gold_labels, predictions, idx2label, threshold=50)
            
        
        if it_counter%10000==0:
            print("Saving model", it_counter, file=sys.stderr,flush=True)
            model.save(os.path.join(args.store_cpoint,"model_{:09}.torch".format(it_counter)),{"optimizer_state_dict":optimizer.state_dict(), "it_counter":it_counter})
        it_counter+=1


def read_gold_labels(fname):

    # read gold labels
    gold_labels = []
    import gzip
    with gzip.open(fname, "rt", encoding="utf-8") as f:
        for line in f:
            gold_labels.append([])
            line=line.strip()
            classes, _ = line.split("\t", 1)
            for c in classes.split(","):
                gold_labels[-1].append(c)
                
    return gold_labels
        
def predict(args):
    class_stats=json.load(open(args.class_stats_file)) #this is a dict: label -> count

    idx2label,label2idx,class_stats=data.prep_class_stats(args.class_stats_file, args.max_labels)
    filtered_labels=set(k for k,v in class_stats.most_common(args.max_labels))
    label_indices = torch.tensor([ label2idx[l] for l in filtered_labels ]).cuda() # TODO zero class!
    
    print("filtered labels:", type(label_indices), label_indices.size())


    #Do we load from checkpoint?
    if not args.from_cpoint:
        print("Provide a checkpoint!", file=sys.stderr)
        sys.exit(0)
    model,d=tml.MlabelSimple.from_cpoint(args.from_cpoint)
    model.eval()
    model=model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9) # TODO: do we need this?
    if d.get("optimizer_state_dict"):
        optimizer.load_state_dict(d["optimizer_state_dict"])
    torch.set_grad_enabled(False)
                
        
    it_counter = 0


    losses=[]
    predicted_labels=[]
    # prediction_values=[] # TODO return also these
    for batch_in, batch_out, batch_neg in tqdm.tqdm(data.yield_batched(args.dev,args.batch_elements, max_epochs=1, alias=alias, shuffle=False)):
    
        loss, labels = train_batch(model, batch_in, batch_out, batch_neg, optimizer, evaluate=True)
        losses.append(loss)
        predicted_labels+=labels
        
        it_counter+=1

        if it_counter>=100:
            break

    print("DEV LOSS:", (torch.tensor(losses).sum()/torch.tensor(len(losses))).item())
    
    # read gold labels
    gold_labels = read_gold_labels(args.dev)
    for t in [1, 10, 25, 50, 100, 200, 300, 400, 500, 0]:
    #for t in [1, 10]:
        print("threshold =", t)
        calculate_fscore(gold_labels, predicted_labels, idx2label, threshold=t)
        print()
        
        
    # print top labels for first five example
    
    #for i,labels in enumerate(predicted_labels[:5]):
    #    for l in labels[:20]:
    #        per = round((class_stats[idx2label[l]]/sum(class_stats.values()))*100, 4)
    #        print("example",i, "label",idx2label[l],per,"%")
    #    print("---------")
        
    #for i,labels in enumerate(gold_labels[:5]):
    #    for l in labels[:20]:
    #        per = round((class_stats[l]/sum(class_stats.values()))*100, 4)
    #        print("gold example",i, "label",l,per,"%")
    #    print("---------")



def calculate_fscore(gold, pred, index2label, threshold=0):

    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer
    

    pred_labels = []
    for example in pred:
        if threshold!=0:
            example = example[:min(threshold,len(example))]
        example_ = []
        for l in example:
            example_.append(index2label[l])
        pred_labels.append(example_)
    
    
    all_labels = list(set([l for row in gold for l in row]) | set([l for row in pred_labels for l in row]))
    binarizer = MultiLabelBinarizer()
    binarizer.fit([all_labels])
    
    gold_ = binarizer.transform(gold[:len(pred_labels)])
    pred_ = binarizer.transform(pred_labels)
    
    print("Pre/Rec/F-score:",precision_recall_fscore_support(gold_, pred_, average="micro"))

    
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
    parser.add_argument("--lrate",type=float,default=1.0,help="Learning rate. Default %(default)f")
    parser.add_argument("--batch-elements",type=int,default=5000,help="How many elements in a batch? (sum of minibatch matrix sizes, not sequence count). Increase if you have more GPU mem. Default %(default)d")
    parser.add_argument("--predict", action="store_true", default=False, help="Run Prediction")
    parser.add_argument("--report-every", type=int, default=100, help="Report train/devel loss after every X batches")
    args=parser.parse_args()

    with torch.cuda.device(args.gpu):
    
        if args.predict:
            print("Running prediction...", file=sys.stderr)
            predict(args)
        else:
            train(args)

    

