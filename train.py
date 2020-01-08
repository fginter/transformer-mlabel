import transformers
import torch
import modelling_transformer_mlabel as tml

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased-v1"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased-v1/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased-v1"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased-v1"]={'do_lower_case': False}

encoder_model = transformers.BertModel.from_pretrained("bert-base-finnish-cased-v1",output_hidden_states=True)
if torch.cuda.is_available():
    encoder_model = encoder_model.cuda()
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased-v1")

model=tml.MlabelDecoder.from_bert(encoder_model,"decoder_config.json")

