# transformer-mlabel
could we get multilabel from transformer in some reasonable manner


python3 data.py --make-class-stats-file CAFA4-ctrl/class-stats.json --prep-data-in CAFA4-ctrl/train.txt.gz
python3 data.py --class-stats CAFA4-ctrl/class-stats.json --prep-data-in CAFA4-ctrl/*.txt.gz --prep-data-out CAFA4-ctrl --max-labels 5000
