# MV-Attn

## Training:
 ```bash
 sh submit.sh
 ```
 
 ## Generation:
 ```bash
python -u ./thumt/bin/translator.py --input data/newstest2014.bpe.en --vocabulary data/vocab.bpe.en.txt data/vocab.bpe.de.txt --model add  --output $2 --checkpoints $1 --parameters device_list=[0],mode='all'
 ```
