# WSDFramework20
Codice e dati per il framework2.0 di WSD


# Install
- Download and install [Anaconda3](https://pytorch.org/get-started/locally/)
- ```conda create --name wsd_framework```
- ```conda activate wsd_framework```

If you have cuda 10.2 just type
- ```conda install pytorch torchvision -c pytorch```

Otherwise, refer to pytorch website to select the right version https://pytorch.org/get-started/locally/.
``` bash
cd WSDFramework20
pip install -r requirements.txt
pip install git+https://github.com/tommy9114/nlp_resources
wandb init
```
