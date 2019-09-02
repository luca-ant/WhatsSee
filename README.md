# WhatsSee

<p align="center">
  <img width=350px src="https://github.com/luca-ant/WhatsSee/blob/master/static/img/logo.png?raw=true">
</p>
WhatsSee is a simple and humble image captioning application, based on a neural network built with tensorflow. The back-end is written in **Python** and the Web GUI front-end is built with **Flask** framework.

## Getting started

* Clone repository
```
git clone https://github.com/luca-ant/WhatsSee.git
```
or
```
git clone git@github.com:luca-ant/WhatsSee.git
```


* Install dependencies
```
sudo apt install python3-setuptools
sudo apt install python3-pip
sudo apt install python3-venv
```
or
```
sudo pacman -S python-setuptools 
sudo pacman -S python-pip
sudo pacman -S python-virtualenv
```

* Create a virtual environment and install requirements modules
```
cd WhatsSee
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
```


## Running

* **Training:** To train the model. You can choose the dataset, the number of training and validation examples and number of epoch. (All arguments are optional) **Caution! Whole dataset will be downloaded!**

```
python whats_see.py train -d flickr -nt 6000 -nv 1000 -ne 50
```

* **Resume:** To resume last saved training and continue it.

```
python whats_see.py resume
```

* **Evaluate:** To evaluate whole model on test images and calculate **BLEU scores**. You can specify the number of test examples.

```
python whats_see.py evaluate -n 1000
```

* **Test:** To test the model by generating a caption of a test's image and compare the generated caption with the real ones.

```
python whats_see.py test -f TEST_IMAGE_FILE 
```


* **Generate:** To generate a caption of your own image.

```
python whats_see.py generate -f YOUR_IMAGE_FILE 
```




## Deployment
To deploy web aplication, simple run *start_server.sh* script. Open a browser and navigate to [localhost:4753](http://localhost:4753/).

```
./start_server.sh
```

## Result
A pre-trained model can be found on [releases page](https://github.com/luca-ant/WhatsSee/releases/latest)
The neural network was trained on training images of **Flickr dataset** [link](https://github.com/luca-ant/WhatsSee_dataset) and it achieved the following BLEU scores on test images:

* **BLEU-1: 49.3%**
* **BLEU-2: 30.5%**
* **BLEU-3: 21.7%**
* **BLEU-4: 11.1%**

### Examples


[![SCREEN](https://github.com/luca-ant/WhatsSee/blob/master/examples/1.png?raw=true)]()
[![SCREEN](https://github.com/luca-ant/WhatsSee/blob/master/examples/2.png?raw=true)]()
[![SCREEN](https://github.com/luca-ant/WhatsSee/blob/master/examples/3.png?raw=true)]()
[![SCREEN](https://github.com/luca-ant/WhatsSee/blob/master/examples/4.png?raw=true)]()

## Credits
* WhatsSee was developed by Luca Antognetti

