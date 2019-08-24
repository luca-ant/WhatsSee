# WhatsSee
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
sudo apt install python3.7-setuptools
sudo apt install python3.7-pip
sudo apt install python3.7-venv
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
python3.7 -m venv venv
source venv/bin/activate

python3.7 -m pip install -r requirements.txt
```

* Run the python script as:

```
python whats_see.py train dataset=flickr 

```
```
python whats_see.py predict filename=your_image_file 
```

