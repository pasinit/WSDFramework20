if [[ $# -lt 1 ]]; then
	echo "you shall pass the path to python3 (/usr/bin/python3)"
	exit 1
fi

git clone git@github.com:pasinit/WSDFramework20.git
git clone git@github.com:tommy9114/nlp_resources.git
cd WSDFramework20
conda create --name wsd_framework python=3.7
conda activate wsd_framework
cd ../nlp_resources
pip install . 
cd ../WSDFramework20
pip3 install -r requirements.txt

echo '[!!!!!!] Now download data from https://drive.google.com/open?id=12QgZ0LtXGIbsuYTeZF8DPWfdSiuHoCMz [!!!!!]'
