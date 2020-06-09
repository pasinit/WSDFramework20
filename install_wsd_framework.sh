if [[ $# -lt 1 ]]; then
	echo "you shall pass the path to python3 (/usr/bin/python3)"
	exit 1
fi

git clone git@github.com:pasinit/WSDFramework20.git
git clone git@github.com:tommy9114/nlp_resources.git
cd WSDFramework20
virtualenv wsd_framework_venv -p $1
source wsd_framework_venv/bin/activate
cd ../nlp_resources
pip install . 
cd ../WSDFramework20
pip3 install -r requirements.txt

#echo '[!!!!!!] Now download data from https://drive.google.com/open?id=12QgZ0LtXGIbsuYTeZF8DPWfdSiuHoCMz [!!!!!]'
