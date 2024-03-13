install:
#	# python -m venv venv (your default python)
# 	virtualenv venv -p python3.8 (for specific python/ must download version first)
#	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt

cam:
	python  src/webcam.py