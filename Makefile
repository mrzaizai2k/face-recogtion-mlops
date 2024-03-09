install:
#	# python -m venv venv
#	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt
