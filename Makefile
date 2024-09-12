# set python path
PYTHON=python3

test: 
	$(PYTHON) test/runtests.py
.PHONY: test

install:
	$(PYTHON) -m pip install -r requirements.txt
.PHONY: install

