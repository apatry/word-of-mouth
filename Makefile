# Setup a virtualenv for python and automate other tasks.
#
# The python setup is largely inspired from
# https://stackoverflow.com/a/46188210/3057917

VENV_ROOT = _env
VENV = . $(VENV_ROOT)/bin/activate;

venv: $(VENV_ROOT)/bin/activate

_env/bin/activate: requirements.txt
	test -d $(VENV) || virtualenv $(VENV)
	. $(VENV)/bin/activate; pip install -Ur requirements.txt
	touch $(VENV)/bin/activate

# launch emacs in our virtualenv
emacs: venv
	$(VENV) emacs

# run unit tests and type checking
test: venv
	$(VENV) mypy wordofmouth tests
	$(VENV) python -m unittest discover

# download datasets
bands.csv:
	$(VENV) python -m wordofmouth download-bands-dataset bands.csv || rm -f bands.csv

# train models
codec.bin model.bin &: bands.csv
	$(VENV) python -m wordofmouth bandaid-train --training-dataset bands.csv --codec codec.bin --model model.bin --weights model-weights

bandaid-train: model.bin codec.bin

bandaid: bandaid-train
	$(VENV) python -m wordofmouth bandaid --codec codec.bin --model model.bin --weights model-weights $(PREFIXES)
