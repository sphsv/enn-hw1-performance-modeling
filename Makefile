PYTHON ?= python3

.PHONY: test smoke measure calibrate plots

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) measure.py --smoke --skip-energy --overwrite

measure:
	$(PYTHON) measure.py

calibrate:
	$(PYTHON) calibrate.py

plots:
	$(PYTHON) plots.py
