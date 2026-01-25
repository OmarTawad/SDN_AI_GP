PY ?= python3
CONFIG ?= config.yaml

.PHONY: preprocess preprocess-50mb train infer eval explain clean

preprocess:
	$(PY) -m data.preprocess --config $(CONFIG)

preprocess-50mb:
	$(PY) -m data.preprocess --config $(CONFIG) --limit-mb 50

train:
	$(PY) train.py --config $(CONFIG)

infer:
	$(PY) infer.py --config $(CONFIG)

eval:
	$(PY) eval.py --config $(CONFIG)

explain:
	$(PY) explain.py --config $(CONFIG)

clean:
	rm -rf artifacts/ cache/ reports/
