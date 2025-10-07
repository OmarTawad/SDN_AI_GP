PY ?= python3
CONFIG ?= config.yaml

.PHONY: preprocess train infer eval explain clean

preprocess:
	$(PY) -m data.preprocess --config $(CONFIG)

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
