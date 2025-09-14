PY?=python3

preprocess:
	$(PY) data/preprocess.py --config config.yaml --pcaps "$(shell yq '.preprocess.pcaps_glob' config.yaml)" --labels $(shell yq '.preprocess.labels_csv' config.yaml)

lgbm_teacher:
	$(PY) lgb_baseline.py --config config.yaml

train_fast:
	$(PY) train_fast.py --config config.yaml

eval_fast:
	$(PY) eval_fast.py --labels labels/labels.csv


train:
	$(PY) train.py --config config.yaml --pcaps "samples/*.pcap" --labels labels/labels.csv

infer:
	$(PY) infer.py --config config.yaml --pcaps "samples/*.pcap" --out reports/

eval:
	$(PY) eval.py --reports_dir reports/ --labels labels/labels.csv --out reports/

explain:
	$(PY) explain.py --config config.yaml --pcaps "samples/*.pcap" --labels labels/labels.csv --out reports/explain

clean:
	rm -rf runs/ artifacts/ reports/
