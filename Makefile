.PHONY: demo serve test check

demo:
	python3 -m assettone demo

serve:
	python3 -m assettone serve --port 8765

test:
	python3 -m unittest discover -s tests -v

check:
	python3 -m py_compile assettone/*.py
	node --check web/app.js
	python3 -m unittest discover -s tests -v
