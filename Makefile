.PHONY: demo eval test hybrid
demo:   ; python -m ledgerlens demo
eval:   ; python -m ledgerlens eval --wild 0.4 --trace data/trace.jsonl
hybrid: ; python -m ledgerlens eval --wild 0.4 --reasoner hybrid --limit 40 --trace data/trace.jsonl
test:   ; python -m pytest -q
