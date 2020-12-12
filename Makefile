isort = isort optking
black = black -l 120 optking

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(isort) --check-only
	$(black) --check
