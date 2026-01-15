.PHONY: publish test build clean

clean:
	rm -rf dist/

build: clean
	uv build

test:
	uv run pytest tests/

publish: build
	uv publish
