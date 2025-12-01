# Build image
.PHONY: build
build:
	docker build -t ui-rl/simple-data-entry .

# Run locally 
.PHONY: run
run:
	docker run --rm -it -p 5900:5900 -p 8000:8000 --name simple-data-entry ui-rl/simple-data-entry
