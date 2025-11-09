# Makefile for Medical X-ray Triage Project

.PHONY: help setup train eval interpret ui docs clean test

help: ## Show this help message
	@echo "Medical X-ray Triage Project - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup environment and generate sample data
	@echo "Setting up project..."
	python setup.py
	pip install -r requirements.txt

train: ## Train model with default parameters
	@echo "Training model..."
	python -m src.train --epochs 1

train-full: ## Train model with full epochs
	@echo "Training model with full epochs..."
	python -m src.train --epochs 10

eval: ## Evaluate trained model
	@echo "Evaluating model..."
	python -m src.eval

interpret: ## Generate Grad-CAM visualizations
	@echo "Generating Grad-CAM visualizations..."
	python -m src.interpret

ui: ## Launch Streamlit UI
	@echo "Launching Streamlit UI..."
	streamlit run ui/app.py

docs: ## Generate documentation diagrams
	@echo "Generating documentation..."
	python docs/make_docs_art.py

blueprint-pdf: ## Generate blueprint PDF from markdown
	@echo "Generating blueprint PDF..."
	@which pandoc > /dev/null || (echo "Error: pandoc not found. Please install pandoc." && exit 1)
	pandoc reports/blueprint.md -o reports/blueprint.pdf --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt
	@echo "Blueprint PDF generated: reports/blueprint.pdf"

blueprint-html: ## Generate blueprint HTML from markdown
	@echo "Generating blueprint HTML..."
	@which pandoc > /dev/null || (echo "Error: pandoc not found. Please install pandoc." && exit 1)
	pandoc reports/blueprint.md -o reports/blueprint.html --standalone --css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css
	@echo "Blueprint HTML generated: reports/blueprint.html"

test: ## Run quick test (1 epoch training + evaluation)
	@echo "Running quick test..."
	python src/make_sample_data.py
	python -m src.train --epochs 1
	python -m src.eval
	@echo "Test completed!"

clean: ## Clean generated files
	@echo "Cleaning generated files..."
	rm -rf results/*.pt results/*.json results/*.png results/*.yaml
	rm -rf data/sample/images/*.png data/sample/labels.csv
	rm -f docs/architecture.png docs/wireframe.png
	@echo "Clean completed!"

install-deps: ## Install dependencies
	@echo "Installing dependencies..."
	pip install -r requirements.txt

install-conda: ## Install dependencies with conda
	@echo "Installing dependencies with conda..."
	conda env create -f environment.yml
	@echo "Please run: conda activate medxray"

full-pipeline: ## Run complete pipeline
	@echo "Running complete pipeline..."
	python src/make_sample_data.py
	python -m src.train --epochs 5
	python -m src.eval
	python -m src.interpret
	@echo "Complete pipeline finished!"

# Default target
.DEFAULT_GOAL := help


