.POSIX: #portable operating system interface - standards to maintain compatibility betweeen operating systems

NAME = revistral
PYTHON_VERSION = 3.12.2

POETRY = python -m poetry
PYTEST = python -m pytest

# requirements:
#     $(POETRY) export --format requirements.txt --output requirements.txt --without-hashes

# help:
#     @printf "activate \n\
#     TODO \n"

# Run this to create pyenv and initiate
pyenv:
	@echo "Installing Python version..."
	zsh -c 'source ~/.zshrc && pyenv install -s $(PYTHON_VERSION)'
	@echo "Creating virtual environment..."
	zsh -c 'source ~/.zshrc && pyenv virtualenv -f $(PYTHON_VERSION) $(NAME)'
	@echo "Setting local pyenv version..."
	zsh -c 'source ~/.zshrc && pyenv local $(NAME)'
	@echo "Activating virtual environment..."
	zsh -c 'source ~/.zshrc && pyenv activate $(NAME)'
	@echo "Checking which python..."
	zsh -c 'source ~/.zshrc && pyenv which python'
	@echo "Checking Python version..."
	zsh -c 'source ~/.zshrc && python --version'
	@echo "Checking pip version..."
	zsh -c 'source ~/.zshrc && pip --version'
	@echo "Installing poetry..."
	zsh -c 'source ~/.zshrc && pip install poetry'
	@echo "Running poetry bootstrap..."
	zsh -c 'source ~/.zshrc && make poetry-bootstrap'
	zsh -c 'source ~/.zshrc && $(POETRY) install'

activate:
	pyenv activate $(NAME)
	
# Initialize poetry project and add initial dependencies
poetry-bootstrap:
	@poetry init --no-interaction --name $(NAME) --dependency pandas --dependency SQLAlchemy --dependency psycopg2-binary --dependency langchain

remove-pyenv:
	pyenv uninstall $(NAME)

lint:
	python -m black -l 200 .
	python -m isort --profile black $(NAME)
	python -m flake8 --ignore E501,F401,F403,F405 .



test:
	$(PYTEST) -vs -vv

dev:
	$(PYTEST) -vs -vv -k 'test_dev'

