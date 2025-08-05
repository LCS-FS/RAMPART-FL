FROM flwr/serverapp:1.15.2

WORKDIR /app

# Copy dataset files
COPY src/datasets /app/src/datasets

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

# Install pipenv
RUN python -m pip install --no-cache-dir pipenv

# Copy Pipfile and install dependencies into pipenv virtualenv
COPY Pipfile .
RUN pipenv lock
RUN pipenv install --deploy

# Run the entrypoint command within the pipenv environment
ENTRYPOINT ["pipenv", "run", "flwr-serverapp"]
