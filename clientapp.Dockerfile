FROM flwr/clientapp:1.15.2

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

# get dataset
COPY src/datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv .

# Run the entry`int command within the pipenv environment
ENTRYPOINT ["pipenv", "run", "flwr-clientapp"]
 