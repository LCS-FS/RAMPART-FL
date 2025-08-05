# RAMPART-FL

This project demonstrates a federated reinforcement learning approach for anomaly detection in network traffic data. It uses Flower for the federated learning framework and a GAN-based model for anomaly detection.

## Setup and Installation

### Prerequisites

*   Docker and Docker Compose
*   Python 3.12+ and pipenv

### Generating Docker Compose Files

This project includes a script to generate `docker-compose.yaml` files for different numbers of clients.

First, install dependencies:
```bash
pipenv install
```

Then, run the `generate_compose.py` script with the desired number of clients. You can specify multiple numbers to generate multiple compose files.

For example, to generate a configuration for 3 clients:
```bash
python generate_compose.py 3
```
This will create a `docker-compose-3.yaml` file.

To generate files for 5, 10, and 20 clients at once:
```bash
python generate_compose.py 5 10 20
```
This will create `docker-compose-5.yaml`, `docker-compose-10.yaml`, and `docker-compose-20.yaml`.

The script also assigns CPU profiles to clients to simulate a heterogeneous environment.

## Running the Simulation

1.  **Start the Docker containers:**

    Use the `docker-compose` command with the file generated for the desired number of clients. For example, to run with 3 clients:
    ```bash
    docker-compose -f docker-compose-3.yaml up -d
    ```

2.  **Access the pipenv environment:**

    ```bash
    pipenv shell
    ```

3.  **Run the Flower simulation:**

    ```bash
    flwr run . local-deployment --stream
    ```

    This will start the federated learning simulation. You should see the output of the server and clients in your terminal.

## Visualizing the Results

The `visualization/plot_rl_metrics.py` script can be used to generate plots from the reinforcement learning training data. The script reads the `rl_training_data.csv` file, which is generated during the simulation and located in the `logs` directory.

To run the script, use the following command:

```bash
python visualization/plot_rl_metrics.py --log_dir <path_to_log_dir>
```

Replace `<path_to_log_dir>` with the path to the directory containing the `rl_training_data.csv` file (e.g., `logs/<timestamp>`).

The script will generate several plots in the `plots/rl_analysis` directory, including:

*   Global model performance over server rounds
*   RL agent rewards and components over server rounds
*   Q-value evolution for top states

These plots can be used to analyze the performance of the reinforcement learning agent and the federated learning system. 