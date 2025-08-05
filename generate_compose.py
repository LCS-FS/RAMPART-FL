import argparse
import sys
import yaml
import random
import os

BASE_SUPERNODE_PORT = 9094
CONTAINER_LOG_PATH = "/app/logs"

DEVICE_PROFILES = [
    {"name": "Low-End IoT", "cores_min": 1, "cores_max": 2, "description": "e.g., sensors, ARM Cortex-M"},
    {"name": "Mid-Range Edge", "cores_min": 2, "cores_max": 4, "description": "e.g., Raspberry Pi 4, Jetson Nano (CPU part)"},
    {"name": "High-End Edge CPU", "cores_min": 4, "cores_max": 8, "description": "e.g., Intel NUC, Powerful Edge Servers (CPU)"}
]

def get_client_config(client_index: int) -> tuple[str, int]:
    """Determines the device profile and CPU core count for a client."""
    profile_index = client_index % len(DEVICE_PROFILES) 
    profile = DEVICE_PROFILES[profile_index]
    cores = random.randint(profile["cores_min"], profile["cores_max"])
    return profile["name"], cores

def generate_compose_config(num_clients: int, logs_dir: str) -> dict:
    """Generates the docker-compose configuration dictionary (CPU-only)."""

    if num_clients <= 0:
        raise ValueError("Number of clients must be positive.")

    logs_dir_relative = os.path.relpath(logs_dir)

    config = {
        'version': '3.8',
        'services': {
            'superlink': {
                'image': 'flwr/superlink:1.15.2',
                'container_name': 'superlink',
                'command': ["--insecure", "--isolation", "process"],
                'ports': [
                    "9091:9091",
                    "9092:9092",
                    "9093:9093"
                ],
                'networks': ['flwr-network'],
                'restart': 'unless-stopped',
                'volumes': [
                    'logs:/app/logs:rw'
                ]
            },
            'serverapp': {
                'build': {
                    'context': '.',
                    'dockerfile': 'serverapp.Dockerfile'
                },
                'image': 'flwr_serverapp:0.0.1',
                'container_name': 'serverapp',
                'command': ["--insecure", "--serverappio-api-address", "superlink:9091"],
                'networks': ['flwr-network'],
                'restart': 'unless-stopped',
                'volumes': [
                    'logs:/app/logs:rw'
                ],
                'environment': {
                    'LOG_DIR': CONTAINER_LOG_PATH
                }
            }
        },
        'networks': {
            'flwr-network': {
                'driver': 'bridge'
            }
        },
        'volumes': {
            'logs': {
                'driver': 'local',
                'driver_opts': {
                    'type': 'none',
                    'o': 'bind',
                    'device': logs_dir_relative
                }
            }
        }
    }

    print(f"\nAssigning device profiles and CPU cores for {num_clients} clients (CPU-only):")
    for i in range(1, num_clients + 1):
        partition_id = i - 1
        supernode_port = BASE_SUPERNODE_PORT + partition_id
        supernode_name = f'supernode-{i}'
        clientapp_name = f'clientapp-{i}'

        config['services'][supernode_name] = {
            'image': 'flwr/supernode:1.15.2',
            'container_name': supernode_name,
            'command': [
                "--insecure",
                "--superlink", "superlink:9092",
                "--node-config", f"partition-id={partition_id} num-partitions={num_clients}",
                "--clientappio-api-address", f"0.0.0.0:{supernode_port}",
                "--isolation", "process"
            ],
            'ports': [f"{supernode_port}:{supernode_port}"],
            'networks': ['flwr-network'],
            'restart': 'unless-stopped',
            'volumes': [
                'logs:/app/logs:rw'
            ]
        }

        profile_name, client_cpus = get_client_config(i - 1)
        
        client_service_config = {
            'build': {
                'context': '.',
                'dockerfile': 'clientapp.Dockerfile'
            },
            'image': 'flwr_clientapp:0.0.1',
            'container_name': clientapp_name,
            'command': [
                "--insecure",
                "--clientappio-api-address", f"{supernode_name}:{supernode_port}"
            ],
            'networks': ['flwr-network'],
            'restart': 'unless-stopped',
            'cpus': str(float(client_cpus)), 
            'shm_size': '2gb',
            'mem_limit': '2000m',
            'environment': { 
                'ALLOCATED_CORES': str(client_cpus),
                'NUM_CLIENTS': str(num_clients),
                'LOG_DIR': CONTAINER_LOG_PATH,
                'PARTITION_ID': str(partition_id),
                'PROFILE_NAME': profile_name
            },
            'volumes': [
                'logs:/app/logs:rw'
            ]
        }
        
        print(f"  ClientApp-{i} ({clientapp_name}): Profile='{profile_name}', Cores='{client_cpus}' (CPU-only)")
        
        config['services'][clientapp_name] = client_service_config

    return config

def main():
    parser = argparse.ArgumentParser(
        description="Generate docker-compose.yaml for Flower simulation with N clients based on CPU device profiles."
    )
    parser.add_argument(
        "num_clients",
        type=int,
        nargs='+', 
        help="Number of client nodes to generate configuration for."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Directory to save the generated compose files (default: current directory)."
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory for Flower execution logs (default: logs)."
    )

    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("Error: PyYAML library not found. Please install it: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    
    random.seed(None) 

    os.makedirs(args.logs_dir, exist_ok=True)
    print(f"Logs directory: {args.logs_dir}")

    for num in args.num_clients:
        try:
            config_data = generate_compose_config(num, args.logs_dir)
            output_filename = f"{args.output_dir}/docker-compose-{num}.yaml"

            with open(output_filename, 'w') as f:
                yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, width=1000)

            print(f"\nSuccessfully generated {output_filename} for {num} clients (CPU-only).")
            print(f"Flower execution logs will be stored in: {args.logs_dir}")

        except ValueError as e:
            print(f"Error generating config for {num} clients: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred for {num} clients: {e}", file=sys.stderr)

if __name__ == "__main__":
    main() 