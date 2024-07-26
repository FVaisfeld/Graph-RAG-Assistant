import yaml

class Config:
    @staticmethod
    def load_config(file_path="src/config.yaml"):
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config

    @staticmethod
    def save_config(config, file_path="src/config.yaml"):
        """Save configuration to a YAML file."""
        with open(file_path, 'w') as config_file:
            yaml.safe_dump(config, config_file)

# Example usage:
if __name__ == "__main__":
    config = Config.load_config()
    print(config)
