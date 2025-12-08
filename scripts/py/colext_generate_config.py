import argparse

from jinja2 import Template


"""
Simple script to create a colext config for a given experiment. It injects the fl_algorithm,
the data configuration, and any further configuration
"""

def generate():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-client-updates", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--model", required=True, type=str, choices=["set2set", "clmean", "te"])
    parser.add_argument("--num-partitions", required=True, type=int)
    parser.add_argument("--fraction-fit", required=True, type=float)
    parser.add_argument("--num-rounds", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    args = parser.parse_args()
    args_dict = vars(args)

    with open("./colext/training_conf_template.yaml", "r") as file:
        template = Template(file.read())

    rendered_content = template.render(args_dict)

    with open("./colext_config.yaml", "w") as file:
        file.write(rendered_content)

    print("YAML file generated successfully.")


if __name__ == "__main__":
    generate()
