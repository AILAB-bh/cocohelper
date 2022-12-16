import tomli

if __name__ == "__main__":
    with open("pyproject.toml", mode="rb") as fp:
        project = tomli.load(fp)
    version = project['tool']['poetry']['version']
    print(f"::set-output name=version::{version}")