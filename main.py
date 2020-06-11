from config import data_params
from data_creator import DataCreator


def main():
    data = DataCreator(**data_params)


if __name__ == '__main__':
    main()

