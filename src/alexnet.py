from data import Data

project_root = "~/code/personal/alexnet"
data_path = project_root + "/data"
weights_path = project_root + "/weights"


def main():

    d = Data(data_path)

    train_loader, val_loader, test_loader = d.get_loaders()

    train_iterator = iter(train_loader)
    batch = next(train_iterator)

    print(len(batch))


if __name__ == "__main__":
    main()
