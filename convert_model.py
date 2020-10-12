import torch
from tqdm import tqdm


def model_to_dict(model_file):
    model = torch.load(model_file)
    torch.save(model.state_dict(), model_file + ".pt")

    return


if __name__ == '__main__':

    model_files = ["asdf"]

    for i in tqdm(range(len(model_files))):
        model_to_dict(model_files[i])
