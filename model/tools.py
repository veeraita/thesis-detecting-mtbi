import torch


def save_model(model):
    try:
        filename = input('Type the file name to save the model to ').lower()
        if filename:
            torch.save(model.state_dict(), filename)
            print('Model saved to %s.' % (filename))
        else:
            print('Model not saved.')
    except:
        raise Exception('The notebook should be run or validated with skip_training=True.')


def load_model(model, device):
    filename = input('Type the file name to load the model from ').lower()
    if filename:
        model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print('Model loaded from %s.' % filename)
        model.to(device)
        model.eval()
    else:
        print('Model not loaded.')


