import dataset
import model

if __name__ == '__main__':
    dataset = dataset.Dataset(train_folder='../data/train', val_folder='../data/val', test_folder='../data/test',
                              actions_file='../data/dialog_acts.json')

    model.main(dataset)


