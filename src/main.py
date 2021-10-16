import dataset

if __name__ == '__main__':
    # dataset.extract_data(dialogue_dir='../data/train', acts='../data/dialog_acts.json', name='train')
    # dataset.extract_data(dialogue_dir='../data/val', acts='../data/dialog_acts.json', name='val')
    # dataset.extract_data(dialogue_dir='../data/test', acts='../data/dialog_acts.json', name='test')

    dataset = dataset.Dataset(train_folder='../data/train', val_folder='../data/val', test_folder='../data/test',
                              actions_file='../data/dialog_acts.json')
    #for utt, act in zip(dataset.train.utterances[:1], dataset.train.actions[:1]):
    print(dataset.train.actions[0].shape)
    exit()
    utt, act = dataset.train.utterances[0], dataset.train.actions[0]
    print(utt, act)
    print(utt, dataset.ints2words(utt))
    print(act, dataset.ints2actions(act))
    print('=============================')
