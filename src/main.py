import dataset

if __name__ == '__main__':
    dataset = dataset.Dataset(train_folder='../data/train', val_folder='../data/val', test_folder='../data/test',
                              actions_file='../data/dialog_acts.json')
    for utt, act in zip(dataset.train_utt[:2], dataset.train_act[:2]):
        print("Utt_int =", utt, "utt =", dataset.ints2words(utt), "act_int =", act, "act =", dataset.ints2actions(act))


