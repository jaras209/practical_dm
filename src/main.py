from classification import main, parser

if __name__ == '__main__':
    main(parser.parse_args([
        # "--save_folder", f"/home/safar/HCN/models/classification_model_roberta_taxi",
        "--save_folder", f"/home/safar/HCN/models/classification_model_electra_taxi",
        # "--pretrained_model", 'roberta-base',
        "--pretrained_model", 'google/electra-base-discriminator',
        "--epochs", "100",
        "--train",
    ]))
