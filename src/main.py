from classification import main, parser

if __name__ == '__main__':
    main(parser.parse_args([
        "--save_folder", f"/home/safar/HCN/models/classification_model_electra",
        # "--save_folder", "../models/classification_model",
        "--epochs", "30",
        "--train"
    ]))
