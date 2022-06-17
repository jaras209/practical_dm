from transformer_model import main, parser

if __name__ == '__main__':
    main(parser.parse_args([
        "--save_folder", f"/home/safar/HCN/models/taxi_transformer",
        "--epochs", "20",
        "--train"
    ]))
