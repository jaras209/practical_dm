from seq2seq_model import main, parser

if __name__ == '__main__':
    for hidden in [256]:
        for action_embed_size in [128]:
            for l_r in [1e-2]:
                print("================================================")
                print("================================================")
                print(f"TRAINING MODEL {hidden=}_{action_embed_size=}_{l_r=}")
                main(parser.parse_args([
                    "--save_folder", f"/home/safar/HCN/models/taxi_only_train_data_model_{hidden=}_{action_embed_size=}_{l_r=}",
                    "--epochs", "20",
                    "--hidden_size", f"{hidden}",
                    "--action_embedding_size", f"{action_embed_size}",
                    "--learning_rate", f"{l_r}",
                    "--train"
                ]))
