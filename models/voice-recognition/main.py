import argparse
import gc
import os
import pickle
from glob import glob
from pathlib import Path

import nsml
import pandas as pd
import torch
import wandb
# from nsml import DATASET_PATH
from torch import nn
from torch.utils.data import DataLoader

from data import CustomTokenizer, CustomDataset
from stt_model import Transformer

print('torch version: ', torch.__version__)


def evaluate(model, imgs):
    model.to(device)
    # as the target is english, the first word to the transformer should be the
    # english start token.
    tokenizer = dict_for_infer['tokenizer']
    decoder_input = torch.tensor([tokenizer.txt2idx['<sos>']] * imgs.size(0), dtype=torch.long).to(device)
    output = decoder_input.unsqueeze(1).to(device)
    enc_output = None
    for i in range(max_length + 1):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        with torch.no_grad():
            # predictions, attention_weights, enc_output = transformer([imgs, output, enc_output])
            predictions, attention_weights, enc_output = model([imgs, output, enc_output])
        # select the last token from the seq_len dimension
        predictions_ = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = torch.tensor(torch.argmax(predictions_, axis=-1), dtype=torch.int32)

        output = torch.cat([output, predicted_id], dim=-1)
    output = output.cpu().numpy()

    result_list = []
    token_list = []
    for token in output:
        summary = tokenizer.convert(token)
        result_list.append(summary)
        token_list.append(token)

    return result_list, token_list


def train_step(batch_item, training):
    src = batch_item['magnitude'].to(device)
    tar = batch_item['target'].to(device)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    if training is True:
        # transformer.train()
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        return loss, acc, round(lr, 10)
    else:
        # transformer.eval()
        model.eval()
        with torch.no_grad():
            output, _, _ = model([src, tar_inp, None])
            # output, _, _ = transformer([src, tar_inp, None])
            loss = loss_function(tar_real, output)
        acc = accuracy_function(tar_real, output)
        return loss, acc


def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = criterion(pred.permute(0, 2, 1), real)
    mask = torch.tensor(mask, dtype=loss_.dtype)
    loss_ = mask * loss_

    return torch.sum(loss_) / torch.sum(mask)


def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, 0))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)

    return torch.sum(accuracies) / torch.sum(mask)

def glob_files(folder, file_type='*'):
    search_string = os.path.join(folder, file_type)
    files = glob(search_string)

    print('Searching files ', search_string)
    paths = []
    for f in files:
      if os.path.isdir(f):
        sub_paths = glob_files(f + '/')
        paths += sub_paths
      else:
        paths.append(f)

    # We sort the images in alphabetical order to match them
    #  to the annotation files
    paths.sort()

    return paths


def glob_folders(folder, file_type='*'):
    search_string = os.path.join(folder, file_type)
    files = glob(search_string)

    print('Searching folders {} {}'.format(search_string, files))
    paths = []
    for f in files:
      if os.path.isdir(f):
        paths.append(f)

    # We sort the images in alphabetical order to match them
    #  to the annotation files
    paths.sort()

    return paths


def glob_files_all(folder, file_type='*'):
    print("Searching in {} from {}".format(folder, os.getcwd()))
    sub_folders = glob_folders(folder)
    print("Found {} sub folders".format(len(sub_folders)))

    files = []
    for sub_folder in sub_folders:
        tmp_files = glob_files(sub_folder, file_type)
        if tmp_files:
            files.extend(tmp_files)
    print("Found {} files".format(len(files)))
    return files


def path_loader(root_path, divide_id=8000, use_column=1, is_test=False):
    print("root_path: {} divide_id: {} use_column: {} is_test: {}".format(root_path, divide_id, use_column, is_test))

    if is_test:
        file_list = sorted(glob(os.path.join(root_path, 'test_data', '*')))

        return file_list

    if args.mode == 'train':
        train_path = root_path
        # os.path.join(root_path, 'train')
        file_list = sorted(glob_files_all(os.path.join(train_path, 'train_data', '')))
        # file_list = sorted(glob_files_all(os.path.join(train_path, 'train_lite', '')))
        print("Data files loaded {}".format(len(file_list)))
        # file_list = sorted(glob(os.path.join(train_path, 'train_data', '*')))
        # label = pd.read_csv(os.path.join(train_path, 'labels_tw_lite.txt'))
        label = pd.read_csv(os.path.join(train_path, 'train_label.txt'))
        # label = pd.read_csv(os.path.join(train_path, 'labels_ai-hub_tw_shuffle.txt'))
        print("Loaded label {}".format(len(label)))

        file_dict = dict()
        for full_file_path in file_list:
            filename = Path(os.path.basename(full_file_path)).stem
            if file_dict.get(filename.lower()):
                print("ERROR: duplicate file name found {}".format(filename))
            else:
                file_dict[filename.lower()] = full_file_path

        print("file_dict {}".format(len(file_dict)))
        for id, (key, val) in enumerate(file_dict.items()):
            print("{} {}".format(key, val))
            if id > 3:
                break

        file_list_selected = []
        data_file_paths = label.iloc[:, 0]
        for data_file_path in data_file_paths:
            data_filename = Path(os.path.basename(data_file_path)).stem
            data_filename = data_filename.replace("[\"", "")
            if file_dict.get(data_filename.lower()):
                file_list_selected.append(file_dict[data_filename.lower()])
            else:
                print("ERROR: file not found {}".format(data_filename))

    return file_list_selected[:divide_id], label[:divide_id]


def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        save_checkpoint(dict_for_infer, save_dir)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):
        save_dir = os.path.join(dir_name, 'checkpoint')

        global checkpoint
        checkpoint = torch.load(save_dir)

        model.load_state_dict(checkpoint['model'])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = checkpoint['device']
        test_file_list = path_loader(test_path, is_test=True)
        test_dataset = CustomDataset(test_file_list, None, 160000, 'test')
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=10)
        result_list = []

        for step, batch in enumerate(test_data_loader):
            inp = batch['magnitude'].to(device)
            output, _ = evaluate(model, inp)
            result_list.extend(output)

        prob = [1] * len(result_list)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--iteration', type=str, default='5')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--divide_id", action="store", dest="divide_id", type=int, default=50)
    parser.add_argument("--use_column", action="store", dest="use_column", type=int, default=1)
    args = parser.parse_args()

    max_length = 30
    batch_size = 32
    num_layers = 6
    d_model = 512
    dff = 2048
    num_heads = 8
    dropout_rate = 0.1
    epochs = args.epochs
    learning_rate = 5e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_vocab_size = 5000

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        DATASET_PATH = args.path_in
        # DATASET_PATH = "/Users/changsin/PycharmProjects/ModelTrain/data/senior_voice_commands/train"
        file_list, label = path_loader(DATASET_PATH, args.divide_id, args.use_column)

        print("Loaded files {} labels {}".format(len(file_list), len(label)))

        split_num = int(len(label) * 0.9)
        train_file_list = file_list[:split_num]
        val_file_list = file_list[split_num:]

        train_label = label.iloc[:split_num]
        val_label = label.iloc[split_num:]

        tokenizer = CustomTokenizer(max_length=max_length, max_vocab_size=max_vocab_size)
        tokenizer.fit(train_label.iloc[:, 1])

        target_size = len(tokenizer.txt2idx)

        train_tokens = tokenizer.txt2token(train_label.iloc[:, 1])
        val_tokens = tokenizer.txt2token(val_label.iloc[:, 1])
        train_dataset = CustomDataset(train_file_list, train_tokens)
        valid_dataset = CustomDataset(val_file_list, val_tokens)

        print("train_tokens: {} train_file_list: {}".format(len(train_tokens), len(train_file_list)))
        print("val_tokens: {} val_file_list: {}".format(len(val_tokens), len(val_file_list)))

        print("train_token samples: {} train_file samples: {}".format(train_label.iloc[:, 1][:3], train_file_list[:3]))
        print("val_token samples: {} val_file samples: {}".format(val_label.iloc[:, 1][:3], val_file_list[:3]))
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        model = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            target_size=max_vocab_size + 4,
            pe_target=max_length + 1,
            device=device,
            rate=dropout_rate
        )

        bind_model(model=model, parser=args)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        use_wandb = False
        if use_wandb:
            wandb.init("voice-recognition")

        for epoch in range(args.epochs):
            gc.collect()
            total_train_loss, total_valid_loss = 0, 0
            total_train_acc, total_valid_acc = 0, 0

            training = True
            avg_batch_loss, avg_batch_acc = 0, 0
            iterations = 0
            for batch in train_dataloader:
                batch = batch
                batch_loss, batch_acc, lr = train_step(batch, training)
                total_train_loss += batch_loss
                total_train_acc += batch_acc
                if use_wandb:
                    wandb.log({"train_batch_acc": batch_acc, "train_batch_loss":batch_loss})

                avg_batch_loss += batch_loss
                avg_batch_acc += batch_acc
                iterations += 1

            if use_wandb:
                wandb.log({
                    "avg_train_batch_acc": avg_batch_acc / float(iterations),
                    "avg_train_batch_loss": avg_batch_loss / float(iterations)})

            print(f'avg_train_batch_acc: {avg_batch_acc / float(iterations)}')
            print(f'avg_train_batch_loss: {avg_batch_acc / float(iterations)}')

            training = False
            iterations = 0
            avg_batch_loss, avg_batch_acc = 0, 0
            for batch in valid_dataloader:
                batch = batch
                batch_loss, batch_acc = train_step(batch, training)
                total_valid_loss += batch_loss
                total_valid_acc += batch_acc
                if use_wandb:
                    wandb.log({"valid_batch_acc": batch_acc, "valid_batch_loss": batch_loss})

                avg_batch_loss += batch_loss
                avg_batch_acc += batch_acc
                iterations += 1

            if use_wandb:
                wandb.log({"avg_valid_batch_acc": avg_batch_acc/float(iterations),
                           "avg_valid_batch_loss": avg_batch_loss/float(iterations)})

            print(f'avg_valid_batch_acc: {avg_batch_acc / float(iterations)}')
            print(f'avg_valid_batch_loss: {avg_batch_acc / float(iterations)}')

            print('=================Epoch: {} Iterations: {}'.format(epoch, iterations))
            print(f'total_train_loss: {total_train_loss}')
            print(f'total_valid_loss: {total_valid_loss}')
            print(f'total_train_acc : {total_train_acc}')
            print(f'total_valid_acc : {total_valid_acc}')

            dict_for_infer = {
                'model': model.state_dict(),
                'max_length': max_length,
                'target_size': target_size,
                'num_layers': num_layers,
                'd_model': d_model,
                'dff': dff,
                'num_heads': num_heads,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'tokenizer': tokenizer,
                'device': device
            }

            if use_wandb:
                wandb.log({
                    "total_train_loss": total_train_loss, "total_valid_loss": total_valid_loss,
                    "total_train_acc": total_train_acc,   "total_valid_acc": total_valid_acc
                       })
            print("Writing to ./checkpoint{}".format(args.use_column))
            save_checkpoint(checkpoint=dict_for_infer, dir='./checkpoint'.format(args.use_column))
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
